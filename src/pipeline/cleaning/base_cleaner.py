from __future__ import annotations

"""
本文件只负责基础的PDF噪声信息去除, 至于进一步的数据处理, 会在source_cleaner实现
由具体的source决定清洗策略
"""

import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple


TEXT_FIELDS: Tuple[str, ...] = ("content_markdown", "content", "abstract")
MIN_TEXT_CHARS = 20

# Common mojibake observed in data/origin/*_origin.jsonl after PDF extraction.
MOJIBAKE_REPLACEMENTS = {
    "\u920d\u6a9a": "'s",
    "\u9225\u6a9a": "'s",
    "\u920d\u6a9b": "'t",
    "\u9225\u6a9b": "'t",
    "\u920d\u6a99": "'r",
    "\u9225\u6a99": "'r",
    "\u920d\u6a9d": "'v",
    "\u9225\u6a9d": "'v",
    "\u920d\u6a91": "'l",
    "\u9225\u6a91": "'l",
    "\u920d?": "'",
    "\u9225?": "'",
    "\u920d\ufffd": "'",
    "\u9225\ufffd": "'",
    "\u920d\u63f3": "-",
    "\u9225\u63f3": "-",
    "\u920d\u64dc": "-",
    "\u9225\u64dc": "-",
    "\u920d\u6dcf": '"',
    "\u9225\u6dcf": '"',
    "\u920d\u6dd0": '"',
    "\u9225\u6dd0": '"',
    "\u920d\u64e5": "",
    "\u9225\u64e5": "",
    "\u920d\u7797": "...",
    "\u9225\u7797": "...",
    "\u9218\u6a9a": "'s",
    "\u9218?": "'",
    "\u9218\ufffd": "'",
    "\u922e?": ">=",
    "\u922e\ufffd": ">=",
    "\u923a?": "<=",
    "\u923a\ufffd": "<=",
    "\u922d?": "-",
    "\u922d\ufffd": "-",
    "\u923c?": "x",
    "\u923c\ufffd": "x",
    "\u6f0f": "(c)",
    "\ufffd": "",
}

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CID_RE = re.compile(r"\(cid:\d+\)")
LONG_DOTS_RE = re.compile(r"\.{6,}")
MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")
PAGE_FOOTER_RE = re.compile(
    r"(?i)\b(?:page\s*)?\d+\s+(?:of|/)\s+\d+\b|"
    r"^\s*page\s*\d+\s*$|"
    r"^\s*\d+\s*$"
)
HEADER_FOOTER_RE = re.compile(
    r"(?i)\b(?:all rights reserved|subject to notice of rights|"
    r"some rights reserved|morbidity and mortality weekly report)\b"
)
BOILERPLATE_LINE_RE = re.compile(
    r"(?i)^\s*(?:path|citation|footnotes?|references?|table of contents|contents)\s*$"
)
MARKDOWN_PREFIX_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*|[-*]\s+)")
SINGLE_MARKER_RE = re.compile(r"(?i)^\s*(?:\[\d+\]|\([a-z]\)|[a-z]|\d+)\s*$")
URL_WRAP_RE = re.compile(r"(?i)(https?://\S+)\s+(\S+)")
INLINE_PAGE_RE = re.compile(r"(?i)\bpage\s+\d+\s+of\s+\d+\b")
INLINE_BOILERPLATE_RE = re.compile(
    r"(?i)(?:"
    r"\u00a9\s+[^.]{2,120}\s+\d{4}\.\s+(?:All|Some)\s+rights reserved\.|"
    r"Subject to Notice of rights\s+\([^)]*\)\.?|"
    r"Copyright\s+\u00a9\s+[^|.\n]{2,160}"
    r")"
)
PDF_WORD_GLUE_REPAIRS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bJournalof\s*Clinical\s*SleepMedicine\b"), "Journal of Clinical Sleep Medicine"),
    (re.compile(r"(?i)\bGRADEcertaintyofevidence\b"), "GRADE certainty of evidence"),
    (re.compile(r"(?i)\bspecificrecommendationrequirestheclinician(?=to|use|\b)"), "specific recommendation requires the clinician "),
    (re.compile(r"(?i)\brecommendationrequirestheclinician(?=to|use|\b)"), "recommendation requires the clinician "),
    (re.compile(r"(?i)\brecommendationsinthisguideline\b"), "recommendations in this guideline"),
    (re.compile(r"(?i)\bTheAASM(?=\w)"), "The AASM "),
    (re.compile(r"(?i)TheTF(?=\w)"), "The TF "),
    (re.compile(r"(?i)theTF(?=\w)"), " the TF "),
    (re.compile(r"(?i)\bsuggeststhatclinicians"), "suggests that clinicians "),
    (re.compile(r"(?i)\brecommendsthatclinicians"), "recommends that clinicians "),
    (re.compile(r"(?i)\bclinicianstreat"), "clinicians treat "),
    (re.compile(r"(?i)\bcliniciansuse"), "clinicians use "),
    (re.compile(r"(?i)\btreatchildrenandadolescentswith"), "treat children and adolescents with "),
    (re.compile(r"(?i)\busestrategically"), "use strategically"),
    (re.compile(r"(?i)\bchildrenandadolescentswith"), "children and adolescents with "),
    (re.compile(r"(?i)\bforallinterventions"), "for all interventions "),
    (re.compile(r"(?i)\bforallpatients\b"), "for all patients"),
    (re.compile(r"(?i)\bpatientswith\b"), "patients with"),
    (re.compile(r"(?i)\badultswith\b"), "adults with"),
    (re.compile(r"(?i)\bchildrenwith\b"), "children with"),
    (re.compile(r"(?i)\btreatmentwith\b"), "treatment with"),
    (re.compile(r"(?i)\btreatmentof\b"), "treatment of"),
    (re.compile(r"(?i)\bevidenceof\b"), "evidence of"),
    (re.compile(r"(?i)\briskfor\b"), "risk for"),
    (re.compile(r"(?i)\buseof\b"), "use of"),
)
REVERSED_COPYRIGHT_RE = re.compile(r"(?i)\benicideMpeelSfoymedacAnaciremA\s*(?:\u00a9|\?)\s*thgirypoC\b")


def get_text_field(record: Dict[str, Any], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> str:

    for field in text_fields:
        if str(record.get(field) or "").strip():
            return field
    return text_fields[0]


#负责最底层的编码/字符规范化
def normalize_text_encoding(text: str) -> str:
    text = str(text or "")
    text = html.unescape(text)
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)

    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)

    text = CID_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub("", text)
    return text


# 对PDF文件的噪声信息进行清洗
def remove_inline_pdf_noise(text: str) -> str:

    text = INLINE_BOILERPLATE_RE.sub(" ", text)
    text = REVERSED_COPYRIGHT_RE.sub(" ", text)
    text = INLINE_PAGE_RE.sub(" ", text)
    return text


def repair_pdf_word_glue(text: str) -> str:
    repaired = str(text or "")
    for pattern, replacement in PDF_WORD_GLUE_REPAIRS:
        repaired = pattern.sub(replacement, repaired)
    return repaired


#对单行的文本进行清洗
def clean_inline_text(text: str) -> str:

    text = normalize_text_encoding(text)
    text = remove_inline_pdf_noise(text)
    text = repair_pdf_word_glue(text)
    text = MARKDOWN_PREFIX_RE.sub("", text)
    text = LONG_DOTS_RE.sub(" ", text)
    text = re.sub(r"(?i)\bcitation\s+", "", text)
    text = re.sub(r"\s+\([a-z]\)\s+", " ", text)
    text = URL_WRAP_RE.sub(r"\1\2", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text.strip()


# 判断某一行是否应该被丢弃
def should_drop_line(line: str) -> bool:

    stripped = line.strip()
    if not stripped:
        return True
    if BOILERPLATE_LINE_RE.match(stripped):
        return True
    if PAGE_FOOTER_RE.search(stripped) and len(stripped) <= 40:
        return True
    if HEADER_FOOTER_RE.search(stripped) and len(stripped) <= 140:
        return True
    if SINGLE_MARKER_RE.match(stripped):
        return True
    return False


# 对整篇文本进行清洗
def clean_structured_text(text: str) -> str:

    text = normalize_text_encoding(text)
    text = remove_inline_pdf_noise(text)
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)

    lines: List[str] = []
    previous = ""
    for raw_line in text.split("\n"):
        line = clean_inline_text(raw_line)
        if should_drop_line(line):
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line

    return "\n".join(lines).strip()


# 清洗单条数据
def clean_record(record: Dict[str, Any], text_fields: Tuple[str, ...] = TEXT_FIELDS) -> Dict[str, Any]:

    field = get_text_field(record, text_fields)
    cleaned = dict(record)
    cleaned[field] = clean_structured_text(str(record.get(field) or ""))
    return cleaned



def clean_records(
    records: Iterable[Dict[str, Any]],
    text_fields: Tuple[str, ...] = TEXT_FIELDS,
    min_chars: int = MIN_TEXT_CHARS,
) -> List[Dict[str, Any]]:

    cleaned_records: List[Dict[str, Any]] = []
    for record in records:
        cleaned = clean_record(record, text_fields=text_fields)
        text = str(cleaned.get(get_text_field(cleaned, text_fields)) or "").strip()
        if len(text) < min_chars:
            continue
        cleaned_records.append(cleaned)
    return cleaned_records

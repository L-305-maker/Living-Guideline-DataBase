"""Best-effort metadata extraction from Markdown and file paths."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


PATH_HINTS = {
    "aan_pdf": "AAN",
    "aasld_pdf": "AASLD",
    "aasm_pdf": "AASM",
    "aha_pdf": "AHA",
    "ats_pdf": "ATS",
    "cdc_pdf": "CDC",
    "china_pdf": "China",
    "cma_pdf": "CMA",
    "esc_pdf": "ESC",
    "gina_pdf": "GINA",
    "gold_pdf": "GOLD",
    "idsa_pdf": "IDSA",
    "jacc_pdf": "JACC",
    "kdigo_pdf": "KDIGO",
    "nice_pdf": "NICE",
    "pmc_pdf": "PMC",
    "sign_pdf": "SIGN",
    "trip_pdf": "TRIP",
    "uspstf_pdf": "USPSTF",
    "vadod_pdf": "VA/DOD",
    "who_pdf": "WHO",
}
TEXT_SOURCE_PATTERNS = [
    ("WHO", ("world health organization", "\u4e16\u754c\u536b\u751f\u7ec4\u7ec7")),
    ("NICE", ("national institute for health and care excellence", "nice guideline", "nice guidelines")),
    ("CDC", ("centers for disease control", "centers for disease control and prevention")),
    ("VA/DOD", ("va/dod", "department of veterans affairs")),
    ("AAN", ("american academy of neurology",)),
    ("AHA", ("american heart association",)),
    ("ATS", ("american thoracic society",)),
    ("GINA", ("global initiative for asthma",)),
    ("USPSTF", ("u.s. preventive services task force", "us preventive services task force")),
    ("AASLD", ("american association for the study of liver diseases",)),
]
CMA_TEXT_PATTERNS = (
    "\u4e2d\u534e\u533b\u5b66\u4f1a",
    "\u4e2d\u56fd\u533b\u5e08\u534f\u4f1a",
    "\u4e2d\u534e\u9884\u9632\u533b\u5b66\u4f1a",
    "\u4e2d\u56fd\u533b\u836f\u6559\u80b2\u534f\u4f1a",
    "\u4e2d\u56fd\u6297\u764c\u534f\u4f1a",
    "\u4e2d\u56fd\u533b\u9662\u534f\u4f1a",
    "\u4e2d\u56fd\u7814\u7a76\u578b\u533b\u9662\u5b66\u4f1a",
    "\u4e2d\u56fd\u8001\u5e74\u533b\u5b66\u5b66\u4f1a",
    "\u4e2d\u56fd\u5eb7\u590d\u533b\u5b66\u4f1a",
    "\u4e2d\u56fd\u4e2d\u897f\u533b\u7ed3\u5408\u5b66\u4f1a",
    "\u4e2d\u56fd\u8425\u517b\u5b66\u4f1a",
    "\u4e2d\u56fd\u5352\u4e2d\u5b66\u4f1a",
    "\u4e2d\u56fd\u836f\u5b66\u4f1a",
    "\u4e2d\u56fd\u533b\u7597\u4fdd\u5065\u56fd\u9645\u4ea4\u6d41\u4fc3\u8fdb\u4f1a",
    "\u56fd\u5bb6\u611f\u67d3\u6027\u75be\u75c5\u4e34\u5e8a\u533b\u5b66\u7814\u7a76\u4e2d\u5fc3",
    "\u56fd\u5bb6\u4f20\u67d3\u75c5\u533b\u5b66\u4e2d\u5fc3",
    "\u56fd\u5bb6\u536b\u751f\u5065\u5eb7\u59d4",
    "\u75be\u75c5\u9884\u9632\u63a7\u5236\u4e2d\u5fc3",
    "\u9884\u9632\u533b\u5b66\u4f1a",
    "cma.j.",
    "Chinese Medical Association",
    "Chinese Medical Doctor Association",
    "Chinese Preventive Medicine Association",
)
CMA_GENERIC_ORG_RE = re.compile(r"(?:\u4e2d\u534e|\u4e2d\u56fd)[\u4e00-\u9fff]{0,30}(?:\u533b\u5b66\u4f1a|\u533b\u5e08\u534f\u4f1a|\u534f\u4f1a|\u5b66\u4f1a|\u8054\u76df|\u59d4\u5458\u4f1a|\u4e13\u5bb6\u7ec4)")
ABSTRACT_BOUNDARY_RE = re.compile(r"(\u3010\s*(?:\u6458\u8981|\u63d0\u8981)\s*\u3011|(?:\u6458\u8981|\u63d0\u8981|Abstract)\s*[:\uff1a])", re.I)
YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
FULLWIDTH_RE = re.compile(r"[\uff01-\uff5e]")
BAD_TITLE_CHARS_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_title_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"^\d{5,}[_\-\s]+", "", stem)
    stem = re.sub(r"^\d{5,}", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return "Untitled"
    return stem if re.search(r"[\u4e00-\u9fff]", stem) else stem.title()


def helper_source_from_path(path: Path) -> str | None:
    for part in path.parts:
        hint = PATH_HINTS.get(part.lower())
        if hint:
            return hint
        lowered = part.lower()
        if lowered.endswith("_pdf") and len(lowered) > 4:
            return lowered[:-4]
    return None


def helper_source_header(text: str, max_chars: int = 3000) -> str:
    head = (text or "")[:max_chars]
    match = ABSTRACT_BOUNDARY_RE.search(head)
    return head[: match.start()] if match else head


def helper_is_consensus_path(path: Path) -> bool:
    return any(part.lower() == "consensus" for part in path.parts)


def helper_has_cma_signal(text: str) -> bool:
    if any(pattern in text for pattern in CMA_TEXT_PATTERNS):
        return True
    return bool(CMA_GENERIC_ORG_RE.search(text))


def extract_source_institution(pdf_path: str | Path, text: str = "", title: str = "") -> str:
    path = Path(pdf_path)
    path_hint = helper_source_from_path(path)
    if path_hint:
        return path_hint

    # 仅使用标题、文件名和摘要前的作者区推断来源；扫描全文会把正文引用的机构误判为发布方。
    header = helper_source_header(text)
    source_haystack = f"{path.name}\n{title}\n{header}"
    if helper_has_cma_signal(source_haystack):
        return "CMA"
    if helper_is_consensus_path(path):
        return "Unknown"

    lower = source_haystack.lower()
    for source, patterns in TEXT_SOURCE_PATTERNS:
        if any(pattern.lower() in lower for pattern in patterns):
            return source
    return "Unknown"


def helper_normalize_digits(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def helper_candidate_years(text: str) -> list[int]:
    normalized = helper_normalize_digits(text)
    years = [int(match.group(1)) for match in YEAR_RE.finditer(normalized)]
    return [year for year in years if 2012 <= year <= 2026]


def extract_publication_date(pdf_path: str | Path, text: str = "") -> str:
    # 优先使用文件名中的年份，期刊页眉可能包含与发布日期无关的历史年份。
    file_years = helper_candidate_years(str(pdf_path))
    if file_years:
        return f"{file_years[0]}-01-01"
    haystack = f"{Path(pdf_path).name}\n{text[:12000]}"
    years = helper_candidate_years(haystack)
    if years:
        return f"{years[0]}-01-01"
    return "unknown"


def helper_title_score(title: str) -> float:
    normalized = helper_normalize_digits(title or "").strip()
    if not normalized:
        return -100.0
    visible = [ch for ch in normalized if not ch.isspace()]
    if not visible:
        return -100.0
    letters_digits = sum(1 for ch in visible if ch.isalpha() or ch.isdigit() or "\u4e00" <= ch <= "\u9fff")
    cjk = sum(1 for ch in visible if "\u4e00" <= ch <= "\u9fff")
    controls = sum(1 for ch in visible if unicodedata.category(ch)[0] == "C")
    ascii_symbols = sum(1 for ch in visible if ord(ch) < 128 and not ch.isalnum())
    score = letters_digits / len(visible)
    score += min(cjk, 12) * 0.03
    score -= controls * 0.5
    score -= ascii_symbols / max(1, len(visible)) * 0.5
    if len(normalized) < 6:
        score -= 0.4
    if len(normalized) > 160:
        score -= 0.2
    if re.search(r"\b(vol|no|journal|杂志|第\d+卷|第\d+期)\b", normalized, re.I):
        score -= 0.15
    return score


def is_suspicious_title(title: str) -> bool:
    normalized = helper_normalize_digits(title or "").strip()
    if len(normalized) < 4 or BAD_TITLE_CHARS_RE.search(normalized):
        return True
    journal_header_patterns = [
        r"中华.{0,12}杂志.{0,80}第\s*\d+\s*卷.{0,40}第\s*\d+\s*期",
        r"Chin\s+J\s+.{0,80}\bVol\.?\s*\d+.{0,40}\bNo\.?\s*\d+",
        r"\bVol\.?\s*\d+.{0,40}\bNo\.?\s*\d+",
        r"^\d+\s+年\s*\d+\s*月\s*第\s*\d+\s*卷\s*第\s*\d+\s*期",
    ]
    if any(re.search(pattern, normalized, re.I) for pattern in journal_header_patterns):
        return True
    visible = [ch for ch in normalized if not ch.isspace()]
    if not visible:
        return True
    letters_digits = sum(1 for ch in visible if ch.isalpha() or ch.isdigit() or "\u4e00" <= ch <= "\u9fff")
    ascii_symbols = sum(1 for ch in visible if ord(ch) < 128 and not ch.isalnum())
    if letters_digits / len(visible) < 0.25:
        return True
    if len(visible) > 30 and ascii_symbols / len(visible) > 0.45:
        return True
    return False


def clean_title(title: str) -> str:
    title = helper_normalize_digits(title or "")
    title = BAD_TITLE_CHARS_RE.sub("", title)
    title = re.sub(r"<!--.*?-->", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^[\W_]+|[\W_]+$", "", title, flags=re.UNICODE).strip()
    return title[:180].strip()


def extract_title(markdown: str, pdf_path: str | Path) -> str:
    candidates: list[str] = []
    match = HEADING_RE.search(markdown)
    if match and match.group(1).strip():
        candidates.append(match.group(1).strip())
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("<!--"):
            candidates.append(stripped[:220])
            if len(candidates) >= 12:
                break
    candidates.append(clean_title_from_filename(pdf_path))
    cleaned = [clean_title(candidate) for candidate in candidates]
    usable = [candidate for candidate in cleaned if candidate and not is_suspicious_title(candidate)]
    if usable:
        return max(usable, key=helper_title_score)
    return max((candidate for candidate in cleaned if candidate), key=helper_title_score, default=clean_title_from_filename(pdf_path))


def extract_abstract(markdown: str, max_chars: int = 1200) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        lines.append(stripped)
        if sum(len(item) for item in lines) >= max_chars:
            break
    return " ".join(lines)[:max_chars]

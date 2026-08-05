"""Loss-preserving Markdown cleanup.

清洗目标（loss-preserving 强调『尽量保留原文，谨慎删除』）：
- OCR 残留：控制字符、坏 glyph、私有 Unicode 区字符、空格化拉丁数字；
- 结构噪声：页码标记 <!-- page: N -->、重复的页眉/页脚；
- 元数据噪声：DOI / 邮箱 / 通信作者 / 期刊卷期号 / PII 等；
- mojibake 检测：通过 signal_ratio / weird_ratio / 拼写碎片等指标识别，
  触发条件后写入 pdf_text_mojibake flag，并补 metadata 提示后续需要 OCR。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from src.models.schemas import DocumentRecord, dump_model
from src.utils.clinical_department import classify_clinical_departments
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.ids import sha256_text
from src.utils.io import DATA_DIR, ensure_dir, iter_markdown_files
from src.utils.metadata import extract_abstract


PAGE_RE = re.compile(r"^\s*<!--\s*page:\s*\d+\s*-->\s*$", re.I)  # 页面标记 <!-- page: N -->，用于页眉/页脚去重与清洗。
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # 控制字符（U+0000-U+001F，除 \n \r \t），直接删除。
BAD_GLYPHS_RE = re.compile(r"[\ufffd\u25a0\u25a1\u25aa]")  # 明显的乱码占位字符：\uFFFD 替换符、方块占位符等。
PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")  # Unicode 私有区字符（PUA），常因字体缺失产生，全部删除。
WEIRD_TEXT_LAYER_RE = re.compile(r"[\x7f-\x9f\u00ac\u00ae\u00af\u00b1\u00b4-\u00b6\u00b8\u00bc-\u00be\u0370-\u03ff]")  # PDF 文本层偶发的非 ASCII Latin 字符与希腊字母，可能是 mojibake 信号。
HEADING_RE = re.compile(r"^#{1,6}\s+", re.M)  # Markdown 标题前缀（行首的 1-6 个 #）。
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")  # 中文字符范围（含平假名/片假名），用于中英文段判定。
CJK_TERMINAL_RE = re.compile(r"[。！？!?；;：:]$")  # 中文/英文句末标点，决定 CJK 行是否需要继续合并。
REFERENCE_HEADING_RE = re.compile(r"^(#{1,6}\s+)(references|bibliography|\u53c2\u8003\u6587\u732e)\b", re.I | re.M)  # 参考文献标题（references/bibliography/参考文献），用于插入标记。
SPACED_ALNUM_RUN_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9]\s+){3,}[A-Za-z0-9](?![A-Za-z0-9])")  # 被空格拆散的拉丁数字连续串，OCR 常见错误，尝试合并。
MOJIBAKE_MARKER_CHARS = set("摇揖铱鄄臆茁冶誗ꎬꎻꎮ")
FORM_WORD_RE = re.compile(r"\b(?:date|time|name|notes?|score|pain|headache|medication|yes|no)\b", re.I)  # 表单常见词（date/name/score/pain 等），用于识别 PDF 表单残留。
REPEATED_BLANK_RE = re.compile(r"_{3,}")  # 连续下划线（___+），识别为表单填空符。
JOURNAL_HEADER_RE = re.compile(  # 期刊卷期号页眉（如『中华医学杂志 第 30 卷 第 5 期』）。
    r"(?:中华.{0,24}杂志.{0,120}(?:第\s*\d+\s*卷|Vol\.?\s*\d+).{0,80}(?:第\s*\d+\s*期|No\.?\s*\d+))|"
    r"(?:(?:Chinese\s*Journal|Chin\s*J|中华).{0,120}(?:Vol\.?\s*\d+|No\.?\s*\d+))",
    re.I,
)
DOI_RE = re.compile(  # DOI 字符串（含多种 URL 形式），删除避免出现在正文。
    r"(?:https?://)?(?:dx\.)?doi\.org/\S+|"
    r"https?://(?:dx\.)?doi\.\S*|"
    r"\bdoi\b\s*[:：]?\s*(?:10\.\d{4,9}/\S+)?",
    re.I,
)
PII_RE = re.compile(r"\bpii\s*[:：]\s*\S+", re.I)  # PII 编号（出版商内部编号），类似 DOI 处理。
EMAIL_ADDRESS_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?=\d|[^A-Z0-9]|$)", re.I)  # 邮箱地址，删除避免 PII 泄漏。
CONTACT_LINE_RE = re.compile(  # 通信作者行（Address correspondence to ...），整行删除。
    r"^\s*(?:Address\s+correspondence\s+to|Correspondence\s+to|Corresponding\s+author|For\s+correspondence\b|Contact\s*:).{0,300}$",
    re.I,
)
INLINE_NOISE_RE = re.compile(  # 行内噪声（DOI/邮箱/通信作者等），按出现位置截断。
    r"(?:DOI|doi)\s*[:：]\s*\S+|"
    r"(?:E[-￣\s]*mail|Email)\s*[:：]\s*\S+|"
    r"(?:共同)?(?:通信|通讯|第一)作者\s*[:：][^。；;\n]{0,160}|"
    r"(?:Co[-\s]*(?:first|corresponding)\s+author)\s*[:：][^.;\n]{0,180}",
    re.I,
)
ABSTRACT_MARKER_RE = re.compile(r"(【\s*(?:提要|摘要)\s*】|(?:摘要|提要|Abstract)\s*[:：])", re.I)  # 摘要/提要标记（【摘要】 / Abstract:），保留正文。
NOISY_LINE_RE = re.compile(r"^[\d\s!！?？|/\\.,;:：；()（）\\-—_+=*'\"`~<>A-Za-z]{8,}$")  # 纯符号/数字/标点的『乱码行』模式，启发式判定。
LATIN_OR_NUMBER_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")  # 拉丁字母或数字 token，文本信号统计用。
OCR_PUNCT_TRANSLATION = str.maketrans(
    {
        "ꎬ": "，",
        "ꎻ": "；",
        "ꎮ": "。",
        "￣": "-",
        "＠": "@",
    }
)


def helper_split_pages(body: str) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    for line in body.splitlines():
        if PAGE_RE.match(line) and current:
            pages.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        pages.append(current)
    return pages


def helper_repeated_headers_footers(body: str) -> set[str]:
    pages = helper_split_pages(body)
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        content = [line.strip() for line in page if line.strip() and not PAGE_RE.match(line)]
        for candidate in set(content[:3] + content[-3:]):
            if 0 < len(candidate) < 140:
                counts[candidate] += 1
    threshold = max(2, int(len(pages) * 0.30 + 0.999))
    return {line for line, count in counts.items() if count >= threshold}


def helper_fix_english_linebreaks(text: str) -> str:
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[A-Za-z])", "", text)
    return re.sub(r"(?<=[a-z,;])\n(?=[a-z(])", " ", text)


def helper_join_spaced_alnum_runs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        compact = re.sub(r"\s+", "", value)
        if sum(char.isalpha() for char in compact) >= 4:
            return compact
        return value

    return SPACED_ALNUM_RUN_RE.sub(repl, text)


def helper_normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(OCR_PUNCT_TRANSLATION)
    text = PRIVATE_USE_RE.sub("", text)
    return "\n".join(helper_join_spaced_alnum_runs(line) for line in text.splitlines())


def helper_visible_chars(text: str) -> list[str]:
    return [char for char in text if not char.isspace()]


def helper_is_weird_text_layer_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x7F <= codepoint <= 0x9F
        or codepoint in {0x00AC, 0x00AE, 0x00AF, 0x00B1}
        or 0x00B4 <= codepoint <= 0x00B6
        or 0x00B8 <= codepoint <= 0x00BE
        or 0x0370 <= codepoint <= 0x03FF
    )


def helper_text_signal_stats(text: str) -> dict[str, float]:
    visible_count = 0
    signal = 0
    cjk = 0
    weird = 0
    mojibake_markers = 0
    punctuation = 0
    for char in text:
        if char.isspace():
            continue
        visible_count += 1
        is_cjk = "\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff"
        if char.isalnum() or is_cjk:
            signal += 1
        if is_cjk:
            cjk += 1
        if helper_is_weird_text_layer_char(char):
            weird += 1
        if char in MOJIBAKE_MARKER_CHARS:
            mojibake_markers += 1
        if not char.isalnum() and not is_cjk:
            punctuation += 1
    total = max(1, visible_count)
    latin = sum(1 for _ in LATIN_OR_NUMBER_TOKEN_RE.finditer(text))
    return {
        "visible": float(visible_count),
        "signal": float(signal),
        "cjk": float(cjk),
        "signal_ratio": signal / total,
        "cjk_ratio": cjk / total,
        "latin_token_ratio": latin / total,
        "weird_ratio": weird / total,
        "mojibake_marker_ratio": mojibake_markers / total,
        "punctuation_ratio": punctuation / total,
    }


def helper_looks_like_form_text(text: str) -> bool:
    visible_count = 0
    underline_count = 0
    for char in text:
        if char.isspace():
            continue
        visible_count += 1
        if char == "_":
            underline_count += 1
    total = max(1, visible_count)
    repeated_blanks = sum(1 for _ in REPEATED_BLANK_RE.finditer(text))
    underline_ratio = underline_count / total
    form_word_count = sum(1 for _ in FORM_WORD_RE.finditer(text))
    return (repeated_blanks >= 5 and form_word_count >= 3) or (repeated_blanks >= 12 and underline_ratio > 0.12)


def helper_looks_like_pdf_text_mojibake(text: str) -> bool:
    stats = helper_text_signal_stats(text)
    if stats["visible"] < 1000:
        return False
    if helper_looks_like_form_text(text):
        return False
    if stats["weird_ratio"] > 0.12:
        return True
    if stats["signal_ratio"] < 0.55 and stats["punctuation_ratio"] > 0.45:
        return True
    if stats["signal_ratio"] < 0.60 and stats["cjk_ratio"] < 0.05 and stats["mojibake_marker_ratio"] > 0.008:
        return True
    return False


def helper_content_visible_count(body: str) -> int:
    return sum(
        1
        for line in body.splitlines()
        if not PAGE_RE.match(line.strip())
        for char in line
        if not char.isspace()
    )


def helper_metadata_flags(metadata: dict[str, Any]) -> list[str]:
    return [flag.strip() for flag in re.split(r"[,;]\s*", str(metadata.get("cleaning_flags", ""))) if flag.strip()]


def helper_add_metadata_flag(metadata: dict[str, Any], flag: str) -> None:
    """向 cleaning_flags 追加单个 flag（去重）。"""
    flags = helper_metadata_flags(metadata)
    if flag not in flags:
        flags.append(flag)
    metadata["cleaning_flags"] = ",".join(flags)


def helper_trim_bibliographic_prefix(line: str) -> str:
    match = ABSTRACT_MARKER_RE.search(line)
    if match and match.start() > 80:
        return line[match.start() :]
    return line


def helper_strip_inline_noise(line: str) -> str:
    if CONTACT_LINE_RE.match(line):
        return ""
    line = JOURNAL_HEADER_RE.sub(" ", line)
    line = DOI_RE.sub(" ", line)
    line = PII_RE.sub(" ", line)
    line = EMAIL_ADDRESS_RE.sub(" ", line)
    line = INLINE_NOISE_RE.sub(" ", line)
    line = helper_trim_bibliographic_prefix(line)
    return re.sub(r"\s+", " ", line).strip()


def helper_is_low_signal_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or HEADING_RE.match(stripped):
        return False
    if NOISY_LINE_RE.match(stripped):
        letters = sum(1 for char in stripped if char.isalpha() or "\u4e00" <= char <= "\u9fff")
        symbols = sum(1 for char in stripped if not char.isalnum() and not char.isspace())
        return letters < 6 or symbols / max(1, len(stripped)) > 0.25
    visible = [char for char in stripped if not char.isspace()]
    if len(visible) < 12:
        return False
    signal = sum(1 for char in visible if char.isalnum() or "\u4e00" <= char <= "\u9fff")
    return signal / max(1, len(visible)) < 0.25


def helper_is_bad_heading_line(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"^#{1,6}\s*$", stripped):
        return True
    if not HEADING_RE.match(stripped):
        return False
    title = HEADING_RE.sub("", stripped).strip()
    if not title:
        return True
    letters = sum(1 for char in title if char.isalpha() or "\u4e00" <= char <= "\u9fff")
    if letters == 0:
        return True
    return bool(NOISY_LINE_RE.match(title) and letters < 6)


def helper_fragmented_latin_ocr(body: str) -> bool:
    token_count = 0
    single_alpha_count = 0
    digit_count = 0
    natural_alpha_count = 0
    for match in LATIN_OR_NUMBER_TOKEN_RE.finditer(body):
        token = match.group(0)
        token_count += 1
        if token.isalpha():
            if len(token) == 1:
                single_alpha_count += 1
            elif len(token) >= 4 and re.search(r"[aeiouAEIOU]", token):
                natural_alpha_count += 1
        elif token.isdigit():
            digit_count += 1
    if token_count < 40:
        return False
    fragmented_ratio = (single_alpha_count + digit_count) / max(1, token_count)
    natural_ratio = natural_alpha_count / max(1, token_count)
    return fragmented_ratio > 0.45 and natural_ratio < 0.18


def helper_is_noisy_title(title: str) -> bool:
    normalized = unicodedata.normalize("NFKC", title or "").strip()
    if not normalized:
        return False
    letters = sum(1 for char in normalized if char.isalpha() or "\u4e00" <= char <= "\u9fff")
    if NOISY_LINE_RE.match(normalized) and letters < 8:
        return True
    visible = [char for char in normalized if not char.isspace()]
    symbols = sum(1 for char in visible if not char.isalnum())
    return len(visible) >= 12 and symbols / max(1, len(visible)) > 0.35 and letters < 12


def helper_is_cjk_prose(line: str) -> bool:
    stripped = line.strip()
    if not stripped or PAGE_RE.match(stripped) or HEADING_RE.match(stripped):
        return False
    if re.match(r"^([-*+]|\d+[.)、])\s+", stripped):
        return False
    return len(CJK_RE.findall(stripped)) >= 3


def helper_fix_cjk_linebreaks(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if merged and helper_is_cjk_prose(merged[-1]) and helper_is_cjk_prose(line) and not CJK_TERMINAL_RE.search(merged[-1].strip()):
            merged[-1] = merged[-1].rstrip() + line.lstrip()
        else:
            merged.append(line)
    return merged


def helper_mark_references(body: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}\n<!-- reference_section: true -->"

    return REFERENCE_HEADING_RE.sub(repl, body)


def assess_cleaned_body(body: str) -> dict[str, Any]:
    """评估清洗后的正文质量，返回 quality（ok/warning/poor）+ flags + signal_ratio。"""
    stats = helper_text_signal_stats(body)
    visible_count = int(stats["visible"])
    cjk = int(stats["cjk"])
    form_like = helper_looks_like_form_text(body)
    journal_headers = len(JOURNAL_HEADER_RE.findall(body))
    doi_residue = len(DOI_RE.findall(body))
    email_residue = len(EMAIL_ADDRESS_RE.findall(body))
    page_markers = sum(1 for line in body.splitlines() if PAGE_RE.match(line))
    bad_glyphs = len(BAD_GLYPHS_RE.findall(body)) + len(PRIVATE_USE_RE.findall(body))
    noisy_lines = sum(1 for line in body.splitlines() if helper_is_low_signal_line(line) or helper_is_bad_heading_line(line))
    signal_ratio = stats["signal_ratio"]
    flags: list[str] = []
    # 多个独立信号共同描述清洗质量；单个弱信号不会直接判定 OCR 失败。
    if visible_count < 300:
        flags.append("very_short_text")
    if signal_ratio < 0.45 and not form_like:
        flags.append("low_text_signal")
    if not form_like and (noisy_lines >= 2 or (visible_count < 500 and noisy_lines >= 1)):
        flags.append("noisy_ocr_lines")
    if not form_like and cjk == 0 and visible_count > 300 and (signal_ratio < 0.60 or helper_fragmented_latin_ocr(body)):
        flags.append("likely_ocr_failure")
    if helper_looks_like_pdf_text_mojibake(body):
        flags.append("pdf_text_mojibake")
        if "likely_ocr_failure" not in flags:
            flags.append("likely_ocr_failure")
    if journal_headers:
        flags.append("journal_header_residue")
    if doi_residue:
        flags.append("doi_residue")
    if email_residue:
        flags.append("email_residue")
    if page_markers:
        flags.append("page_marker_residue")
    if bad_glyphs:
        flags.append("bad_glyph_residue")
    # 只有影响正文可读性的核心标记判为 poor，残留页码等轻问题只标 warning。
    quality = "poor" if {"low_text_signal", "likely_ocr_failure", "noisy_ocr_lines"} & set(flags) else ("warning" if flags else "ok")
    return {
        "quality": quality,
        "flags": flags,
        "signal_ratio": round(signal_ratio, 3),
        "visible_chars": visible_count,
    }


def helper_apply_quality_metadata(metadata: dict[str, str], body: str) -> None:
    """根据正文评估把 cleaning_quality / cleaning_flags / OCR 提示写入 front-matter。"""
    report = assess_cleaned_body(body)
    flags = helper_metadata_flags(metadata) + list(report["flags"])
    if helper_is_noisy_title(metadata.get("title", "")):
        flags.append("noisy_ocr_title")
    # PDF 文本层乱码通常需要 OCR，统一补充失败标记供后续重排降权。
    if "pdf_text_mojibake" in flags and "likely_ocr_failure" not in flags:
        flags.append("likely_ocr_failure")
    seen_flags = []
    for flag in flags:
        if flag not in seen_flags:
            seen_flags.append(flag)
    quality = report["quality"]
    if "pdf_text_mojibake" in seen_flags:
        metadata["pdf_text_quality"] = "poor"
        metadata["pdf_needs_ocr"] = "true"
        metadata["source_pdf_text_quality"] = "poor"
        metadata["source_pdf_needs_ocr"] = "true"
        if not metadata.get("ocr_status") or metadata.get("ocr_status") in {"not_needed", "needed_not_applied"}:
            metadata["ocr_status"] = "needed_unavailable"
        if not metadata.get("ocr_error"):
            metadata["ocr_error"] = "PDF text layer appears garbled; OCR is required."
    if {"low_text_signal", "likely_ocr_failure", "noisy_ocr_lines", "noisy_ocr_title", "pdf_text_mojibake"} & set(seen_flags):
        quality = "poor"
    elif seen_flags:
        quality = "warning"
    metadata["cleaning_quality"] = quality
    metadata["cleaning_flags"] = ",".join(seen_flags)


def helper_loss_preserving_body(body: str, repeated: set[str]) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if PAGE_RE.match(stripped):
            continue
        if stripped in repeated:
            continue
        cleaned_line = helper_strip_inline_noise(line)
        lines.append(re.sub(r"[ \t]+", " ", cleaned_line).rstrip())
    lines = helper_fix_cjk_linebreaks(lines)
    return helper_mark_references("\n".join(lines))


def helper_should_use_loss_preserving_body(original_body: str, cleaned_body: str, original_mojibake: bool = False) -> bool:
    original_visible = helper_content_visible_count(original_body)
    cleaned_visible = helper_content_visible_count(cleaned_body)
    if original_visible < 1000:
        return False
    if original_mojibake:
        return False
    if cleaned_visible < 300:
        return True
    return cleaned_visible / max(1, original_visible) < 0.35


def clean_markdown_text(markdown: str) -> str:
    metadata, body = parse_front_matter(markdown)
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = helper_normalize_ocr_text(body)
    body = CONTROL_RE.sub("", body)
    body = BAD_GLYPHS_RE.sub("", body)
    body = helper_fix_english_linebreaks(body)
    normalized_body = body
    original_mojibake = helper_looks_like_pdf_text_mojibake(normalized_body)
    if original_mojibake:
        helper_add_metadata_flag(metadata, "pdf_text_mojibake")
    repeated = helper_repeated_headers_footers(body)
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if PAGE_RE.match(stripped):
            continue
        if stripped in repeated:
            continue
        cleaned_line = helper_strip_inline_noise(line)
        if helper_is_bad_heading_line(cleaned_line):
            continue
        if helper_is_low_signal_line(cleaned_line):
            continue
        lines.append(re.sub(r"[ \t]+", " ", cleaned_line).rstrip())
    lines = helper_fix_cjk_linebreaks(lines)
    body = "\n".join(lines)
    body = helper_mark_references(body)
    if helper_should_use_loss_preserving_body(normalized_body, body, original_mojibake=original_mojibake):
        body = helper_loss_preserving_body(normalized_body, repeated)
        helper_add_metadata_flag(metadata, "clean_loss_fallback")
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    if metadata:
        helper_apply_quality_metadata(metadata, body)
    return dump_front_matter(metadata, body) if metadata else body


def clean_file(raw_path: str | Path, output_dir: str | Path = DATA_DIR / "markdown_clean") -> DocumentRecord:
    raw_file = Path(raw_path)
    cleaned = clean_markdown_text(raw_file.read_text(encoding="utf-8", errors="replace"))
    metadata, body = parse_front_matter(cleaned)
    doc_id = metadata["id"]
    abstract = extract_abstract(body)
    department_result = classify_clinical_departments(str(metadata.get("title", "")), abstract, body)
    clinical_department = str(department_result["clinical_department"])
    metadata["clinical_department"] = clinical_department
    metadata["clinical_departments"] = department_result["clinical_departments"]
    metadata["department_scope"] = department_result["department_scope"]
    if "cleaning_quality" not in metadata:
        helper_apply_quality_metadata(metadata, body)
    cleaned = dump_front_matter(metadata, body)
    out_path = ensure_dir(output_dir) / f"{doc_id}.md"
    out_path.write_text(cleaned, encoding="utf-8", newline="\n")
    return DocumentRecord(
        doc_id=doc_id,
        title=metadata.get("title", ""),
        publication_date=metadata.get("publication_date") or "unknown",
        source_institution=metadata.get("source_institution") or "Unknown",
        clinical_department=clinical_department,
        clinical_departments=department_result["clinical_departments"],
        department_scope=str(department_result["department_scope"]),
        document_kind=str(metadata.get("document_kind") or "guideline"),
        source_file=metadata.get("source_file", ""),
        markdown_raw_path=str(raw_file),
        markdown_clean_path=str(out_path),
        abstract=abstract,
        content_sha256=sha256_text(cleaned),
        cleaning_quality=metadata.get("cleaning_quality", ""),
        cleaning_flags=metadata.get("cleaning_flags", ""),
        source_pdf_text_quality=metadata.get("source_pdf_text_quality", ""),
        source_pdf_needs_ocr=metadata.get("source_pdf_needs_ocr") == "true",
        source_pdf_is_scanned=metadata.get("source_pdf_is_scanned") == "true",
        pdf_text_quality=metadata.get("pdf_text_quality", ""),
        pdf_needs_ocr=metadata.get("pdf_needs_ocr") == "true",
        pdf_is_scanned=metadata.get("pdf_is_scanned") == "true",
        ocr_engine=metadata.get("ocr_engine", ""),
        ocr_applied=metadata.get("ocr_applied") == "true",
        ocr_status=metadata.get("ocr_status", ""),
        ocr_error=metadata.get("ocr_error", ""),
    )


def helper_raw_metadata(path: Path) -> dict[str, Any]:
    """读取 front-matter 元数据（仅解析，不读 body）。"""
    metadata, _body = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
    return metadata


def helper_source_file_exists(metadata: dict[str, Any]) -> bool:
    value = str(metadata.get("source_file") or "").strip()
    return bool(value and Path(value).is_file())


def helper_document_kind_priority(path: Path) -> tuple[bool, bool, str]:
    metadata = helper_raw_metadata(path)
    is_consensus = str(metadata.get("document_kind") or "guideline") == "consensus"
    return not helper_source_file_exists(metadata), is_consensus, str(path)


def clean_all(
    input_dir: str | Path = DATA_DIR / "markdown_raw",
    output_dir: str | Path = DATA_DIR / "markdown_clean",
    manifest_path: str | Path = DATA_DIR / "documents.jsonl",
    progress_every: int = 0,
) -> dict[str, Any]:
    # 批量清洗逐文档隔离失败，并在完成后删除不再对应输入文档的陈旧产物。
    manifest = Path(manifest_path)
    ensure_dir(manifest.parent)
    clean_dir = ensure_dir(output_dir)
    excluded_manifest = manifest.with_name("documents_excluded.jsonl")
    count = 0
    duplicates_skipped = 0
    missing_sources_skipped = 0
    excluded: list[dict[str, str]] = []
    active_clean_paths: set[Path] = set()
    seen_body_hashes: dict[str, str] = {}
    paths = list(iter_markdown_files(input_dir))
    paths.sort(key=helper_document_kind_priority)
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for path in paths:
            metadata = helper_raw_metadata(path)
            doc_id = str(metadata.get("id") or path.stem)
            if not helper_source_file_exists(metadata):
                excluded.append(
                    {
                        "doc_id": doc_id,
                        "reason": "missing_source_pdf",
                        "source_file": str(metadata.get("source_file") or ""),
                        "markdown_raw_path": str(path),
                    }
                )
                missing_sources_skipped += 1
                continue
            record = clean_file(path, output_dir)
            clean_path = Path(record.markdown_clean_path)
            _clean_metadata, body = parse_front_matter(clean_path.read_text(encoding="utf-8", errors="replace"))
            body_hash = sha256_text(body)
            if body_hash in seen_body_hashes:
                clean_path.unlink(missing_ok=True)
                excluded.append(
                    {
                        "doc_id": record.doc_id,
                        "reason": "duplicate_clean_body",
                        "canonical_doc_id": seen_body_hashes[body_hash],
                        "source_file": record.source_file,
                        "markdown_raw_path": str(path),
                    }
                )
                duplicates_skipped += 1
                continue
            seen_body_hashes[body_hash] = record.doc_id
            active_clean_paths.add(clean_path.resolve())
            handle.write(json.dumps(dump_model(record), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if progress_every > 0 and count % progress_every == 0:
                print(
                    json.dumps({"cleaned": count, "current": path.name}, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )
    stale_clean_removed = 0
    for path in iter_markdown_files(clean_dir):
        if path.resolve() not in active_clean_paths:
            path.unlink()
            stale_clean_removed += 1
    with excluded_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in excluded:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "cleaned": count,
        "duplicates_skipped": duplicates_skipped,
        "missing_sources_skipped": missing_sources_skipped,
        "stale_clean_removed": stale_clean_removed,
        "manifest": str(manifest_path),
        "excluded_manifest": str(excluded_manifest),
    }


def main() -> None:
    """CLI 入口：python -m src.pipeline.cleaning.cleaner。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA_DIR / "markdown_raw"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--manifest", default=str(DATA_DIR / "documents.jsonl"))
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(clean_all(args.input_dir, args.output_dir, args.manifest, args.progress_every), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

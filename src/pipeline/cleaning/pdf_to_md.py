"""PDF to Markdown conversion with front matter and stable document IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from src.models.schemas import DocumentRecord, dump_model
from src.pipeline.ocr.mineru_runner import MineruRun, mineru_enabled, run_mineru
from src.pipeline.ocr.ocrmypdf_runner import OcrResult, run_ocrmypdf
from src.pipeline.ocr.pdf_quality import PdfTextLayerReport, inspect_pdf_text_layer
from src.utils.clinical_department import classify_clinical_departments
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.ids import make_doc_id, sha256_file, sha256_text
from src.utils.io import DATA_DIR, ensure_dir, read_jsonl, write_jsonl
from src.utils.metadata import extract_abstract, extract_publication_date, extract_source_institution, extract_title


SECTION_NUMBER_RE = re.compile(r"^(?:\d+(?:\.\d+){1,4}|[IVXLCM]+\.)\s+[A-Z0-9(][\w\s,;:/&()\-–—]+$", re.I)
COMMON_SECTION_RE = re.compile(
    r"^(abstract|summary|executive summary|introduction|background|methods?|methodology|"
    r"recommendations?|guidelines?|conclusions?|discussion|results?|evidence|scope|"
    r"target population|references|bibliography|appendix|acknowledg(?:e)?ments?)\b",
    re.I,
)
CHINESE_COMMON_SECTION_RE = re.compile(
    r"^(摘要|提要|背景|前言|引言|方法|推荐意见|推荐|建议|指南|共识|结论|讨论|结果|证据|"
    r"适用范围|适用人群|目标人群|参考文献|附录|致谢)\b"
)
CHINESE_NUMBERED_SECTION_RE = re.compile(r"^(?:[一二三四五六七八九十]+[、.．]|（[一二三四五六七八九十]+）|第[一二三四五六七八九十\d]+[章节])")
CITATION_LABEL_RE = re.compile(r"^\[?\d+(?:\.\d+){1,5}\]?\s*(?:\([A-Z]+\))?$")
REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography|\u53c2\u8003\u6587\u732e)$", re.I)
TABLE_TITLE_RE = re.compile(r"^(table|fig(?:ure)?|box|algorithm)\s+\d+", re.I)
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\([a-zA-Z0-9]+\)\s+)")
TERMINAL_SENTENCE_RE = re.compile(r"[.!?。！？]\s*$")
WHITESPACE_RE = re.compile(r"\s+")
TABLE_VALUE_HEADINGS = {"standard", "guideline", "option", "level", "description", "term", "definition"}


@dataclass(frozen=True)
class PdfTextLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool


def helper_needs_join_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        return False
    if "\u4e00" <= left[-1] <= "\u9fff" or "\u4e00" <= right[0] <= "\u9fff":
        return False
    if right[0] in ",.;:!?，。；：！？)]}":
        return False
    if left[-1] in "([{":
        return False
    return True


def helper_merge_adjacent_line_fragments(lines: list[PdfTextLine]) -> list[PdfTextLine]:
    if not lines:
        return []
    merged: list[PdfTextLine] = []
    for line in sorted(lines, key=lambda item: (round(item.y0, 1), round(item.x0, 1))):
        if not merged:
            merged.append(line)
            continue
        previous = merged[-1]
        same_baseline = abs(previous.y0 - line.y0) <= max(1.8, min(previous.size, line.size) * 0.18)
        similar_size = abs(previous.size - line.size) <= max(1.0, min(previous.size, line.size) * 0.12)
        small_gap = -1.0 <= line.x0 - previous.x1 <= max(8.0, min(previous.size, line.size) * 0.8)
        if same_baseline and similar_size and small_gap:
            spacer = " " if helper_needs_join_space(previous.text, line.text) else ""
            combined_text = f"{previous.text}{spacer}{line.text}"
            total_len = max(1, len(previous.text) + len(line.text))
            merged[-1] = PdfTextLine(
                text=combined_text,
                x0=min(previous.x0, line.x0),
                y0=min(previous.y0, line.y0),
                x1=max(previous.x1, line.x1),
                y1=max(previous.y1, line.y1),
                size=((previous.size * len(previous.text)) + (line.size * len(line.text))) / total_len,
                bold=previous.bold and line.bold,
            )
            continue
        merged.append(line)
    return merged


def helper_open_doc(pdf_path: str | Path):
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required. Install pymupdf.") from exc
    return fitz.open(str(pdf_path))


def helper_with_pymupdf4llm(pdf_path: Path) -> str | None:
    try:
        import pymupdf4llm  # type: ignore
    except ImportError:
        return None
    try:
        return pymupdf4llm.to_markdown(str(pdf_path))
    except Exception:
        return None


def helper_normalize_pdf_text_line(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").replace("\u00a0", " ")).strip()


def helper_is_bold_span(span: dict[str, Any]) -> bool:
    font = str(span.get("font") or "").lower()
    flags = int(span.get("flags") or 0)
    return "bold" in font or bool(flags & 16)


def helper_join_spans(spans: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    previous_x1: float | None = None
    for span in spans:
        text = str(span.get("text") or "")
        if not text:
            continue
        bbox = span.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        x0 = float(bbox[0])
        if (
            parts
            and previous_x1 is not None
            and x0 - previous_x1 > 1.0
            and not parts[-1].endswith((" ", "-", "‐", "‑", "–", "—"))
            and not text.startswith((" ", ",", ".", ";", ":", ")", "]"))
        ):
            parts.append(" ")
        parts.append(text)
        previous_x1 = float(bbox[2])
    return "".join(parts)


def helper_collect_text_lines(page: Any) -> list[PdfTextLine]:
    lines: list[PdfTextLine] = []
    for block in page.get_text("dict", sort=False).get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = [span for span in raw_line.get("spans", []) if str(span.get("text") or "").strip()]
            text = helper_normalize_pdf_text_line(helper_join_spans(spans))
            if not text:
                continue
            bbox = raw_line.get("bbox") or block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            weights = [max(1, len(str(span.get("text") or "").strip())) for span in spans]
            sizes = [float(span.get("size") or 0.0) for span in spans]
            weighted_size = sum(size * weight for size, weight in zip(sizes, weights)) / max(1, sum(weights))
            bold_weight = sum(weight for span, weight in zip(spans, weights) if helper_is_bold_span(span))
            lines.append(
                PdfTextLine(
                    text=text,
                    x0=float(bbox[0]),
                    y0=float(bbox[1]),
                    x1=float(bbox[2]),
                    y1=float(bbox[3]),
                    size=weighted_size,
                    bold=bold_weight / max(1, sum(weights)) >= 0.55,
                )
            )
    return helper_merge_adjacent_line_fragments(lines)


def helper_body_text_size(lines: list[PdfTextLine]) -> float:
    candidates = [
        line.size
        for line in lines
        if len(line.text) >= 25 and not line.bold and not TABLE_TITLE_RE.match(line.text)
    ]
    if not candidates:
        candidates = [line.size for line in lines if len(line.text) >= 15]
    return float(median(candidates)) if candidates else 10.0


def helper_uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.upper() == char and char.lower() != char) / len(letters)


def helper_detect_column_start(lines: list[PdfTextLine], page_width: float) -> float | None:
    midpoint = page_width / 2.0
    left = [
        line
        for line in lines
        if line.y0 >= 40 and len(line.text) >= 12 and line.x0 < midpoint * 0.95 and line.x1 <= page_width * 0.72
    ]
    right = [line for line in lines if line.y0 >= 40 and len(line.text) >= 12 and line.x0 >= midpoint * 0.85]
    if len(left) < 5 or len(right) < 5:
        return None
    paired_ys: list[float] = []
    for left_line in sorted(left, key=lambda item: item.y0):
        matches = [right_line for right_line in right if abs(right_line.y0 - left_line.y0) <= 16]
        if matches:
            paired_ys.append(min(left_line.y0, matches[0].y0))
    for y0 in sorted(paired_ys):
        if sum(1 for y in paired_ys if y0 <= y <= y0 + 140) >= 4:
            return y0
    return None


def helper_reading_order(lines: list[PdfTextLine], page_width: float) -> list[PdfTextLine]:
    column_start = helper_detect_column_start(lines, page_width)
    if column_start is None:
        return sorted(lines, key=lambda item: (round(item.y0, 1), round(item.x0, 1)))

    midpoint = page_width / 2.0
    pre_column = [line for line in lines if line.y0 < column_start - 4]
    column_lines = [line for line in lines if line.y0 >= column_start - 4]
    left_column = [line for line in column_lines if line.x0 < midpoint]
    right_column = [line for line in column_lines if line.x0 >= midpoint]
    return (
        sorted(pre_column, key=lambda item: (round(item.y0, 1), round(item.x0, 1)))
        + sorted(left_column, key=lambda item: (round(item.y0, 1), round(item.x0, 1)))
        + sorted(right_column, key=lambda item: (round(item.y0, 1), round(item.x0, 1)))
    )


def helper_heading_level(line: PdfTextLine, body_size: float) -> int | None:
    # 标题等级按 Markdown 标记、编号模式和版式信号依次判定，弱信号不能覆盖强信号。
    text = line.text.strip()
    if not text or text.startswith("#") or LIST_ITEM_RE.match(text):
        return None
    if len(text) > 180:
        return None
    normalized = text.strip(" :")
    if normalized.casefold() in TABLE_VALUE_HEADINGS or normalized.startswith(("[", "(")) or CITATION_LABEL_RE.match(normalized):
        return None
    uppercase_ratio = helper_uppercase_ratio(normalized)
    common_heading = bool(COMMON_SECTION_RE.match(normalized) or CHINESE_COMMON_SECTION_RE.match(normalized))
    chinese_numbered_heading = bool(CHINESE_NUMBERED_SECTION_RE.match(normalized))

    if REFERENCE_HEADING_RE.match(normalized):
        return 2
    if chinese_numbered_heading and len(normalized) <= 120:
        return 2
    if SECTION_NUMBER_RE.match(normalized) and (uppercase_ratio >= 0.45 or line.bold or line.size >= body_size + 0.4):
        return 2
    if TABLE_TITLE_RE.match(normalized) and len(normalized) <= 160 and (line.bold or re.match(r"^(?:table|fig(?:ure)?|box|algorithm)\s+\d+\s*[:—–-]", normalized, re.I)):
        return 3
    if line.size >= body_size + 3.0 and len(normalized) >= 8:
        return 2
    if line.size >= body_size + 1.5 and common_heading:
        return 2
    if common_heading and len(normalized) <= 90 and (line.bold or text.rstrip().endswith(":") or uppercase_ratio >= 0.45):
        return 3
    if line.bold and 6 <= len(normalized) <= 120 and uppercase_ratio >= 0.55 and not TERMINAL_SENTENCE_RE.search(normalized):
        return 3
    return None


def helper_append_markdown_line(parts: list[str], line: str) -> None:
    if not line:
        if parts and parts[-1] != "":
            parts.append("")
        return
    parts.append(line)


def helper_format_pdf_lines_as_markdown(lines: list[PdfTextLine], page_width: float) -> list[str]:
    ordered = helper_reading_order(lines, page_width)
    body_size = helper_body_text_size(ordered)
    parts: list[str] = []
    for line in ordered:
        level = helper_heading_level(line, body_size)
        if level is None:
            helper_append_markdown_line(parts, line.text)
            continue
        if parts and parts[-1] != "":
            parts.append("")
        helper_append_markdown_line(parts, f"{'#' * level} {line.text.strip()}")
        parts.append("")
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def helper_with_pymupdf_plain(pdf_path: Path) -> str:
    doc = helper_open_doc(pdf_path)
    try:
        parts: list[str] = []
        for page_no, page in enumerate(doc, start=1):
            parts.append(f"<!-- page: {page_no} -->")
            text = page.get_text("text", sort=True).strip()
            if text:
                parts.append(text)
            parts.append("")
        return "\n".join(parts).strip() + "\n"
    finally:
        doc.close()


def helper_with_pymupdf(pdf_path: Path) -> str:
    doc = helper_open_doc(pdf_path)
    try:
        parts: list[str] = []
        for page_no, page in enumerate(doc, start=1):
            parts.append(f"<!-- page: {page_no} -->")
            lines = helper_collect_text_lines(page)
            if lines:
                parts.extend(helper_format_pdf_lines_as_markdown(lines, float(page.rect.width)))
            else:
                text = page.get_text("text", sort=True).strip()
                if text:
                    parts.append(text)
            parts.append("")
        markdown = "\n".join(parts).strip()
        return markdown + "\n" if markdown else helper_with_pymupdf_plain(pdf_path)
    finally:
        doc.close()


def helper_ensure_title_heading(markdown: str, title: str) -> str:
    body = markdown.strip()
    if body.startswith("# "):
        return body + "\n"
    return f"# {title}\n\n{body}\n"


def helper_inspect_pdf_for_ingestion(pdf: Path) -> PdfTextLayerReport | None:
    try:
        return inspect_pdf_text_layer(pdf)
    except Exception:
        return None


def helper_ocr_output_path(pdf: Path, output_dir: str | Path | None) -> Path:
    root = Path(output_dir) if output_dir else pdf.parent / "ocr_pdf"
    return root / f"{pdf.stem}.ocr.pdf"


def helper_unique_doc_id_and_path(doc_id: str, pdf: Path, output_dir: str | Path) -> tuple[str, Path]:
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{doc_id}.md"
    if not out_path.exists():
        return doc_id, out_path
    try:
        existing_metadata, _body = parse_front_matter(out_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        existing_metadata = {}
    if existing_metadata.get("source_file") == str(pdf):
        return doc_id, out_path
    suffix = hashlib.sha1(str(pdf).encode("utf-8")).hexdigest()[:8]
    unique_doc_id = f"{doc_id}_{suffix}"
    return unique_doc_id, out_dir / f"{unique_doc_id}.md"


def helper_maybe_run_ocr(
    pdf: Path,
    report: PdfTextLayerReport | None,
    *,
    ocr_mode: str,
    ocr_output_dir: str | Path | None,
    ocr_languages: str,
) -> OcrResult:
    if ocr_mode == "never":
        return OcrResult("none", False, str(pdf), "", "")
    if ocr_mode not in {"auto", "force"}:
        raise ValueError("ocr_mode must be one of: never, auto, force")
    should_ocr = ocr_mode == "force" or bool(report and report.needs_ocr)
    if not should_ocr:
        return OcrResult("none", False, str(pdf), "", "")
    return run_ocrmypdf(pdf, helper_ocr_output_path(pdf, ocr_output_dir), languages=ocr_languages, force_ocr=ocr_mode == "force")


def helper_try_run_mineru(
    pdf: Path,
    report: PdfTextLayerReport | None,
    *,
    ocr_mode: str,
) -> MineruRun:
    """Route scanned PDFs to MinerU and return the produced Markdown.

    Returns a no-op ``MineruRun`` when MinerU is disabled, the PDF does not
    need OCR, or ``ocr_mode`` forbids OCR, so callers fall back to the
    OCRmyPDF path unchanged.
    """
    if ocr_mode == "never":
        return MineruRun(applied=False)
    if ocr_mode not in {"auto", "force"}:
        raise ValueError("ocr_mode must be one of: never, auto, force")
    should_ocr = ocr_mode == "force" or bool(report and report.needs_ocr)
    if not should_ocr or not mineru_enabled():
        return MineruRun(applied=False)
    try:
        with tempfile.TemporaryDirectory(prefix="mineru_ocr_", ignore_cleanup_errors=True) as tmp:
            return run_mineru(pdf, output_dir=Path(tmp))
    except Exception as exc:  # noqa: BLE001
        return MineruRun(applied=False, error=f"mineru: {exc}")


def helper_pdf_quality_metadata(report: PdfTextLayerReport | None, prefix: str = "pdf") -> dict[str, str]:
    if report is None:
        return {
            f"{prefix}_text_quality": "unknown",
            f"{prefix}_needs_ocr": "false",
            f"{prefix}_is_scanned": "false",
        }
    return report.as_metadata(prefix)


def helper_ocr_status(ocr_mode: str, source_report: PdfTextLayerReport | None, output_report: PdfTextLayerReport | None, result: OcrResult) -> str:
    source_needs_ocr = bool(source_report and source_report.needs_ocr)
    if ocr_mode == "never":
        return "needed_but_disabled" if source_needs_ocr else "not_needed"
    if result.applied:
        if output_report and output_report.needs_ocr:
            return "applied_needs_review"
        return "applied"
    if source_needs_ocr or ocr_mode == "force":
        if result.error:
            return "needed_unavailable" if "not found" in result.error.lower() else "failed"
        return "needed_not_applied"
    return "not_needed"


def convert_pdf(
    pdf_path: str | Path,
    output_dir: str | Path = DATA_DIR / "markdown_raw",
    *,
    ocr_mode: str = "auto",
    ocr_output_dir: str | Path | None = None,
    ocr_languages: str = "chi_sim+eng",
    document_kind: str = "guideline",
) -> DocumentRecord:
    """Convert one PDF to front-matter Markdown under data/markdown_raw/{doc_id}.md."""
    # 先评估原 PDF 再决定 OCR，最终元数据同时保留原始与转换后质量状态。

    if document_kind not in {"guideline", "consensus"}:
        raise ValueError("document_kind must be guideline or consensus")
    pdf = Path(pdf_path)
    pdf_report = helper_inspect_pdf_for_ingestion(pdf)
    raw: str | None = None
    ocr_result = OcrResult("none", False, str(pdf), str(pdf), "")
    conversion_report: PdfTextLayerReport | None = pdf_report
    # 自动分流:扫描件优先由 MinerU 直接产出 Markdown;失败时回退 OCRmyPDF + PyMuPDF。
    mineru_failure = ""
    if ocr_mode != "never":
        mineru_run = helper_try_run_mineru(pdf, pdf_report, ocr_mode=ocr_mode)
        if mineru_run.applied:
            raw = mineru_run.markdown
            ocr_result = OcrResult("mineru", True, str(pdf), str(pdf), "")
            # MinerU 直接产出 Markdown,不再检查"转换后 PDF"的文本层,故 pdf_* 元数据
            # 为 unknown/false(表示已无需 OCR);ocr_status 仍为 applied。
            conversion_report = None
        elif mineru_run.error:
            mineru_failure = f"mineru: {mineru_run.error}"
    if raw is None:
        fallback = helper_maybe_run_ocr(
            pdf,
            pdf_report,
            ocr_mode=ocr_mode,
            ocr_output_dir=ocr_output_dir,
            ocr_languages=ocr_languages,
        )
        ocr_error = f"{mineru_failure}; " + fallback.error if mineru_failure else fallback.error
        ocr_result = OcrResult(fallback.engine, fallback.applied, fallback.input_pdf, fallback.output_pdf, ocr_error)
        conversion_pdf = Path(ocr_result.output_pdf) if ocr_result.applied else pdf
        conversion_report = helper_inspect_pdf_for_ingestion(conversion_pdf) if ocr_result.applied else pdf_report
        raw = helper_with_pymupdf4llm(conversion_pdf) or helper_with_pymupdf(conversion_pdf)
    title = extract_title(raw, pdf)
    source_institution = extract_source_institution(pdf, raw, title=title)
    publication_date = extract_publication_date(pdf, raw)
    file_sha = sha256_file(pdf)
    doc_id = make_doc_id(source_institution, publication_date, file_sha)
    doc_id, out_path = helper_unique_doc_id_and_path(doc_id, pdf, output_dir)
    body = helper_ensure_title_heading(raw, title)
    abstract = extract_abstract(body)
    department_result = classify_clinical_departments(title, abstract, body)
    clinical_department = str(department_result["clinical_department"])
    metadata = {
        "id": doc_id,
        "title": title,
        "publication_date": publication_date,
        "source_institution": source_institution,
        "source_file": str(pdf),
        "clinical_department": clinical_department,
        "clinical_departments": department_result["clinical_departments"],
        "department_scope": department_result["department_scope"],
        "document_kind": document_kind,
        **helper_pdf_quality_metadata(pdf_report, "source_pdf"),
        **helper_pdf_quality_metadata(conversion_report, "pdf"),
        "ocr_engine": ocr_result.engine,
        "ocr_applied": str(ocr_result.applied).lower(),
        "ocr_status": helper_ocr_status(ocr_mode, pdf_report, conversion_report, ocr_result),
        "ocr_error": ocr_result.error,
    }
    markdown = dump_front_matter(metadata, body)
    out_path.write_text(markdown, encoding="utf-8", newline="\n")
    return DocumentRecord(
        doc_id=doc_id,
        title=title,
        publication_date=publication_date,
        source_institution=source_institution,
        clinical_department=clinical_department,
        clinical_departments=department_result["clinical_departments"],
        department_scope=str(department_result["department_scope"]),
        document_kind=document_kind,
        source_file=str(pdf),
        markdown_raw_path=str(out_path),
        markdown_clean_path="",
        abstract=abstract,
        content_sha256=sha256_text(markdown),
        source_pdf_text_quality=metadata.get("source_pdf_text_quality", ""),
        source_pdf_needs_ocr=metadata.get("source_pdf_needs_ocr") == "true",
        source_pdf_is_scanned=metadata.get("source_pdf_is_scanned") == "true",
        pdf_text_quality=metadata.get("pdf_text_quality", ""),
        pdf_needs_ocr=metadata.get("pdf_needs_ocr") == "true",
        pdf_is_scanned=metadata.get("pdf_is_scanned") == "true",
        ocr_engine=ocr_result.engine,
        ocr_applied=ocr_result.applied,
        ocr_status=metadata.get("ocr_status", ""),
        ocr_error=ocr_result.error,
    )


def helper_deduplicate_doc_id(record: DocumentRecord, seen: set[str]) -> DocumentRecord:
    if record.doc_id not in seen:
        seen.add(record.doc_id)
        return record
    suffix = hashlib.sha1(record.source_file.encode("utf-8")).hexdigest()[:8]
    old_doc_id = record.doc_id
    record.doc_id = f"{old_doc_id}_{suffix}"
    raw_path = Path(record.markdown_raw_path)
    markdown = raw_path.read_text(encoding="utf-8")
    markdown = markdown.replace(f'id: "{old_doc_id}"', f'id: "{record.doc_id}"', 1)
    new_path = raw_path.with_name(f"{record.doc_id}.md")
    new_path.write_text(markdown, encoding="utf-8", newline="\n")
    record.markdown_raw_path = str(new_path)
    seen.add(record.doc_id)
    return record


def helper_convert_pdf_worker(payload: tuple[str, str, str, str | None, str, str]) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    pdf_path, output_dir, ocr_mode, ocr_output_dir, ocr_languages, document_kind = payload
    try:
        return (
            dump_model(
                convert_pdf(
                    pdf_path,
                    output_dir,
                    ocr_mode=ocr_mode,
                    ocr_output_dir=ocr_output_dir,
                    ocr_languages=ocr_languages,
                    document_kind=document_kind,
                )
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return None, {"source_file": str(pdf_path), "error": str(exc)}


def helper_progress(done: int, total: int, progress_every: int) -> None:
    if progress_every <= 0:
        return
    if done == total or done % progress_every == 0:
        print(json.dumps({"processed": done, "total": total}, ensure_ascii=False), file=sys.stderr, flush=True)


def convert_all(
    input_dir: str | Path = DATA_DIR / "raw_pdf",
    output_dir: str | Path = DATA_DIR / "markdown_raw",
    manifest_path: str | Path = DATA_DIR / "documents_raw.jsonl",
    *,
    ocr_mode: str = "auto",
    ocr_output_dir: str | Path | None = None,
    ocr_languages: str = "chi_sim+eng",
    workers: int = 1,
    progress_every: int = 100,
    document_kind: str = "guideline",
    append: bool = False,
) -> dict[str, Any]:
    if document_kind not in {"guideline", "consensus"}:
        raise ValueError("document_kind must be guideline or consensus")
    all_existing_records = list(read_jsonl(manifest_path)) if append and Path(manifest_path).exists() else []
    existing_records = [
        record for record in all_existing_records
        if (path := Path(record.get("source_file") or "")).is_file()
    ]
    stale_records_skipped = len(all_existing_records) - len(existing_records)
    existing_hashes = {
        sha256_file(path) for record in existing_records
        if (path := Path(record.get("source_file") or "")).is_file()
    }
    raw_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    pdf_paths = []
    duplicate_files = 0
    seen_hashes = set(existing_hashes)
    for path in sorted(Path(input_dir).rglob("*.pdf")):
        # 内容哈希用于跳过完全相同的 PDF；文件名不同不代表证据不同。
        file_hash = sha256_file(path)
        if file_hash in seen_hashes:
            duplicate_files += 1
            continue
        seen_hashes.add(file_hash)
        pdf_paths.append(path)
    total = len(pdf_paths)
    # 单进程路径便于调试；多进程路径只传递可序列化参数，避免共享 PDF 句柄。
    if workers <= 1:
        for done, pdf_path in enumerate(pdf_paths, start=1):
            try:
                raw_records.append(
                    dump_model(
                        convert_pdf(
                            pdf_path,
                            output_dir,
                            ocr_mode=ocr_mode,
                            ocr_output_dir=ocr_output_dir,
                            ocr_languages=ocr_languages,
                            document_kind=document_kind,
                        )
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"source_file": str(pdf_path), "error": str(exc)})
            helper_progress(done, total, progress_every)
    else:
        payloads = [
            (str(pdf_path), str(output_dir), ocr_mode, str(ocr_output_dir) if ocr_output_dir else None, ocr_languages, document_kind)
            for pdf_path in pdf_paths
        ]
        # 使用 as_completed 收集结果，使慢 PDF 不会阻塞其他已完成任务落盘。
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(helper_convert_pdf_worker, payload) for payload in payloads]
            for done, future in enumerate(as_completed(futures), start=1):
                record, error = future.result()
                if record is not None:
                    raw_records.append(record)
                if error is not None:
                    errors.append(error)
                helper_progress(done, total, progress_every)

    records: list[dict[str, Any]] = list(existing_records)
    # 内容标题可能生成相同 doc_id，最终写 manifest 前按来源路径追加稳定后缀。
    seen_doc_ids: set[str] = {record["doc_id"] for record in existing_records}
    for record in sorted(raw_records, key=lambda item: item.get("source_file", "")):
        records.append(dump_model(helper_deduplicate_doc_id(DocumentRecord(**record), seen_doc_ids)))
    write_jsonl(manifest_path, records)
    write_jsonl(Path(manifest_path).with_name(f"pdf_to_md_{document_kind}_errors.jsonl"), errors)
    return {
        "converted": len(raw_records),
        "total_manifest_records": len(records),
        "duplicates_skipped": duplicate_files,
        "stale_records_skipped": stale_records_skipped,
        "document_kind": document_kind,
        "failed": len(errors),
        "manifest": str(manifest_path),
        "ocr_mode": ocr_mode,
        "ocr_languages": ocr_languages,
        "ocr_candidates": sum(1 for record in records if record.get("source_pdf_needs_ocr")),
        "ocr_applied": sum(1 for record in records if record.get("ocr_applied")),
        "ocr_needs_review": sum(1 for record in records if record.get("ocr_status") in {"needed_unavailable", "failed", "applied_needs_review"}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA_DIR / "raw_pdf"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "markdown_raw"))
    parser.add_argument("--manifest", default=str(DATA_DIR / "documents_raw.jsonl"))
    parser.add_argument("--ocr-mode", choices=["never", "auto", "force"], default="auto")
    parser.add_argument("--ocr-output-dir", default=None)
    parser.add_argument("--ocr-languages", default="chi_sim+eng")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--document-kind", choices=["guideline", "consensus"], default="guideline")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            convert_all(
                args.input_dir,
                args.output_dir,
                args.manifest,
                ocr_mode=args.ocr_mode,
                ocr_output_dir=args.ocr_output_dir,
                ocr_languages=args.ocr_languages,
                workers=args.workers,
                progress_every=args.progress_every,
                document_kind=args.document_kind,
                append=args.append,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Tiny YAML-like front matter parser/writer for flat metadata."""

from __future__ import annotations

from collections import OrderedDict


FRONT_MATTER_KEYS = [
    "id",
    "title",
    "publication_date",
    "source_institution",
    "source_file",
    "clinical_department",
    "source_pdf_text_quality",
    "source_pdf_needs_ocr",
    "source_pdf_is_scanned",
    "pdf_page_count",
    "pdf_text_chars",
    "pdf_text_chars_per_page",
    "pdf_low_text_page_ratio",
    "pdf_image_page_ratio",
    "pdf_scanned_page_ratio",
    "pdf_is_scanned",
    "pdf_needs_ocr",
    "pdf_text_quality",
    "ocr_engine",
    "ocr_applied",
    "ocr_status",
    "ocr_error",
    "cleaning_quality",
    "cleaning_flags",
]


def dump_front_matter(metadata: dict[str, object], body: str) -> str:
    lines = ["---"]
    for key in FRONT_MATTER_KEYS:
        value = str(metadata.get(key, "") or "").replace('"', '\\"')
        lines.append(f'{key}: "{value}"')
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.strip() + "\n"


def parse_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end].strip().splitlines()
    metadata: dict[str, str] = OrderedDict()
    for line in raw:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace('\\"', '"')
        metadata[key.strip()] = value
    body = text[text.find("\n", end + 1) + 1 :]
    return metadata, body

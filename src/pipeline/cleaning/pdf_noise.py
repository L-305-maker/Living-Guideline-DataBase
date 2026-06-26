from __future__ import annotations

import re
from collections import Counter
from typing import Optional, List, Tuple

from src.pipeline.cleaning.base_cleaner import clean_inline_text, should_drop_line
from src.common.extraction_common import JsonDict

PDF_NOISE_LINE_RE = re.compile(
    r"(?i)\b(?:"
    r"all rights reserved|copyright|subject to notice of rights|terms of use|"
    r"downloaded from|for personal use only|unauthorized reproduction|"
    r"permission is granted|creative commons licence|isbn|issn|"
    r"www\.[^\s]+|https?://[^\s]+|doi:\s*10\."
    r")\b"
)
PRESERVE_SECTION_MARKER_RE = re.compile(
    r"(?i)^\s*(references|bibliography|recommendations?|methods?|methodology|evidence|background|rationale|remarks?)\s*$"
)
REPEATED_FOOTER_HINT_RE = re.compile(
    r"(?i)\b(?:"
    r"guideline|guidelines|journal|copyright|rights reserved|page|doi|"
    r"www\.|https?://|downloaded|published by|licensed under"
    r")\b"
)
INLINE_PDF_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)\bISBN(?:\s+(?:97[89][-\s]*)?\d[-\d\s]{6,}){1,3}"),
    re.compile(r"(?i)\b(?:Licence|License):\s*CC BY[-A-Z0-9. ]+"),
    re.compile(r"(?i)\bCataloguing-in-Publication\s*\(CIP\)\s*data\.[^.]{0,220}\."),
    re.compile(r"(?i)\bOpen Access\s+This article is licensed under[^.]{0,360}\."),
    re.compile(r"(?i)\bDownloaded\s+from\s+[^\s.]+(?:\s+by\s+[^\n.]{0,120})?"),
    re.compile(r"(?i)\bAll\s+rights\s+reserved\.?"),
    re.compile(r"(?i)Allrightsreserved\.?"),
    re.compile(r"(?i)SubjecttoNoticeofrights\.?"),
    re.compile(r"(?i)Seetheoriginalguidanceat(?:https?://)?www\.nice\.org\.uk/guidance/?\S*(?:\s*\|\s*\w+)?"),
    re.compile(r"(?i)Seewww\.nice\.org\.uk/guidance/?\S*(?:\s*\|\s*\w+)?"),
    re.compile(r"(?i)(?:https?://)?www\.nice\.org\.uk/guidance/?\S*(?:\s*\|\s*\w+)?"),
    re.compile(r"(?i)(?:https?://)?doi\.org/10\.\S+"),
    re.compile(r"(?i)(?:https?://)?dx\.doi\.org/10\.\S+"),
]

def remove_inline_pdf_boilerplate(text: str) -> Tuple[str, int]:
    cleaned = str(text or "")
    removed = 0
    for pattern in INLINE_PDF_BOILERPLATE_PATTERNS:
        cleaned, count = pattern.subn(" ", cleaned)
        removed += count
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned, removed

def enhanced_pdf_noise_reason(line: str, repeated_lines: set[str]) -> Optional[str]:
    stripped = clean_inline_text(line)
    if not stripped:
        return "blank"
    if PRESERVE_SECTION_MARKER_RE.match(stripped):
        return None
    if should_drop_line(stripped):
        return "base_drop_rule"
    if PDF_NOISE_LINE_RE.search(stripped) and len(stripped) <= 220:
        return "pdf_boilerplate"
    if stripped in repeated_lines and len(stripped) <= 160 and REPEATED_FOOTER_HINT_RE.search(stripped):
        return "repeated_header_footer"
    if len(stripped) <= 3 and not re.search(r"[A-Za-z0-9]", stripped):
        return "symbol_noise"
    return None

def enhance_pdf_noise_cleaning(record: JsonDict) -> JsonDict:
    cleaned = dict(record)
    content = str(cleaned.get("content") or "")
    if not content.strip():
        cleaned["cleaning_report"] = {
            **dict(cleaned.get("cleaning_report") or {}),
            "enhanced_pdf_noise_cleaning": {
                "applied": False,
                "reason": "empty_content",
            },
        }
        return cleaned

    content, inline_removed = remove_inline_pdf_boilerplate(content)
    raw_lines = content.splitlines()
    normalized_lines = [clean_inline_text(line) for line in raw_lines]
    line_counts = Counter(line for line in normalized_lines if line)
    repeated_lines = {line for line, count in line_counts.items() if count >= 3}

    kept_lines: List[str] = []
    dropped_reasons: Counter[str] = Counter()
    previous = ""

    for line in normalized_lines:
        reason = enhanced_pdf_noise_reason(line, repeated_lines)
        if reason:
            dropped_reasons[reason] += 1
            continue
        if line == previous:
            dropped_reasons["consecutive_duplicate"] += 1
            continue
        kept_lines.append(line)
        previous = line

    cleaned["content"] = "\n".join(kept_lines).strip()
    cleaned["cleaning_report"] = {
        **dict(cleaned.get("cleaning_report") or {}),
        "enhanced_pdf_noise_cleaning": {
            "applied": True,
            "original_line_count": len(raw_lines),
            "cleaned_line_count": len(kept_lines),
            "dropped_line_count": sum(dropped_reasons.values()),
            "dropped_reasons": dict(dropped_reasons),
            "inline_boilerplate_removed_count": inline_removed,
        },
    }
    return cleaned

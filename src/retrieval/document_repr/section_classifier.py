"""Heuristic section and chunk classification for guideline search signals."""

from __future__ import annotations

import re
from typing import Iterable


SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "recommendation": re.compile(
        r"(recommendation|recommendations|we recommend|we suggest|should be|is recommended|"
        r"\u63a8\u8350|\u5efa\u8bae|\u5e94\u8be5|\u5b9c|\u4e0d\u5efa\u8bae)",
        re.I,
    ),
    "scope_population": re.compile(
        r"(scope|population|target population|intended audience|applicability|eligib|"
        r"\u8303\u56f4|\u9002\u7528|\u4eba\u7fa4|\u76ee\u6807\u4eba\u7fa4|\u9002\u7528\u5bf9\u8c61)",
        re.I,
    ),
    "pico": re.compile(
        r"(\bPICO\b|clinical question|key question|research question|patient.?intervention|"
        r"\u4e34\u5e8a\u95ee\u9898|\u5faa\u8bc1\u95ee\u9898|\u5173\u952e\u95ee\u9898)",
        re.I,
    ),
    "evidence": re.compile(
        r"(evidence|certainty|quality of evidence|systematic review|meta-analysis|GRADE|"
        r"\u8bc1\u636e|\u8bc1\u636e\u8d28\u91cf|\u8bc1\u636e\u7b49\u7ea7|\u7cfb\u7edf\u8bc4\u4ef7|\u835f\u8403)",
        re.I,
    ),
    "table": re.compile(r"(\btable\b|\bfig(?:ure)?\b|\balgorithm\b|\u8868\s*\d*|\u56fe\s*\d*|\u6d41\u7a0b)", re.I),
    "conclusion": re.compile(r"(conclusion|summary|take-home|key points|\u7ed3\u8bba|\u603b\u7ed3|\u8981\u70b9)", re.I),
    "background": re.compile(r"(background|introduction|epidemiology|\u80cc\u666f|\u524d\u8a00|\u5f15\u8a00)", re.I),
    "reference": re.compile(r"^(references|bibliography|\u53c2\u8003\u6587\u732e)$", re.I),
}

CHUNK_TYPE_WEIGHTS: dict[str, float] = {
    "recommendation": 1.55,
    "table": 1.35,
    "pico": 1.25,
    "evidence": 1.20,
    "scope_population": 1.05,
    "conclusion": 1.00,
    "background": 0.50,
    "reference": 0.10,
    "other": 0.90,
}


def _join_path(section_path: Iterable[object] | None) -> str:
    return " ".join(str(item) for item in (section_path or []) if item is not None)


def classify_section(heading: str = "", section_path: Iterable[object] | None = None, content: str = "") -> str:
    """Return the highest-value semantic type for a section."""

    path_text = _join_path(section_path)
    heading_text = heading or path_text
    haystack = f"{heading_text}\n{path_text}\n{content[:1200]}"
    normalized_heading = (heading_text or "").strip()
    if normalized_heading and SECTION_PATTERNS["reference"].match(normalized_heading):
        return "reference"
    for section_type in [
        "recommendation",
        "scope_population",
        "pico",
        "evidence",
        "table",
        "conclusion",
        "background",
    ]:
        if SECTION_PATTERNS[section_type].search(haystack):
            return section_type
    return "other"


def classify_chunk(chunk: dict[str, object]) -> str:
    """Classify an existing chunk record using section path and content."""

    chunk_type = str(chunk.get("chunk_type") or "").strip()
    if chunk_type:
        return chunk_type
    if chunk.get("is_reference_section"):
        return "reference"
    section_path = chunk.get("section_path")
    heading = ""
    if isinstance(section_path, list) and section_path:
        heading = str(section_path[-1])
    return classify_section(heading=heading, section_path=section_path if isinstance(section_path, list) else [], content=str(chunk.get("content") or ""))


def chunk_type_weight(chunk_type: str) -> float:
    return CHUNK_TYPE_WEIGHTS.get(chunk_type, CHUNK_TYPE_WEIGHTS["other"])

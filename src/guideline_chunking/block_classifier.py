"""Rule-based coarse block classification.

The labels are candidate labels for downstream processing, not final medical
structure extraction.
"""

from __future__ import annotations

import re


RECOMMENDATION_RE = re.compile(
    r"\b(recommendation|recommendations|we recommend|we suggest|should|should not|offer|do not offer|"
    r"consider|do not routinely|is recommended|is not recommended|strong recommendation|"
    r"conditional recommendation|good practice statement)\b",
    re.I,
)
BAD_RECOMMENDATION_RE = re.compile(
    r"\b(references|bibliography|methods|conflict of interest|declaration|acknowledgements|"
    r"copyright|appendix index|table of contents)\b",
    re.I,
)
EVIDENCE_RE = re.compile(
    r"\b(evidence|evidence summary|summary of evidence|quality of evidence|certainty of evidence|GRADE|"
    r"evidence profile|summary of findings|risk of bias|imprecision|inconsistency|indirectness|"
    r"publication bias|effect estimate|relative effect|absolute effect|confidence interval)\b",
    re.I,
)
PICO_RE = re.compile(
    r"\b(PICO|review question|clinical question|key question|population|intervention|comparator|"
    r"outcome|outcomes)\b",
    re.I,
)
RATIONALE_RE = re.compile(
    r"\b(rationale|evidence to decision|EtD|committee discussion|committee considerations|"
    r"benefits and harms|values and preferences|resource use|equity|acceptability|feasibility|"
    r"implementation considerations)\b",
    re.I,
)
SCOPE_RE = re.compile(
    r"\b(scope|target population|applicability|inclusion|exclusion|who this guideline is for|"
    r"who should use this guideline)\b",
    re.I,
)
METHOD_RE = re.compile(r"\b(methods|methodology|search strategy|conflict of interest|funding|declaration)\b", re.I)
REFERENCE_RE = re.compile(r"\b(references|bibliography|appendix)\b", re.I)
BACKGROUND_RE = re.compile(r"\b(background|introduction|overview|epidemiology)\b", re.I)
ALGORITHM_RE = re.compile(r"\b(algorithm|pathway|flowchart|flow chart|decision tree|流程|路径)\b", re.I)
ABSTRACT_RE = re.compile(r"\b(abstract|summary)\b", re.I)
POPULATION_RE = re.compile(r"\b(target population|population|intended audience)\b", re.I)


def classify_block_type(block_text: str, heading_path: list[str]) -> str:
    text = block_text or ""
    heading_text = " > ".join(heading_path or [])
    combined = f"{heading_text}\n{text}".strip()
    low_value_context = bool(BAD_RECOMMENDATION_RE.search(combined))

    if helper_looks_like_markdown_table(text):
        return "table"
    if not heading_path and len(text.splitlines()) == 1 and len(text) < 180:
        return "title"
    if REFERENCE_RE.search(combined):
        return "reference"
    if METHOD_RE.search(combined):
        return "method"
    if SCOPE_RE.search(combined):
        return "scope_candidate"
    if POPULATION_RE.search(combined) and not helper_looks_like_pico(text):
        return "population_candidate"
    if RECOMMENDATION_RE.search(combined) and not low_value_context:
        return "recommendation_candidate"
    if RATIONALE_RE.search(combined):
        return "rationale_candidate"
    if EVIDENCE_RE.search(combined):
        return "evidence_candidate"
    if PICO_RE.search(combined):
        return "pico_candidate"
    if ALGORITHM_RE.search(combined):
        return "algorithm_candidate"
    if ABSTRACT_RE.search(combined):
        return "abstract"
    if BACKGROUND_RE.search(combined):
        return "background"
    return "unknown"


def helper_looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("|") and line.endswith("|") for line in lines)


def helper_looks_like_pico(text: str) -> bool:
    lower = text.lower()
    return sum(label in lower for label in ("population", "intervention", "comparator", "outcome")) >= 2

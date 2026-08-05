# 基于规则的粗粒度 block 分类器。
#
# 关键设计：
# - 12 类候选标签（recommendation / evidence / pico / rationale / scope / ...），均为下游候选标签；
# - 优先级：table / reference / method / scope / population / recommendation / rationale / evidence / pico / algorithm / abstract / background → unknown；
# - BAD_RECOMMENDATION_RE 抑制误判（如 references / bibliography 不能算 recommendation）；
# - helper_looks_like_pico：PICO 包含 4 要素中至少 2 个才算 pico，避免 population 误命中。

# 注意：标签是"候选标签"，最终医学结构化需由 LLM 或人工 review 兜底。
"""Rule-based coarse block classification.

The labels are candidate labels for downstream processing, not final medical
structure extraction.
"""

from __future__ import annotations

import re


# 推荐类关键词（含 weak/strong recommendation），配合 BAD_RECOMMENDATION_RE 抑制误判。
RECOMMENDATION_RE = re.compile(
    r"\b(recommendation|recommendations|we recommend|we suggest|should|should not|offer|do not offer|"
    r"consider|do not routinely|is recommended|is not recommended|strong recommendation|"
    r"conditional recommendation|good practice statement)\b",
    re.I,
)
# 反向抑制：references / bibliography / appendix 等区域不应被误判为 recommendation。
BAD_RECOMMENDATION_RE = re.compile(
    r"\b(references|bibliography|methods|conflict of interest|declaration|acknowledgements|"
    r"copyright|appendix index|table of contents)\b",
    re.I,
)
# 证据类关键词（含 GRADE / meta-analysis / 偏倚评估）。
EVIDENCE_RE = re.compile(
    r"\b(evidence|evidence summary|summary of evidence|quality of evidence|certainty of evidence|GRADE|"
    r"evidence profile|summary of findings|risk of bias|imprecision|inconsistency|indirectness|"
    r"publication bias|effect estimate|relative effect|absolute effect|confidence interval)\b",
    re.I,
)
# PICO 关键词（含 population / intervention / comparator / outcome）。
PICO_RE = re.compile(
    r"\b(PICO|review question|clinical question|key question|population|intervention|comparator|"
    r"outcome|outcomes)\b",
    re.I,
)
# 理由 / 决策依据（EtD framework）。
RATIONALE_RE = re.compile(
    r"\b(rationale|evidence to decision|EtD|committee discussion|committee considerations|"
    r"benefits and harms|values and preferences|resource use|equity|acceptability|feasibility|"
    r"implementation considerations)\b",
    re.I,
)
# 范围 / 适用人群。
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
    """对单 block 文本做粗粒度分类，返回候选标签。

    决策流程：
    1. 纯表格 → "table"；
    2. 单行 < 180 字符 + 无 heading → "title"；
    3. reference / method 高优先级（避免推荐类抢占）；
    4. scope / population_candidate（避免被 PICO 误判） → recommendation → rationale → evidence → pico → algorithm → abstract → background；
    5. 全部未命中 → "unknown"。
    """
    # 分类规则按强信号优先级依次判定，默认类型只能在所有专用规则均未命中后使用。
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
    """判定是否为合法 Markdown 表格：所有非空行都以 '|' 开头和结尾。"""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("|") and line.endswith("|") for line in lines)


def helper_looks_like_pico(text: str) -> bool:
    """PICO 包含至少 2 个要素（population / intervention / comparator / outcome）才判为 PICO。"""
    lower = text.lower()
    return sum(label in lower for label in ("population", "intervention", "comparator", "outcome")) >= 2
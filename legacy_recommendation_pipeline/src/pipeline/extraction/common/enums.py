"""抽取公共工具文件：提供抽取阶段共享枚举、映射、校验器和通用辅助逻辑。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from typing import Any


RECOMMENDATION_DIRECTIONS = {"for", "against", "neutral", "unclear"}
RECOMMENDATION_STRENGTHS = {"strong", "conditional", "weak", "good_practice", "unclear"}
CERTAINTIES = {"high", "moderate", "low", "very_low", "unclear", "not_reported"}
GRADE_DOMAIN_JUDGEMENTS = {"no_concern", "serious", "very_serious", "serious_or_concern", "unclear", "not_reported"}
PUBLICATION_BIAS_JUDGEMENTS = {"undetected", "suspected", "strongly_suspected", "unclear", "not_reported"}
EVIDENCE_TYPES = {
    "evidence_summary",
    "individual_study",
    "meta_analysis",
    "systematic_review",
    "guideline_summary",
    "unclear",
}
STUDY_DESIGNS = {"RCT", "cohort", "meta_analysis", "systematic_review", "unclear"}
EFFECT_DIRECTIONS = {"benefit", "harm", "no_effect", "mixed", "uncertain"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_direction(value: Any) -> str:
    text = _norm(value)
    if text in {"for", "positive", "in_favor", "recommend_for"}:
        return "for"
    if text in {"against", "negative", "recommend_against", "do_not_use"}:
        return "against"
    if text in {"neutral", "no_recommendation"}:
        return "neutral"
    return text if text in RECOMMENDATION_DIRECTIONS else "unclear"


def normalize_strength(value: Any) -> str:
    text = _norm(value)
    if text in {"strong", "strong_recommendation"}:
        return "strong"
    if text in {"conditional", "conditional_recommendation", "suggest", "weak_for"}:
        return "conditional"
    if text in {"weak", "weak_recommendation"}:
        return "weak"
    if text in {"good_practice", "good_practice_statement", "gps"}:
        return "good_practice"
    return text if text in RECOMMENDATION_STRENGTHS else "unclear"


def normalize_certainty(value: Any) -> str:
    text = _norm(value)
    text = text.replace("_certainty_of_evidence", "").replace("_quality_of_evidence", "")
    if text in {"very_low", "verylow"}:
        return "very_low"
    if text in {"high", "moderate", "low", "unclear", "not_reported"}:
        return text
    if text in {"none", "not_extracted", "not_applicable", ""}:
        return "not_reported"
    return "unclear"


def normalize_grade_domain(value: Any) -> str:
    text = _norm(value)
    if text in {"serious_concern", "serious_or_concern", "concern", "some_concern"}:
        return "serious_or_concern"
    if text in {"no_concern", "not_serious", "none"}:
        return "no_concern"
    if text in {"not_extracted", "not_applicable", ""}:
        return "not_reported"
    return text if text in GRADE_DOMAIN_JUDGEMENTS else "unclear"


def normalize_publication_bias(value: Any) -> str:
    text = _norm(value)
    if text in {"not_detected", "no_concern", "none"}:
        return "undetected"
    if text in {"strongly_suspected", "strong_suspicion"}:
        return "strongly_suspected"
    if text in {"not_extracted", "not_applicable", ""}:
        return "not_reported"
    return text if text in PUBLICATION_BIAS_JUDGEMENTS else "unclear"


def normalize_evidence_type(value: Any) -> str:
    text = _norm(value)
    if text in {"summary", "evidence_summary"}:
        return "evidence_summary"
    if text in {"individual_study", "study"}:
        return "individual_study"
    if text in {"meta_analysis", "metaanalysis"}:
        return "meta_analysis"
    if text in {"systematic_review"}:
        return "systematic_review"
    if text in {"guideline", "guideline_summary"}:
        return "guideline_summary"
    return text if text in EVIDENCE_TYPES else "unclear"


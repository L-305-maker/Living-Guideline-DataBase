from __future__ import annotations

import re
from typing import List, Tuple

from src.common.extraction_common import normalize_text


ActionPattern = Tuple[str, str]

ACTION_PATTERN_SPECS: List[ActionPattern] = [
    ("we_recommend", r"\bwe\s+recommend\b"),
    ("we_suggest", r"\bwe\s+suggest\b"),
    ("recommendation_numbered", r"\brecommendation\s*\d+"),
    ("panel_recommends", r"\b(?:the\s+)?panel\s+recommends?\b"),
    ("aasm_recommends", r"\b(?:the\s+)?aasm\s+recommends?\b"),
    ("cdc_recommends", r"\b(?:the\s+)?cdc\s+recommends?\b"),
    ("guideline_body_recommends", r"\b(?:committee|task\s+force|guideline|who)\s+(?:recommend|recommends|suggest|suggests)\b"),
    ("clinicians_should", r"\bclinicians?\s+should\b"),
    ("patients_should", r"\bpatients?\s+should\b"),
    ("subject_should_receive", r"\b[A-Z][A-Za-z\s]{0,60}\s+should\s+receive\b"),
    ("direct_clinical_action", r"^\s*(?:offer|provide|consider|refer|advise|administer|use)\b"),
    ("subject_should_be_offered", r"\b(?:adults?|children|patients?|people|individuals?)\s+should\s+be\s+(?:offered|provided|referred|advised)\b"),
    ("recommended_that", r"\bit\s+is\s+recommended\s+that\b"),
    ("suggested_that", r"\bit\s+is\s+suggested\s+that\b"),
    ("recommends_against", r"\brecommends?\s+against\b"),
    ("suggests_against", r"\bsuggests?\s+against\b"),
    ("should_not", r"\bshould\s+not\b"),
    ("not_recommended", r"\bis\s+not\s+recommended\b"),
]

ACTION_RECOMMENDATION_RE = re.compile("|".join(f"(?:{pattern})" for _, pattern in ACTION_PATTERN_SPECS), re.I)
RECOMMENDATION_SIGNAL_RE = ACTION_RECOMMENDATION_RE


# 返回命中的动作推荐模式名称；router、extractor、audit 共享同一套口径。
def action_patterns(text: str) -> List[str]:
    clean = normalize_text(text)
    return [name for name, pattern in ACTION_PATTERN_SPECS if re.search(pattern, clean, re.I)]


def has_action_pattern(text: str) -> bool:
    return bool(ACTION_RECOMMENDATION_RE.search(normalize_text(text)))


def matched_action_texts(text: str) -> List[str]:
    clean = normalize_text(text)
    return sorted(set(match.group(0).lower() for match in ACTION_RECOMMENDATION_RE.finditer(clean)))

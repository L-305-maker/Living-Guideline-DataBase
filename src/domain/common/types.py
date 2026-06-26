from __future__ import annotations

from typing import Any, Dict, Literal


JsonDict = Dict[str, Any]

Direction = Literal["for", "against", "neutral", "no_recommendation", "unclear"]
RecommendationStrength = Literal["strong", "conditional", "weak", "good_practice", "none", "unclear"]
Certainty = Literal["high", "moderate", "low", "very_low", "none", "unclear", "not_reported"]
ExtractionMethod = Literal["rule", "model", "hybrid", "manual"]
CandidateStatus = Literal["pending", "accepted", "rejected", "needs_review", "merged"]
GuidelineType = Literal["living", "standard", "rapid", "consensus"]
GuidelineStatus = Literal["draft", "active", "archived", "retired"]
PicoStatus = Literal["active", "retired", "merged", "under_review"]
Priority = Literal["high", "medium", "low"]
GradeSystem = Literal["GRADE", "ACC_AHA_COR_LOE", "KDIGO", "USPSTF", "ADA", "NICE", "unknown"]
GradeDomainJudgement = Literal["no_concern", "serious", "very_serious", "serious_or_concern", "unclear", "not_reported"]
PublicationBiasJudgement = Literal["undetected", "suspected", "strongly_suspected", "unclear", "not_reported"]
ModelTaskType = Literal[
    "recommendation_extraction",
    "grade_extraction",
    "pico_extraction",
    "evidence_extraction",
    "evidence_matching",
    "update_prediction",
    "combined_extraction",
]
ModelMethod = Literal["rule", "model", "hybrid", "manual"]
InputEntityType = Literal["record", "section", "block", "chunk", "paper", "recommendation_candidate"]
RecommendationChangeType = Literal["new", "modified", "unchanged", "withdrawn", "reaffirmed"]
RecommendationStatus = Literal["draft", "active", "withdrawn", "superseded"]
ArticleType = Literal["RCT", "cohort", "case_control", "meta_analysis", "systematic_review", "guideline", "other", "unclear"]
ScreeningStatus = Literal["unscreened", "included", "excluded", "uncertain", "duplicate", "pending"]
StudyDesign = Literal[
    "RCT",
    "cohort",
    "case_control",
    "cross_sectional",
    "meta_analysis",
    "systematic_review",
    "case_series",
    "guideline",
    "unclear",
]
EffectDirection = Literal["benefit", "harm", "no_effect", "uncertain", "mixed"]
UpdateType = Literal[
    "new_recommendation",
    "text_modified",
    "strength_changed",
    "direction_changed",
    "certainty_changed",
    "withdrawn",
    "reaffirmed",
    "evidence_updated",
]

Block_type = Literal[
    "heading","paragraph","list_item","table","unknown"
]

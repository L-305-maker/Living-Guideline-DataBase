"""Small evaluation metrics for reviewed examples."""

from __future__ import annotations

from src.guideline_information.enums import ReviewDecisionType
from src.guideline_information.models import ReviewDecision


def decision_counts(decisions: list[ReviewDecision]) -> dict[str, int]:
    counts = {item.value: 0 for item in ReviewDecisionType}
    for decision in decisions:
        counts[decision.decision.value] += 1
    return counts

"""Review state machine."""

from __future__ import annotations

from src.guideline_information.enums import ReviewDecisionType, ReviewStatus
from src.guideline_information.models import ReviewDecision, ReviewSample
from src.guideline_information.review.models import ReviewStateChange


class InvalidReviewTransition(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[ReviewStatus, set[ReviewStatus]] = {
    ReviewStatus.PENDING: {ReviewStatus.ASSIGNED, ReviewStatus.CANCELLED, ReviewStatus.INVALID_SOURCE},
    ReviewStatus.ASSIGNED: {ReviewStatus.IN_REVIEW, ReviewStatus.CANCELLED, ReviewStatus.INVALID_SOURCE},
    ReviewStatus.IN_REVIEW: {ReviewStatus.SUBMITTED, ReviewStatus.CANCELLED, ReviewStatus.INVALID_SOURCE},
    ReviewStatus.SUBMITTED: {ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.NEEDS_ADJUDICATION},
    ReviewStatus.NEEDS_ADJUDICATION: {ReviewStatus.ADJUDICATED, ReviewStatus.REJECTED},
    ReviewStatus.ADJUDICATED: {ReviewStatus.APPROVED},
    ReviewStatus.APPROVED: {ReviewStatus.SUPERSEDED},
    ReviewStatus.REJECTED: {ReviewStatus.SUPERSEDED},
    ReviewStatus.INVALID_SOURCE: {ReviewStatus.SUPERSEDED},
    ReviewStatus.SUPERSEDED: set(),
    ReviewStatus.CANCELLED: set(),
}


def transition_sample(sample: ReviewSample, to_status: ReviewStatus, actor_id: str, reason: str = "") -> ReviewSample:
    allowed = ALLOWED_TRANSITIONS[sample.review_status]
    if to_status not in allowed:
        raise InvalidReviewTransition(f"Invalid review transition: {sample.review_status.value} -> {to_status.value}")
    change = ReviewStateChange(
        sample_id=sample.sample_id,
        from_status=sample.review_status,
        to_status=to_status,
        actor_id=actor_id,
        reason=reason,
    )
    history = list(sample.state_history)
    history.append(change.model_dump(mode="json"))
    return sample.model_copy(update={"review_status": to_status, "state_history": history})


def submit_decisions(sample: ReviewSample, decisions: list[ReviewDecision], actor_id: str) -> ReviewSample:
    if sample.review_status not in {ReviewStatus.IN_REVIEW, ReviewStatus.SUBMITTED}:
        raise InvalidReviewTransition(f"Sample is not submittable from {sample.review_status.value}")
    if not decisions:
        raise ValueError("At least one review decision is required")
    submitted = sample
    if sample.review_status == ReviewStatus.IN_REVIEW:
        submitted = transition_sample(sample, ReviewStatus.SUBMITTED, actor_id, "review submitted")
    if len(decisions) == 1:
        return submitted
    first = decisions[0].decision
    agreed = all(decision.decision == first and decision.corrected_annotation == decisions[0].corrected_annotation for decision in decisions)
    if agreed:
        return transition_sample(submitted, ReviewStatus.APPROVED, actor_id, "double review agreement")
    return transition_sample(submitted, ReviewStatus.NEEDS_ADJUDICATION, actor_id, "double review disagreement")


def approve_submitted(sample: ReviewSample, actor_id: str) -> ReviewSample:
    return transition_sample(sample, ReviewStatus.APPROVED, actor_id, "approved")


def decision_is_gold_eligible(decision: ReviewDecision) -> bool:
    return decision.decision in {ReviewDecisionType.ACCEPT, ReviewDecisionType.EDIT}

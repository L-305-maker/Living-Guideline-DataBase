"""Enums for guideline information records."""

from __future__ import annotations

from enum import Enum


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_REVIEW = "IN_REVIEW"
    SUBMITTED = "SUBMITTED"
    NEEDS_ADJUDICATION = "NEEDS_ADJUDICATION"
    ADJUDICATED = "ADJUDICATED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INVALID_SOURCE = "INVALID_SOURCE"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"


class ReviewDecisionType(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    EDIT = "EDIT"
    UNRESOLVED = "UNRESOLVED"


class RecommendedRoute(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ADJUDICATION = "ADJUDICATION"
    REJECT = "REJECT"


class SelectorName(str, Enum):
    RANDOM_AUDIT = "RandomAuditSelector"
    UNCERTAINTY = "UncertaintySelector"
    EXTRACTOR_VERIFIER_DISAGREEMENT = "ExtractorVerifierDisagreementSelector"
    RULE_MODEL_CONFLICT = "RuleModelConflictSelector"
    HARD_NEGATIVE = "HardNegativeSelector"
    MULTI_CLAIM = "MultiClaimSelector"
    FORMAT_STRATIFIED = "FormatStratifiedSelector"

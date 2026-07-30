"""Recommendation extraction evaluation metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.guideline_information.enums import AnnotationTier, IssueCode, RecommendedRoute, ReviewDecisionType, ReviewOrigin, SpanIssueCode, SpanSupportStatus
from src.guideline_information.models import ExtractionResult, GoldRecommendationExample, ReviewDecision, RouteResult, ValidationResult


def decision_counts(decisions: list[ReviewDecision]) -> dict[str, int]:
    counts = {item.value: 0 for item in ReviewDecisionType}
    for decision in decisions:
        counts[decision.decision.value] += 1
    return counts


def build_evaluation_report(
    *,
    candidates_count: int,
    extractions: list[ExtractionResult],
    validations: list[ValidationResult],
    routes: list[RouteResult],
    decisions: list[ReviewDecision],
    gold_examples: list[GoldRecommendationExample],
    evaluation_scope: str = "ALL_RUN_RECORDS",
    fake_runs_excluded: bool = False,
    synthetic_reviews_excluded: bool = False,
) -> dict[str, Any]:
    gold_by_candidate = {gold.candidate_id: gold for gold in gold_examples}
    extraction_by_candidate = {item.candidate_id: item for item in extractions}
    issue_counts = Counter(code for decision in decisions for code in decision.issue_codes)
    formal_gold = {candidate_id for candidate_id, gold in gold_by_candidate.items() if gold.final_annotation.get("is_formal_recommendation", True)}
    predicted_formal = {item.candidate_id for item in extractions if item.is_formal_recommendation}
    tp = len(predicted_formal & formal_gold)
    fp = len(predicted_formal - formal_gold) if gold_by_candidate else 0
    fn = len(formal_gold - predicted_formal)
    precision = _safe_div(tp, tp + fp) if gold_by_candidate else None
    recall = _safe_div(tp, tp + fn) if gold_by_candidate else None
    field_metrics = _field_metrics(gold_by_candidate, extraction_by_candidate)
    return {
        "evaluation_scope": evaluation_scope,
        "fake_runs_excluded": fake_runs_excluded,
        "synthetic_reviews_excluded": synthetic_reviews_excluded,
        "candidate_stage": {
            "candidate_total": candidates_count,
            "formal_recommendation_gold": len(formal_gold),
            "hard_negative_count": issue_counts.get(IssueCode.FALSE_POSITIVE_RECOMMENDATION.value, 0),
            "candidate_precision": precision,
            "candidate_recall": recall,
            "candidate_f1": _f1(precision, recall),
            "recall_note": "recall omitted unless pilot has complete human labels" if not gold_by_candidate else "computed from reviewed gold examples only",
        },
        "recommendation_classification": {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": None},
        },
        "field_extraction": field_metrics,
        "evidence_constraints": _evidence_constraint_metrics(validations),
        "review_efficiency": _review_efficiency_metrics(routes, decisions, gold_examples),
        "run_cost": _run_cost_metrics(extractions),
        "issue_code_distribution": dict(issue_counts),
    }


def _evidence_constraint_metrics(validations: list[ValidationResult]) -> dict[str, Any]:
    span_total = sum(len(statuses) for validation in validations for statuses in validation.field_statuses.values())
    issue_counts = Counter(
        issue.value if hasattr(issue, "value") else str(issue)
        for validation in validations
        for issues in validation.field_issue_codes.values()
        for issue in issues
    )
    invalid_spans = issue_counts.get(SpanIssueCode.SPAN_OUT_OF_RANGE.value, 0)
    unsupported = issue_counts.get(SpanIssueCode.UNSUPPORTED_FIELD.value, 0) + issue_counts.get(SpanIssueCode.QUOTE_NOT_FOUND.value, 0)
    return {
        "exact_span_rate": _safe_div(issue_counts.get(SpanIssueCode.EXACT_MATCH.value, 0), span_total),
        "normalized_match_rate": _safe_div(issue_counts.get(SpanIssueCode.NORMALIZATION_ONLY.value, 0), span_total),
        "partial_evidence_rate": _safe_div(issue_counts.get(SpanIssueCode.PARTIAL_EVIDENCE.value, 0), span_total),
        "evidence_span_valid_rate": _safe_div(span_total - invalid_spans, span_total),
        "unsupported_field_rate": _safe_div(unsupported, span_total),
        "invalid_span_rate": _safe_div(invalid_spans, span_total),
        "quote_not_found_rate": _safe_div(issue_counts.get(SpanIssueCode.QUOTE_NOT_FOUND.value, 0), span_total),
        "wrong_source_rate": _safe_div(
            issue_counts.get(SpanIssueCode.WRONG_SOURCE_BLOCK.value, 0)
            + issue_counts.get(SpanIssueCode.WRONG_CONTEXT_SOURCE.value, 0),
            span_total,
        ),
        "over_inferred_rate": _safe_div(issue_counts.get(SpanIssueCode.OVER_INFERRED.value, 0), span_total),
        "source_revision_mismatch_count": sum(
            validation.overall_status == SpanSupportStatus.SOURCE_REVISION_MISMATCH for validation in validations
        ),
        "span_issue_distribution": dict(issue_counts),
        "by_field": _span_issues_by_field(validations),
    }


def _review_efficiency_metrics(
    routes: list[RouteResult],
    decisions: list[ReviewDecision],
    gold_examples: list[GoldRecommendationExample],
) -> dict[str, Any]:
    route_counts = Counter(route.route.value for route in routes)
    return {
        "auto_accept_rate": _safe_div(route_counts.get(RecommendedRoute.AUTO_ACCEPT.value, 0), len(routes)),
        "human_review_rate": _safe_div(route_counts.get(RecommendedRoute.HUMAN_REVIEW.value, 0), len(routes)),
        "reject_rate": _safe_div(route_counts.get(RecommendedRoute.REJECT.value, 0), len(routes)),
        "random_audit_error_rate": None,
        "average_fields_corrected_per_sample": _safe_div(
            sum(len(decision.corrected_annotation) for decision in decisions),
            len(decisions),
        ),
        "reviewer_agreement": None,
        "adjudication_rate": _safe_div(
            sum(gold.annotation_tier == AnnotationTier.GOLD_ADJUDICATED for gold in gold_examples),
            len(gold_examples),
        ),
    }


def _run_cost_metrics(extractions: list[ExtractionResult]) -> dict[str, Any]:
    return {
        "model_calls": len(extractions),
        "input_tokens": sum(int(item.usage.get("input_tokens") or item.usage.get("prompt_tokens") or 0) for item in extractions),
        "output_tokens": sum(int(item.usage.get("output_tokens") or item.usage.get("completion_tokens") or 0) for item in extractions),
        "failed_calls": 0,
        "retried_calls": None,
        "average_latency_ms": _safe_div(sum(item.latency_ms for item in extractions), len(extractions)),
        "money_cost": None,
    }


def build_real_model_evaluation_report(
    *,
    candidates_count: int,
    extractions: list[ExtractionResult],
    validations: list[ValidationResult],
    routes: list[RouteResult],
    decisions: list[ReviewDecision],
    gold_examples: list[GoldRecommendationExample],
) -> dict[str, Any]:
    real_extraction_ids = {item.extraction_id for item in extractions if not _is_fake_extraction(item)}
    real_candidate_ids = {item.candidate_id for item in extractions if item.extraction_id in real_extraction_ids}
    real_extractions = [item for item in extractions if item.extraction_id in real_extraction_ids]
    real_validations = [item for item in validations if item.extraction_id in real_extraction_ids]
    real_routes = [item for item in routes if item.extraction_id in real_extraction_ids]
    human_decisions = [item for item in decisions if item.review_origin == ReviewOrigin.HUMAN]
    human_review_ids = {item.review_id for item in human_decisions}
    real_gold = [
        item
        for item in gold_examples
        if item.extraction_id in real_extraction_ids and set(item.provenance_review_ids).issubset(human_review_ids)
    ]
    return build_evaluation_report(
        candidates_count=len(real_candidate_ids) if real_candidate_ids else candidates_count,
        extractions=real_extractions,
        validations=real_validations,
        routes=real_routes,
        decisions=human_decisions,
        gold_examples=real_gold,
        evaluation_scope="REAL_MODEL_HUMAN_REVIEWED",
        fake_runs_excluded=True,
        synthetic_reviews_excluded=True,
    )


def _is_fake_extraction(extraction: ExtractionResult) -> bool:
    model_name = (extraction.model_name or extraction.model_version or "").lower()
    return model_name.startswith("fake") or "fake" in model_name


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Guideline Information Evaluation", ""]
    for section, values in report.items():
        lines.append(f"## {section}")
        if isinstance(values, dict):
            for key, value in values.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- value: {values}")
        lines.append("")
    return "\n".join(lines)


def _field_metrics(gold_by_candidate: dict[str, GoldRecommendationExample], extraction_by_candidate: dict[str, ExtractionResult]) -> dict[str, Any]:
    fields = ["direction", "strength", "certainty", "population", "interventions", "dosage", "duration"]
    metrics: dict[str, Any] = {}
    for field in fields:
        total = 0
        exact = 0
        for candidate_id, gold in gold_by_candidate.items():
            if candidate_id not in extraction_by_candidate:
                continue
            expected = gold.final_annotation.get(field)
            if expected in (None, ""):
                continue
            total += 1
            actual = getattr(extraction_by_candidate[candidate_id], field)
            exact += int(_normalize(actual) == _normalize(expected))
        metrics[f"{field}_exact_match"] = _safe_div(exact, total)
    return metrics


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(str(item).strip().lower() for item in value)
    return str(value).strip().lower()


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if not den else num / den


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)



def build_error_examples(*, validations: list[ValidationResult], extractions: list[ExtractionResult], max_excerpt: int = 240) -> list[dict[str, Any]]:
    extraction_by_id = {item.extraction_id: item for item in extractions}
    examples: list[dict[str, Any]] = []
    for validation in validations:
        extraction = extraction_by_id.get(validation.extraction_id)
        for field_name, issues in validation.field_issue_codes.items():
            for index, issue in enumerate(issues):
                issue_value = issue.value if hasattr(issue, "value") else str(issue)
                if issue_value == SpanIssueCode.EXACT_MATCH.value:
                    continue
                evidence = None
                if extraction and field_name in extraction.field_evidence and index < len(extraction.field_evidence[field_name]):
                    evidence = extraction.field_evidence[field_name][index]
                quote = evidence.quote if evidence else ""
                examples.append(
                    {
                        "candidate_id": validation.candidate_id,
                        "field_name": field_name,
                        "issue_code": issue_value,
                        "short_source_excerpt": quote[:max_excerpt],
                        "model_value": evidence.value_normalized if evidence else "",
                        "corrected_value": "",
                        "prompt_version": extraction.prompt_version if extraction else "",
                        "model_name": extraction.model_name if extraction else "",
                    }
                )
    return examples


def _span_issues_by_field(validations: list[ValidationResult]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter] = {}
    for validation in validations:
        for field, issues in validation.field_issue_codes.items():
            counter = result.setdefault(field, Counter())
            for issue in issues:
                counter[issue.value if hasattr(issue, "value") else str(issue)] += 1
    return {field: dict(counter) for field, counter in result.items()}

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from src.common.extraction_common import utc_now


JsonDict = Dict[str, Any]
APPROVE_DECISIONS = {"approve", "approved", "publish"}
REJECT_DECISIONS = {"reject", "rejected"}


def _payload(row: JsonDict) -> JsonDict:
    value = row.get("version_payload")
    return deepcopy(value) if isinstance(value, dict) else {}


def apply_review_decision(
    review_item: JsonDict,
    *,
    decision: str,
    reviewer: str,
    reason: str = "",
) -> JsonDict:
    """根据人工审核结论返回更新后的 review queue 行，不直接写数据库。"""

    normalized_decision = str(decision or "").lower()
    if normalized_decision not in APPROVE_DECISIONS | REJECT_DECISIONS:
        raise ValueError(f"Unsupported review decision: {decision}")
    reviewed = deepcopy(review_item)
    reviewed["review_status"] = "approved" if normalized_decision in APPROVE_DECISIONS else "rejected"
    reviewed["review_decision"] = "approved" if normalized_decision in APPROVE_DECISIONS else "rejected"
    reviewed["reviewer"] = reviewer
    reviewed["review_reason"] = reason
    reviewed["reviewed_at"] = utc_now()
    return reviewed


def approved_version_payload(review_item: JsonDict) -> JsonDict:
    """把已批准的 review item 中保存的版本 payload 转换为 publishable 版本行。"""

    if str(review_item.get("review_status") or "") != "approved":
        raise ValueError("Only approved review items can be promoted.")
    version = _payload(review_item)
    if not version:
        raise ValueError("Review item does not contain version_payload.")
    normalized = version.get("normalized_payload")
    if not isinstance(normalized, dict):
        normalized = {}
    gate = normalized.get("publish_gate")
    if not isinstance(gate, dict):
        gate = {}
    gate.update(
        {
            "quality_status": "publishable",
            "publishable": True,
            "blocking_reasons": [],
            "warning_reasons": [],
            "manual_override": {
                "review_item_id": review_item.get("review_item_id"),
                "reviewer": review_item.get("reviewer"),
                "review_reason": review_item.get("review_reason"),
                "reviewed_at": review_item.get("reviewed_at"),
            },
        }
    )
    normalized["publish_gate"] = gate
    version["normalized_payload"] = normalized
    version["quality_status"] = "publishable"
    return version


def reject_review_item(review_item: JsonDict, *, reviewer: str, reason: str = "") -> JsonDict:
    """拒绝发布审核项的便捷函数，保留审核人、原因和审核时间。"""

    return apply_review_decision(review_item, decision="rejected", reviewer=reviewer, reason=reason)


def approve_review_item(review_item: JsonDict, *, reviewer: str, reason: str = "") -> tuple[JsonDict, JsonDict]:
    """批准发布审核项，并同时返回审核行和可发布版本 payload。"""

    reviewed = apply_review_decision(review_item, decision="approved", reviewer=reviewer, reason=reason)
    return reviewed, approved_version_payload(reviewed)


def mark_review_item_published(
    review_item: JsonDict,
    *,
    published_version_id: str,
    update_log_ids: list[str] | None = None,
    published_at: str | None = None,
    summary: str = "",
) -> JsonDict:
    """发布完成后把 approved review item 推进为 published 状态。"""

    if str(review_item.get("review_status") or "") != "approved":
        raise ValueError("Only approved review items can be marked as published.")
    published = deepcopy(review_item)
    published["review_status"] = "published"
    published["published_version_id"] = published_version_id
    published["publication_update_log_ids"] = list(update_log_ids or [])
    published["publication_summary"] = summary
    published["published_at"] = published_at or utc_now()
    return published

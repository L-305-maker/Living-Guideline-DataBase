"""推荐版本构建文件：把推荐、GRADE、PICO 和证据候选组装成不可变 RecommendationVersion。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.common.extraction_common import utc_now
from src.domain.knowledge import Recommendation


JsonDict = Dict[str, Any]
STATE_BUILDER_VERSION = "recommendation_state_builder_v1"


def publish_gate(row: JsonDict) -> JsonDict:
    """读取版本 payload 中的发布门禁结果；缺失或格式错误时返回空对象。"""

    payload = row.get("normalized_payload")
    if not isinstance(payload, dict):
        return {}
    gate = payload.get("publish_gate")
    return gate if isinstance(gate, dict) else {}


def is_publishable_version(row: JsonDict) -> bool:
    """判断版本是否真正可发布，同时检查顶层状态和嵌套 publish_gate。"""

    gate = publish_gate(row)
    if gate.get("publishable") is not True:
        return False
    return str(row.get("quality_status") or gate.get("quality_status") or "") == "publishable"


def _compact_title(text: Any, limit: int = 140) -> str:
    """从推荐正文压缩出当前态标题，避免长推荐语句撑爆列表 UI。"""

    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _previous_value(previous: Optional[JsonDict], key: str) -> Any:
    """安全读取上一条 Recommendation 当前态字段，用于发布时延续历史状态。"""

    if not isinstance(previous, dict):
        return None
    return previous.get(key)


def recommendation_status_for_version(version: JsonDict) -> str:
    """根据版本变化类型推导当前推荐状态；目前撤回版本映射为 withdrawn，其余为 active。"""

    if str(version.get("change_type") or "") == "withdrawn":
        return "withdrawn"
    return "active"


def build_recommendation_state(
    version: JsonDict,
    *,
    previous_recommendation: Optional[JsonDict] = None,
    published_at: Optional[str] = None,
) -> JsonDict:
    """为已发布版本构建 Recommendation 当前态行。

    当前态只能由发布流程调用，不能由候选抽取流程直接创建。这样可以保证
    `recommendations.current_version_id` 永远指向已经通过 publish_gate 的正式版本。
    """

    if not is_publishable_version(version):
        version_id = str(version.get("recommendation_version_id") or "")
        raise ValueError(f"RecommendationVersion is not publishable: {version_id}")

    now = published_at or utc_now()
    recommendation_id = str(version.get("recommendation_id") or "")
    if not recommendation_id:
        raise ValueError("Published RecommendationVersion must include recommendation_id.")
    guideline_id = str(version.get("guideline_id") or "")
    if not guideline_id:
        raise ValueError("Published RecommendationVersion must include guideline_id.")

    status = recommendation_status_for_version(version)
    withdrawn_at = now if status == "withdrawn" else _previous_value(previous_recommendation, "withdrawn_at")
    normalized_payload = {
        "state_builder_version": STATE_BUILDER_VERSION,
        "current_version": {
            "recommendation_version_id": version.get("recommendation_version_id"),
            "version_number": version.get("version_number"),
            "change_type": version.get("change_type"),
            "change_summary": version.get("change_summary"),
            "quality_status": version.get("quality_status"),
            "published_at": now,
        },
        "publish_gate": publish_gate(version),
    }
    recommendation = Recommendation(
        recommendation_id=recommendation_id,
        guideline_id=guideline_id,
        current_version_id=str(version.get("recommendation_version_id") or ""),
        status=status,  # type: ignore[arg-type]
        recommendation_code=version.get("recommendation_code"),
        title=version.get("recommendation_code") or _compact_title(version.get("recommendation_text")),
        population=version.get("population"),
        intervention=version.get("intervention"),
        comparator=version.get("comparator"),
        direction=version.get("direction") or "unclear",
        strength=version.get("strength") or "unclear",
        certainty=version.get("certainty") or "unclear",
        first_created_at=_previous_value(previous_recommendation, "first_created_at") or version.get("created_at") or now,
        last_published_at=now,
        last_updated_at=now,
        withdrawn_at=withdrawn_at,
        withdrawn_reason=version.get("change_summary") if status == "withdrawn" else _previous_value(previous_recommendation, "withdrawn_reason"),
        superseded_by_recommendation_id=_previous_value(previous_recommendation, "superseded_by_recommendation_id"),
        normalized_payload=normalized_payload,
    )
    return recommendation.to_dict()


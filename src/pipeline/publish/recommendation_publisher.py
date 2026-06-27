"""发布阶段文件：把审核通过、闭包完整的候选版本导出为 publish-ready 正式发布包。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.common.extraction_common import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.pipeline.extraction.versioning.recommendation_state_builder import build_recommendation_state, is_publishable_version
from src.pipeline.update.version_diff import build_update_logs
from src.storage.generic_ingest import ingest_tables_atomically
from src.storage.repositories.recommendation_version_reviews import approved_version_payload, mark_review_item_published


JsonDict = Dict[str, Any]
PUBLISHER_VERSION = "recommendation_publisher_v1"


@dataclass(frozen=True)
class PublishContext:
    reviewer: Optional[str] = None
    reason: str = ""
    published_by: str = "recommendation_publisher"
    published_at: str = ""

    def timestamp(self) -> str:
        """返回本次发布使用的时间戳；测试可传固定时间，生产默认使用当前 UTC 时间。"""

        return self.published_at or utc_now()


@dataclass
class PublishBundle:
    recommendation_versions: List[JsonDict] = field(default_factory=list)
    recommendations: List[JsonDict] = field(default_factory=list)
    update_logs: List[JsonDict] = field(default_factory=list)
    recommendation_version_review_queue: List[JsonDict] = field(default_factory=list)
    summary: JsonDict = field(default_factory=dict)

    def table_rows(self) -> Dict[str, List[JsonDict]]:
        """把发布 bundle 转为 storage 层可直接原子入库的表名到行列表映射。"""

        rows: Dict[str, List[JsonDict]] = {
            "recommendation_versions": self.recommendation_versions,
            "recommendations": self.recommendations,
            "update_logs": self.update_logs,
            "recommendation_version_review_queue": self.recommendation_version_review_queue,
        }
        return {table: table_rows for table, table_rows in rows.items() if table_rows}


def _version_number(row: JsonDict) -> int:
    raw = str(row.get("version_number") or "v0").lstrip("vV")
    try:
        return int(raw or 0)
    except ValueError:
        return 0


def latest_previous_version(version: JsonDict, previous_versions: Iterable[JsonDict]) -> Optional[JsonDict]:
    recommendation_id = str(version.get("recommendation_id") or "")
    version_id = str(version.get("recommendation_version_id") or "")
    latest: Optional[JsonDict] = None
    for previous in previous_versions:
        if str(previous.get("recommendation_id") or "") != recommendation_id:
            continue
        if str(previous.get("recommendation_version_id") or "") == version_id:
            continue
        if latest is None or _version_number(previous) > _version_number(latest):
            latest = previous
    return latest


def _with_publication_fields(version: JsonDict, published_at: str) -> JsonDict:
    published = deepcopy(version)
    published["published_at"] = published.get("published_at") or published_at
    normalized = published.get("normalized_payload")
    if not isinstance(normalized, dict):
        normalized = {}
    normalized["publisher"] = {
        "publisher_version": PUBLISHER_VERSION,
        "published_at": published["published_at"],
    }
    published["normalized_payload"] = normalized
    return published


def publish_version(
    version: JsonDict,
    *,
    previous_versions: Iterable[JsonDict] = (),
    previous_recommendation: Optional[JsonDict] = None,
    context: PublishContext = PublishContext(),
) -> PublishBundle:
    """为一条已通过发布门禁的 RecommendationVersion 构建正式发布 bundle。

    这个函数只做发布编排，不直接写数据库。它会同时生成：
    1. 带 published_at 和 publisher 元数据的正式版本行；
    2. 指向该版本的 Recommendation 当前态；
    3. 描述新旧版本差异的 UpdateLog；
    4. 可供调用方审计的 summary。
    """

    if not is_publishable_version(version):
        version_id = str(version.get("recommendation_version_id") or "")
        raise ValueError(f"Cannot publish non-publishable RecommendationVersion: {version_id}")

    published_at = context.timestamp()
    published_version = _with_publication_fields(version, published_at)
    recommendation = build_recommendation_state(
        published_version,
        previous_recommendation=previous_recommendation,
        published_at=published_at,
    )
    previous_version = latest_previous_version(published_version, previous_versions)
    update_logs, update_report = build_update_logs(
        [previous_version] if previous_version else [],
        [published_version],
        updated_by=context.published_by,
    )
    for log in update_logs:
        log["published_at"] = published_at
        log["change_reason"] = context.reason or log.get("change_reason")

    summary = {
        "publisher_version": PUBLISHER_VERSION,
        "published_versions": 1,
        "recommendation_id": published_version.get("recommendation_id"),
        "recommendation_version_id": published_version.get("recommendation_version_id"),
        "previous_recommendation_version_id": (previous_version or {}).get("recommendation_version_id"),
        "updated_current_state": 1,
        "generated_update_logs": len(update_logs),
        "update_report": update_report,
        "published_by": context.published_by,
        "reviewer": context.reviewer,
        "published_at": published_at,
    }
    return PublishBundle(
        recommendation_versions=[published_version],
        recommendations=[recommendation],
        update_logs=update_logs,
        summary=summary,
    )


def publish_approved_review_item(
    review_item: JsonDict,
    *,
    previous_versions: Iterable[JsonDict] = (),
    previous_recommendation: Optional[JsonDict] = None,
    context: PublishContext = PublishContext(),
) -> PublishBundle:
    """把已批准的发布审核项提升为正式发布 bundle。

    输入必须是 `review_status == approved` 的 review item。函数会复用 review item
    中保存的 `version_payload`，先通过人工批准信息改写 publish_gate，再调用
    `publish_version()` 生成正式发布产物，最后把 review item 状态推进为 published。
    """

    if str(review_item.get("review_status") or "") != "approved":
        raise ValueError("Only approved review items can be published.")

    version = approved_version_payload(review_item)
    bundle = publish_version(
        version,
        previous_versions=previous_versions,
        previous_recommendation=previous_recommendation,
        context=context,
    )
    update_log_ids = [str(log.get("update_log_id") or "") for log in bundle.update_logs if log.get("update_log_id")]
    published_review = mark_review_item_published(
        review_item,
        published_version_id=str(version.get("recommendation_version_id") or ""),
        update_log_ids=update_log_ids,
        published_at=bundle.summary["published_at"],
        summary=f"Published {version.get('recommendation_version_id')}",
    )
    bundle.recommendation_version_review_queue.append(published_review)
    bundle.summary["published_review_items"] = 1
    bundle.summary["review_item_id"] = review_item.get("review_item_id")
    return bundle


def write_bundle(bundle: PublishBundle, output_dir: str | Path) -> JsonDict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "recommendation_versions": root / "recommendation_versions.publish.jsonl",
        "recommendations": root / "recommendations.publish.jsonl",
        "update_logs": root / "update_logs.publish.jsonl",
        "recommendation_version_review_queue": root / "recommendation_version_review_queue.publish.jsonl",
        "summary": root / "publish_summary.jsonl",
    }
    write_jsonl(outputs["recommendation_versions"], bundle.recommendation_versions)
    write_jsonl(outputs["recommendations"], bundle.recommendations)
    write_jsonl(outputs["update_logs"], bundle.update_logs)
    write_jsonl(outputs["recommendation_version_review_queue"], bundle.recommendation_version_review_queue)
    write_jsonl(outputs["summary"], [bundle.summary])
    return {key: str(path) for key, path in outputs.items()}


def ingest_publish_bundle(bundle: PublishBundle, *, conn: Any | None = None) -> JsonDict:
    """通过现有原子入库路径持久化发布 bundle。

    这里复用 `ingest_tables_atomically()`，确保版本表、当前态表、更新日志表和
    review queue 状态在同一个事务中提交；任意一步失败都会整体回滚。
    """

    counts = ingest_tables_atomically(bundle.table_rows(), conn=conn)
    return {
        "publisher_version": PUBLISHER_VERSION,
        "status": "ingested",
        "table_counts": counts,
        "published_versions": len(bundle.recommendation_versions),
        "recommendation_updates": len(bundle.recommendations),
        "update_logs": len(bundle.update_logs),
        "review_rows": len(bundle.recommendation_version_review_queue),
        "ingested_at": utc_now(),
    }


def _first_row(path: str | Path) -> JsonDict:
    for row in iter_jsonl(path):
        return row
    raise ValueError(f"No JSONL rows found in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发布已通过审核的 Living-Guideline 推荐版本。")
    parser.add_argument("--review-input", required=True, help="包含 approved review item 的 JSONL 文件。")
    parser.add_argument("--previous-versions-input", help="已有正式 RecommendationVersion JSONL，用于生成版本差异日志。")
    parser.add_argument("--previous-recommendation-input", help="已有 Recommendation 当前态 JSONL，用于保留首次创建时间等状态。")
    parser.add_argument("--output-dir", required=True, help="发布 bundle JSONL 输出目录。")
    parser.add_argument("--published-by", default="recommendation_publisher")
    parser.add_argument("--reason", default="")
    parser.add_argument("--ingest", action="store_true", help="输出发布 bundle 后，使用一个 storage 原子事务直接入库。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous_versions = list(iter_jsonl(args.previous_versions_input)) if args.previous_versions_input else []
    previous_recommendation = _first_row(args.previous_recommendation_input) if args.previous_recommendation_input else None
    bundle = publish_approved_review_item(
        _first_row(args.review_input),
        previous_versions=previous_versions,
        previous_recommendation=previous_recommendation,
        context=PublishContext(published_by=args.published_by, reason=args.reason),
    )
    outputs = write_bundle(bundle, args.output_dir)
    ingest_summary = ingest_publish_bundle(bundle) if args.ingest else {}
    if ingest_summary:
        write_jsonl(Path(args.output_dir) / "publish_ingest_summary.jsonl", [ingest_summary])
    print(
        "published_versions={published_versions} recommendations={recommendations} update_logs={update_logs} review_rows={review_rows} ingested={ingested} summary={summary}".format(
            published_versions=len(bundle.recommendation_versions),
            recommendations=len(bundle.recommendations),
            update_logs=len(bundle.update_logs),
            review_rows=len(bundle.recommendation_version_review_queue),
            ingested=bool(ingest_summary),
            summary=outputs["summary"],
        )
    )


if __name__ == "__main__":
    main()


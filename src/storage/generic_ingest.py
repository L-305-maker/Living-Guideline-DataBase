"""通用 JSONL 入库模块。

本模块把流水线各阶段产出的 JSONL 行映射到 PostgreSQL 表字段，
负责默认值填充、JSON/list 字段序列化、批量 upsert，以及发布门禁后的
正式版本表/发布审核队列拆分。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.storage.connection import get_connection
from src.storage.schema import SCHEMA_TABLES, create_schema
from src.storage.utils import DEFAULT_BATCH_SIZE, batches, execute_many, iter_jsonl, json_list, json_obj, sql_identifier


JsonDict = Dict[str, Any]
connection = Any

GENERIC_TABLE_CONFIG: Dict[str, Dict[str, Any]] = {
    "model_traces": {
        "pk": "model_trace_id",
        "json_fields": {"raw_output", "parsed_output", "parameters", "token_usage"},
        "list_fields": set(),
        "defaults": {"success": True, "human_verified": False, "parameters": {}, "token_usage": {}},
    },
    "recommendation_candidates": {
        "pk": "candidate_id",
        "json_fields": {"normalized_payload", "raw_payload"},
        "list_fields": set(),
        "aliases": {
            "statement": ("recommendation_text",),
            "raw_text": ("source_text",),
            "source_record_id": ("record_id",),
            "confidence": ("extraction_confidence",),
            "outcome": ("outcome_summary",),
        },
        "defaults": {
            "direction": "unclear",
            "strength": "unclear",
            "extraction_method": "rule",
            "status": "pending",
            "normalized_payload": {},
            "raw_payload": {},
        },
    },
    "grade_candidates": {
        "pk": "grade_candidate_id",
        "json_fields": {"normalized_payload", "raw_payload"},
        "list_fields": set(),
        "aliases": {
            "source_grade_raw": ("source_text", "source_span"),
            "chunk_id": ("source_block_id",),
            "confidence": ("extraction_confidence",),
        },
        "defaults": {
            "grade_system": "unknown",
            "certainty": "unclear",
            "strength": "unclear",
            "direction": "unclear",
            "risk_of_bias": "not_reported",
            "inconsistency": "not_reported",
            "indirectness": "not_reported",
            "imprecision": "not_reported",
            "publication_bias": "not_reported",
            "extraction_method": "rule",
            "status": "pending",
            "normalized_payload": {},
            "raw_payload": {},
        },
    },
    "pico_questions": {
        "pk": "pico_id",
        "json_fields": {"outcomes", "normalized_payload", "raw_payload"},
        "list_fields": {"outcomes"},
        "defaults": {"outcomes": [], "status": "active", "normalized_payload": {}, "raw_payload": {}},
    },
    "recommendation_versions": {
        "pk": "recommendation_version_id",
        "json_fields": {"normalized_payload", "raw_payload"},
        "list_fields": set(),
        "defaults": {
            "direction": "unclear",
            "strength": "unclear",
            "certainty": "unclear",
            "change_type": "new",
            "quality_status": "needs_review",
            "normalized_payload": {},
            "raw_payload": {},
        },
    },
    "recommendations": {
        "pk": "recommendation_id",
        "json_fields": {"normalized_payload"},
        "list_fields": set(),
        "defaults": {
            "status": "active",
            "direction": "unclear",
            "strength": "unclear",
            "certainty": "unclear",
            "normalized_payload": {},
        },
    },
    "recommendation_version_review_queue": {
        "pk": "review_item_id",
        "json_fields": {"blocking_reasons", "warning_reasons", "publish_gate", "version_payload", "publication_update_log_ids"},
        "list_fields": {"blocking_reasons", "warning_reasons", "publication_update_log_ids"},
        "defaults": {
            "review_status": "pending",
            "priority": "normal",
            "blocking_reasons": [],
            "warning_reasons": [],
            "publish_gate": {},
            "version_payload": {},
            "publication_update_log_ids": [],
        },
    },
    "evidence_items": {
        "pk": "evidence_id",
        "json_fields": {"outcomes_extracted", "effect_size", "normalized_payload", "raw_payload"},
        "list_fields": {"outcomes_extracted"},
        "defaults": {
            "outcomes_extracted": [],
            "effect_size": {},
            "effect_direction": "uncertain",
            "extraction_method": "manual",
            "screening_status": "uncertain",
            "normalized_payload": {},
            "raw_payload": {},
        },
    },
    "recommendation_version_evidence_links": {
        "pk": "link_id",
        "json_fields": {"normalized_payload"},
        "list_fields": set(),
        "defaults": {
            "link_reason": "release_linked_evidence",
            "link_status": "publish_ready",
            "normalized_payload": {},
        },
    },
    "update_logs": {
        "pk": "update_log_id",
        "json_fields": {"triggering_evidence_ids"},
        "list_fields": {"triggering_evidence_ids"},
        "defaults": {"update_type": "evidence_updated", "change_summary": "", "triggering_evidence_ids": []},
    },
}


def publish_gate_payload(row: JsonDict) -> JsonDict:
    payload = row.get("normalized_payload")
    if not isinstance(payload, dict):
        return {}
    gate = payload.get("publish_gate")
    return gate if isinstance(gate, dict) else {}


def recommendation_version_publish_status(row: JsonDict) -> str:
    gate = publish_gate_payload(row)
    return str(row.get("quality_status") or gate.get("quality_status") or "needs_review")


def is_publishable_recommendation_version(row: JsonDict) -> bool:
    gate = publish_gate_payload(row)
    if gate and gate.get("publishable") is not True:
        return False
    return recommendation_version_publish_status(row) == "publishable"


def partition_recommendation_versions_for_ingest(rows: Sequence[JsonDict]) -> tuple[List[JsonDict], JsonDict]:
    """按发布门禁拆分版本行，确保只有 publishable 版本进入正式版本表。"""

    publishable: List[JsonDict] = []
    skipped_statuses: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    for row in rows:
        if is_publishable_recommendation_version(row):
            publishable.append(row)
            continue
        status = recommendation_version_publish_status(row)
        skipped_statuses[status] += 1
        gate = publish_gate_payload(row)
        reasons = gate.get("blocking_reasons") or gate.get("warning_reasons") or ["publish_gate_not_satisfied"]
        for reason in reasons if isinstance(reasons, list) else ["publish_gate_not_satisfied"]:
            skipped_reasons[str(reason)] += 1
    return publishable, {
        "input_rows": len(rows),
        "publishable_rows": len(publishable),
        "skipped_rows": len(rows) - len(publishable),
        "skipped_status_counts": dict(skipped_statuses),
        "skipped_reason_counts": dict(skipped_reasons),
    }


def review_priority_for(status: str, gate: JsonDict) -> str:
    if status == "blocked":
        return "high"
    warnings = gate.get("warning_reasons")
    if isinstance(warnings, list) and len(warnings) >= 2:
        return "high"
    return "normal"


def recommendation_version_review_row(row: JsonDict) -> JsonDict:
    """把非 publishable 版本转换成可持久化的发布审核队列行。"""

    gate = publish_gate_payload(row)
    status = recommendation_version_publish_status(row)
    version_id = str(row.get("recommendation_version_id") or "")
    blocking = gate.get("blocking_reasons") if isinstance(gate.get("blocking_reasons"), list) else []
    warnings = gate.get("warning_reasons") if isinstance(gate.get("warning_reasons"), list) else []
    return {
        "review_item_id": f"review_{version_id}" if version_id else "",
        "recommendation_version_id": version_id,
        "recommendation_candidate_id": row.get("recommendation_candidate_id"),
        "guideline_id": row.get("guideline_id"),
        "record_id": row.get("record_id"),
        "quality_status": status,
        "review_status": "pending",
        "priority": review_priority_for(status, gate),
        "blocking_reasons": blocking,
        "warning_reasons": warnings,
        "publish_gate": gate,
        "version_payload": row,
    }


def recommendation_version_review_rows(rows: Sequence[JsonDict]) -> List[JsonDict]:
    return [
        recommendation_version_review_row(row)
        for row in rows
        if row.get("recommendation_version_id") and not is_publishable_recommendation_version(row)
    ]


def table_columns(conn: connection, table: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [row[0] for row in cur.fetchall()]


def aliased_value(item: JsonDict, column: str, aliases: Dict[str, Sequence[str]]) -> Any:
    """按 storage 字段别名读取值，兼容新 JSONL 字段和旧 PostgreSQL 列名。"""

    value = item.get(column)
    if value is not None:
        return value
    for alias in aliases.get(column, ()):
        value = item.get(alias)
        if value is not None:
            return value
    return None


def prepare_generic_row(
    item: JsonDict,
    columns: Sequence[str],
    json_fields: set[str],
    list_fields: set[str],
    aliases: Dict[str, Sequence[str]] | None = None,
) -> JsonDict:
    defaults = item.get("__defaults__") or {}
    aliases = aliases or {}
    row: JsonDict = {}
    for column in columns:
        if column == "created_at" or column == "updated_at" or column == "first_seen_at":
            row[column] = item.get(column) or ""
            continue
        value = aliased_value(item, column, aliases)
        if value is None and column in defaults:
            value = defaults[column]
        if column in json_fields:
            row[column] = json_list(value) if column in list_fields else json_obj(value)
        else:
            row[column] = value
    return row


def ingest_table(table: str, jsonl_path: str | Path, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """按表配置把一个 JSONL 文件写入 PostgreSQL。"""

    table = sql_identifier(table)
    if table not in GENERIC_TABLE_CONFIG:
        raise ValueError(f"Unsupported generic ingest table: {table}")
    create_schema(recreate=False)
    config = GENERIC_TABLE_CONFIG[table]
    total = 0
    with get_connection() as conn:
        columns = [col for col in table_columns(conn, table) if col not in {"created_at", "updated_at", "first_seen_at"}]
        pk = config["pk"]
        insert_cols = ", ".join(columns)
        values_cols = ", ".join(
            f"%({col})s::jsonb" if col in config["json_fields"] else f"%({col})s" for col in columns
        )
        update_cols = [col for col in columns if col != pk]
        updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
        sql = f"""
            INSERT INTO {table} ({insert_cols})
            VALUES ({values_cols})
            ON CONFLICT ({pk}) DO UPDATE SET {updates}
        """
        for batch in batches(iter_jsonl(jsonl_path), batch_size):
            review_rows: List[JsonDict] = []
            if table == "recommendation_versions":
                # recommendation_versions 是正式知识表，入库前必须按 publish_gate 分流。
                # 不可发布的版本转入 recommendation_version_review_queue，而不是写进正式版本表。
                review_rows = recommendation_version_review_rows(batch)
                batch, gate_report = partition_recommendation_versions_for_ingest(batch)
                if gate_report["skipped_rows"]:
                    print(
                        "Skipped {skipped_rows} recommendation_versions by publish gate: {skipped_status_counts}".format(
                            **gate_report
                        )
                    )
            rows = [
                prepare_generic_row(
                    {**item, "__defaults__": config.get("defaults", {})},
                    columns,
                    config["json_fields"],
                    config["list_fields"],
                    config.get("aliases"),
                )
                for item in batch
                if item.get(pk)
            ]
            try:
                execute_many(conn, sql, rows)
                if review_rows:
                    _upsert_rows(conn, "recommendation_version_review_queue", review_rows)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise RuntimeError(f"Failed to ingest {table} after {total} rows: {exc}") from exc
            total += len(rows)
            print(f"Ingested {total} rows into {table}")
    return total


def _upsert_rows(conn: connection, table: str, items: Sequence[JsonDict]) -> int:
    table = sql_identifier(table)
    if table not in GENERIC_TABLE_CONFIG:
        raise ValueError(f"Unsupported generic ingest table: {table}")
    config = GENERIC_TABLE_CONFIG[table]
    columns = [col for col in table_columns(conn, table) if col not in {"created_at", "updated_at", "first_seen_at"}]
    pk = config["pk"]
    insert_cols = ", ".join(columns)
    values_cols = ", ".join(f"%({col})s::jsonb" if col in config["json_fields"] else f"%({col})s" for col in columns)
    updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in columns if col != pk)
    sql = f"""
        INSERT INTO {table} ({insert_cols})
        VALUES ({values_cols})
        ON CONFLICT ({pk}) DO UPDATE SET {updates}
    """
    rows = [
        prepare_generic_row(
            {**item, "__defaults__": config.get("defaults", {})},
            columns,
            config["json_fields"],
            config["list_fields"],
            config.get("aliases"),
        )
        for item in items
        if item.get(pk)
    ]
    return execute_many(conn, sql, rows)


def ingest_tables_atomically(
    table_rows: Dict[str, Sequence[JsonDict]],
    *,
    conn: connection | None = None,
) -> Dict[str, int]:
    """在单个事务中写入一个 Living-Guideline 业务 bundle。

    该函数是发布闭环和批量入库的共同安全边界：版本、当前态、审核队列和
    更新日志必须一起成功；任何一步失败都回滚，避免正式知识库出现半发布状态。
    """
    owned_connection = conn is None
    if owned_connection:
        create_schema(recreate=False)
    active_conn = conn or get_connection()
    pending_review_rows = list(table_rows.get("recommendation_version_review_queue") or ())
    counts: Dict[str, int] = {}
    # 固定顺序体现外键依赖，也体现发布语义：先写依赖实体，再写版本、当前态、证据链接和更新日志。
    order = [
        "model_traces",
        "pico_questions",
        "recommendation_candidates",
        "grade_candidates",
        "recommendation_versions",
        "recommendations",
        "recommendation_version_review_queue",
        "evidence_items",
        "recommendation_version_evidence_links",
        "update_logs",
    ]
    try:
        for table in order:
            rows = table_rows.get(table) or ()
            if table == "recommendation_versions":
                # 原子 bundle 入库同样要执行 publish gate 分流，保证正式表只包含 publishable 版本。
                pending_review_rows.extend(recommendation_version_review_rows(rows))
                rows, gate_report = partition_recommendation_versions_for_ingest(rows)
                if gate_report["skipped_rows"]:
                    counts["recommendation_versions_skipped_by_publish_gate"] = gate_report["skipped_rows"]
            if table == "recommendation_version_review_queue":
                rows = pending_review_rows
            if rows:
                counts[table] = _upsert_rows(active_conn, table, rows)
        active_conn.commit()
        return counts
    except Exception as exc:
        active_conn.rollback()
        raise RuntimeError(f"Atomic Living-Guideline ingest failed; transaction rolled back: {exc}") from exc
    finally:
        if owned_connection:
            active_conn.close()


def table_counts() -> JsonDict:
    create_schema(recreate=False)
    counts: JsonDict = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in SCHEMA_TABLES:
                cur.execute(f"SELECT COUNT(*) FROM {sql_identifier(table)}")
                counts[table] = cur.fetchone()[0]
    return counts


def integrity_report() -> JsonDict:
    """生成数据库完整性报告，用于发布后确认没有孤儿外键或低质量正式版本。"""

    create_schema(recreate=False)
    report: JsonDict = {"table_counts": {}, "orphan_counts": {}, "version_quality": {}, "version_review_queue": {}}
    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in SCHEMA_TABLES:
                cur.execute(f"SELECT COUNT(*) FROM {sql_identifier(table)}")
                report["table_counts"][table] = cur.fetchone()[0]
            orphan_queries = {
                "recommendation_candidates_without_guideline": """
                    SELECT COUNT(*)
                    FROM recommendation_candidates rc
                    LEFT JOIN guidelines g ON g.guideline_id = rc.guideline_id
                    WHERE rc.guideline_id IS NOT NULL AND rc.guideline_id <> '' AND g.guideline_id IS NULL
                """,
                "grade_candidates_without_recommendation": """
                    SELECT COUNT(*)
                    FROM grade_candidates gc
                    LEFT JOIN recommendation_candidates rc ON rc.candidate_id = gc.recommendation_candidate_id
                    WHERE gc.recommendation_candidate_id IS NOT NULL
                      AND gc.recommendation_candidate_id <> ''
                      AND rc.candidate_id IS NULL
                """,
                "evidence_items_without_pico": """
                    SELECT COUNT(*)
                    FROM evidence_items ei
                    LEFT JOIN pico_questions pq ON pq.pico_id = ei.pico_id
                    WHERE ei.pico_id IS NOT NULL AND ei.pico_id <> '' AND pq.pico_id IS NULL
                """,
                "recommendation_version_evidence_links_without_version": """
                    SELECT COUNT(*)
                    FROM recommendation_version_evidence_links link
                    LEFT JOIN recommendation_versions rv
                      ON rv.recommendation_version_id = link.recommendation_version_id
                    WHERE rv.recommendation_version_id IS NULL
                """,
                "recommendation_version_evidence_links_without_evidence": """
                    SELECT COUNT(*)
                    FROM recommendation_version_evidence_links link
                    LEFT JOIN evidence_items ei ON ei.evidence_id = link.evidence_id
                    WHERE ei.evidence_id IS NULL
                """,
            }
            for name, query in orphan_queries.items():
                cur.execute(query)
                report["orphan_counts"][name] = cur.fetchone()[0]
            version_queries = {
                "versions_without_grade": "SELECT COUNT(*) FROM recommendation_versions WHERE grade_candidate_id IS NULL OR grade_candidate_id = ''",
                "versions_without_pico": "SELECT COUNT(*) FROM recommendation_versions WHERE pico_id IS NULL OR pico_id = ''",
                "versions_without_source_text": "SELECT COUNT(*) FROM recommendation_versions WHERE source_text IS NULL OR source_text = ''",
                "versions_without_source_span": "SELECT COUNT(*) FROM recommendation_versions WHERE source_span IS NULL OR source_span = ''",
            }
            for name, query in version_queries.items():
                cur.execute(query)
                report["version_quality"][name] = cur.fetchone()[0]
            cur.execute(
                """
                SELECT review_status, COUNT(*)
                FROM recommendation_version_review_queue
                GROUP BY review_status
                """
            )
            report["version_review_queue"]["review_status_counts"] = {row[0]: row[1] for row in cur.fetchall()}
            cur.execute(
                """
                SELECT priority, COUNT(*)
                FROM recommendation_version_review_queue
                GROUP BY priority
                """
            )
            report["version_review_queue"]["priority_counts"] = {row[0]: row[1] for row in cur.fetchall()}
    return report

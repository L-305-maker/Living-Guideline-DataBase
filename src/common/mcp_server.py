"""Read-only MCP tools for the Living-Guideline PostgreSQL knowledge base."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.storage.connection import DATABASE_URL, get_connection, mask_database_url
from src.storage.schema import SCHEMA_TABLES
from src.storage.utils import sql_identifier


MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
DEFAULT_LIMIT = 10
MAX_LIMIT = 50

mcp = FastMCP("living-guideline-knowledge-base", host=MCP_HOST, port=MCP_PORT, json_response=True)


def clamp_limit(value: int | None, *, default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> int:
    if value is None:
        return default
    return max(1, min(int(value), maximum))


def truncate_text(value: Any, max_chars: int = 1000) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def row_dict(columns: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
    return {column: row[index] for index, column in enumerate(columns)}


def fetch_all(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a parameterized read-only query and return JSON-friendly rows."""

    with get_connection() as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15s'")
            cur.execute(sql, params)
            columns = [getattr(desc, "name", desc[0]) for desc in cur.description]
            return [row_dict(columns, row) for row in cur.fetchall()]


def fetch_one(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


@mcp.tool()
def health_check() -> dict[str, Any]:
    """Check whether the MCP server can reach the configured PostgreSQL database."""

    row = fetch_one(
        """
        SELECT
            current_database() AS database,
            current_user AS user,
            current_schema() AS schema,
            version() AS postgres_version
        """
    )
    return {
        "ok": row is not None,
        "database_url": mask_database_url(DATABASE_URL),
        "connection": row,
        "transport": MCP_TRANSPORT,
        "host": MCP_HOST,
        "port": MCP_PORT,
    }


@mcp.tool()
def table_counts() -> dict[str, Any]:
    """Return row counts for all project-managed tables without creating or modifying schema."""

    result: dict[str, Any] = {"database_url": mask_database_url(DATABASE_URL), "table_counts": {}, "missing_tables": []}
    with get_connection() as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15s'")
            for table in SCHEMA_TABLES:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = current_schema() AND table_name = %s
                    )
                    """,
                    (table,),
                )
                if not cur.fetchone()[0]:
                    result["missing_tables"].append(table)
                    continue
                cur.execute(f"SELECT COUNT(*) FROM {sql_identifier(table)}")
                result["table_counts"][table] = cur.fetchone()[0]
    return result


@mcp.tool()
def database_overview() -> dict[str, Any]:
    """Return high-level quality, source, and evidence status summaries."""

    return {
        "recommendation_quality_status": fetch_all(
            """
            SELECT quality_status, COUNT(*) AS count
            FROM recommendation_versions
            GROUP BY quality_status
            ORDER BY count DESC, quality_status
            """
        ),
        "evidence_screening_status": fetch_all(
            """
            SELECT screening_status, COUNT(*) AS count
            FROM evidence_items
            GROUP BY screening_status
            ORDER BY count DESC, screening_status
            """
        ),
        "top_sources": source_stats(limit=20)["sources"],
    }


@mcp.tool()
def source_stats(limit: int = 20) -> dict[str, Any]:
    """Return guideline counts grouped by source."""

    safe_limit = clamp_limit(limit, default=20)
    rows = fetch_all(
        """
        SELECT
            COALESCE(source, '<null>') AS source,
            COUNT(*) AS guideline_count,
            COUNT(*) FILTER (WHERE status = 'active') AS active_guideline_count
        FROM guidelines
        GROUP BY source
        ORDER BY guideline_count DESC, source
        LIMIT %s
        """,
        (safe_limit,),
    )
    return {"sources": rows, "limit": safe_limit}


@mcp.tool()
def search_recommendations(query: str, limit: int = 10) -> dict[str, Any]:
    """Search publishable recommendation versions by keyword across text, title, and PICO-like fields."""

    safe_limit = clamp_limit(limit)
    term = query.strip()
    if not term:
        return {"query": query, "results": [], "error": "query must not be empty"}
    pattern = f"%{term}%"
    rows = fetch_all(
        """
        SELECT
            rv.recommendation_version_id,
            rv.recommendation_id,
            rv.quality_status,
            rv.direction,
            rv.strength,
            rv.certainty,
            rv.population,
            rv.intervention,
            rv.comparator,
            rv.outcome_summary,
            rv.source_section,
            rv.source_url,
            g.guideline_id,
            g.title AS guideline_title,
            g.source,
            g.issuer,
            LEFT(rv.recommendation_text, 1200) AS recommendation_text,
            (
                SELECT COUNT(*)
                FROM recommendation_version_evidence_links link
                WHERE link.recommendation_version_id = rv.recommendation_version_id
            ) AS evidence_count
        FROM recommendation_versions rv
        JOIN guidelines g ON g.guideline_id = rv.guideline_id
        WHERE
            rv.recommendation_text ILIKE %s
            OR g.title ILIKE %s
            OR COALESCE(rv.population, '') ILIKE %s
            OR COALESCE(rv.intervention, '') ILIKE %s
            OR COALESCE(rv.comparator, '') ILIKE %s
            OR COALESCE(rv.outcome_summary, '') ILIKE %s
            OR COALESCE(rv.source_section, '') ILIKE %s
        ORDER BY
            CASE WHEN rv.recommendation_text ILIKE %s THEN 0 ELSE 1 END,
            evidence_count DESC,
            rv.recommendation_version_id
        LIMIT %s
        """,
        (pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern, safe_limit),
    )
    return {"query": term, "limit": safe_limit, "results": rows}


@mcp.tool()
def get_recommendation(recommendation_version_id: str) -> dict[str, Any]:
    """Return one recommendation version with guideline, PICO, GRADE, and evidence-count context."""

    version_id = recommendation_version_id.strip()
    if not version_id:
        return {"error": "recommendation_version_id must not be empty"}
    row = fetch_one(
        """
        SELECT
            rv.recommendation_version_id,
            rv.recommendation_id,
            rv.version_number,
            rv.quality_status,
            rv.recommendation_code,
            rv.direction,
            rv.strength,
            rv.certainty,
            rv.population,
            rv.intervention,
            rv.comparator,
            rv.outcome_summary,
            rv.rationale,
            rv.remarks,
            rv.source_text,
            rv.source_span,
            rv.source_section,
            rv.source_url,
            rv.created_at,
            rv.published_at,
            g.guideline_id,
            g.title AS guideline_title,
            g.source,
            g.issuer,
            pq.pico_id,
            pq.clinical_question,
            pq.outcomes AS pico_outcomes,
            gc.grade_candidate_id,
            gc.grade_system,
            gc.source_grade_raw,
            (
                SELECT COUNT(*)
                FROM recommendation_version_evidence_links link
                WHERE link.recommendation_version_id = rv.recommendation_version_id
            ) AS evidence_count
        FROM recommendation_versions rv
        JOIN guidelines g ON g.guideline_id = rv.guideline_id
        LEFT JOIN pico_questions pq ON pq.pico_id = rv.pico_id
        LEFT JOIN grade_candidates gc ON gc.grade_candidate_id = rv.grade_candidate_id
        WHERE rv.recommendation_version_id = %s
        """,
        (version_id,),
    )
    if row is None:
        return {"recommendation_version_id": version_id, "found": False}
    row["found"] = True
    row["source_text"] = truncate_text(row.get("source_text"), 2500)
    row["source_span"] = truncate_text(row.get("source_span"), 2500)
    return row


@mcp.tool()
def get_recommendation_evidence(recommendation_version_id: str, limit: int = 20) -> dict[str, Any]:
    """Return evidence items linked to a recommendation version."""

    version_id = recommendation_version_id.strip()
    safe_limit = clamp_limit(limit, default=20)
    if not version_id:
        return {"error": "recommendation_version_id must not be empty", "results": []}
    rows = fetch_all(
        """
        SELECT
            link.link_id,
            link.link_status,
            link.link_reason,
            ei.evidence_id,
            ei.study_design,
            ei.sample_size,
            ei.effect_direction,
            ei.screening_status,
            ei.extraction_confidence,
            LEFT(ei.population_extracted, 600) AS population_extracted,
            LEFT(ei.intervention_extracted, 600) AS intervention_extracted,
            LEFT(ei.comparator_extracted, 600) AS comparator_extracted,
            ei.outcomes_extracted,
            ei.effect_size,
            ei.confidence_interval,
            LEFT(ei.source_text, 1500) AS source_text,
            ei.source_section,
            ei.source_url,
            p.paper_id,
            p.title AS paper_title,
            p.pmid,
            p.doi,
            p.journal,
            p.publication_date
        FROM recommendation_version_evidence_links link
        JOIN evidence_items ei ON ei.evidence_id = link.evidence_id
        LEFT JOIN papers p ON p.paper_id = ei.paper_id
        WHERE link.recommendation_version_id = %s
        ORDER BY ei.evidence_id
        LIMIT %s
        """,
        (version_id, safe_limit),
    )
    return {"recommendation_version_id": version_id, "limit": safe_limit, "results": rows}


@mcp.tool()
def integrity_report() -> dict[str, Any]:
    """Return read-only integrity checks for foreign-key closure and version quality."""

    table_count_report = table_counts()
    orphan_queries = {
        "recommendation_candidates_without_guideline": """
            SELECT COUNT(*) AS count
            FROM recommendation_candidates rc
            LEFT JOIN guidelines g ON g.guideline_id = rc.guideline_id
            WHERE rc.guideline_id IS NOT NULL AND rc.guideline_id <> '' AND g.guideline_id IS NULL
        """,
        "grade_candidates_without_recommendation": """
            SELECT COUNT(*) AS count
            FROM grade_candidates gc
            LEFT JOIN recommendation_candidates rc ON rc.candidate_id = gc.recommendation_candidate_id
            WHERE gc.recommendation_candidate_id IS NOT NULL
              AND gc.recommendation_candidate_id <> ''
              AND rc.candidate_id IS NULL
        """,
        "evidence_items_without_pico": """
            SELECT COUNT(*) AS count
            FROM evidence_items ei
            LEFT JOIN pico_questions pq ON pq.pico_id = ei.pico_id
            WHERE ei.pico_id IS NOT NULL AND ei.pico_id <> '' AND pq.pico_id IS NULL
        """,
        "recommendation_version_evidence_links_without_version": """
            SELECT COUNT(*) AS count
            FROM recommendation_version_evidence_links link
            LEFT JOIN recommendation_versions rv
              ON rv.recommendation_version_id = link.recommendation_version_id
            WHERE rv.recommendation_version_id IS NULL
        """,
        "recommendation_version_evidence_links_without_evidence": """
            SELECT COUNT(*) AS count
            FROM recommendation_version_evidence_links link
            LEFT JOIN evidence_items ei ON ei.evidence_id = link.evidence_id
            WHERE ei.evidence_id IS NULL
        """,
    }
    version_queries = {
        "versions_without_grade": """
            SELECT COUNT(*) AS count
            FROM recommendation_versions
            WHERE grade_candidate_id IS NULL OR grade_candidate_id = ''
        """,
        "versions_without_pico": """
            SELECT COUNT(*) AS count
            FROM recommendation_versions
            WHERE pico_id IS NULL OR pico_id = ''
        """,
        "versions_without_source_text": """
            SELECT COUNT(*) AS count
            FROM recommendation_versions
            WHERE source_text IS NULL OR source_text = ''
        """,
        "versions_without_source_span": """
            SELECT COUNT(*) AS count
            FROM recommendation_versions
            WHERE source_span IS NULL OR source_span = ''
        """,
    }
    return {
        "table_counts": table_count_report["table_counts"],
        "missing_tables": table_count_report["missing_tables"],
        "orphan_counts": {name: fetch_one(query)["count"] for name, query in orphan_queries.items()},
        "version_quality": {name: fetch_one(query)["count"] for name, query in version_queries.items()},
        "version_review_queue": {
            "review_status_counts": fetch_all(
                """
                SELECT review_status, COUNT(*) AS count
                FROM recommendation_version_review_queue
                GROUP BY review_status
                ORDER BY count DESC, review_status
                """
            ),
            "priority_counts": fetch_all(
                """
                SELECT priority, COUNT(*) AS count
                FROM recommendation_version_review_queue
                GROUP BY priority
                ORDER BY count DESC, priority
                """
            ),
        },
    }


if __name__ == "__main__":
    if MCP_TRANSPORT not in {"stdio", "streamable-http"}:
        raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable-http'")
    mcp.run(transport=MCP_TRANSPORT)

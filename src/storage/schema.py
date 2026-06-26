from __future__ import annotations

from typing import Any

from src.storage.connection import get_connection
from src.storage.schema_definitions import CONSTRAINT_MIGRATIONS, TABLE_DDL, TABLE_MIGRATIONS
from src.storage.utils import sql_identifier


SCHEMA_TABLES = (
    "storage_schema_migrations",
    "cleaned_records",
    "guidelines",
    "papers",
    "model_traces",
    "recommendation_candidates",
    "grade_candidates",
    "pico_questions",
    "recommendation_versions",
    "recommendations",
    "recommendation_version_review_queue",
    "evidence_items",
    "update_logs",
)


def create_schema(recreate: bool = False) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if recreate:
                for table in reversed(SCHEMA_TABLES):
                    cur.execute(f"DROP TABLE IF EXISTS {sql_identifier(table)} CASCADE")
            for ddl in TABLE_DDL:
                cur.execute(ddl)
            for ddl in TABLE_MIGRATIONS:
                cur.execute(ddl)
            for ddl in CONSTRAINT_MIGRATIONS:
                cur.execute(ddl)
            create_indexes(cur)
        conn.commit()
    print("PostgreSQL Living-Guideline schema is ready.")


def create_indexes(cur: Any) -> None:
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_cleaned_records_source ON cleaned_records(source)",
        "CREATE INDEX IF NOT EXISTS idx_cleaned_records_guideline_id ON cleaned_records(guideline_id)",
        "CREATE INDEX IF NOT EXISTS idx_cleaned_records_paper_id ON cleaned_records(paper_id)",
        "CREATE INDEX IF NOT EXISTS idx_cleaned_records_direct_gin ON cleaned_records USING GIN (direct_extraction)",
        "CREATE INDEX IF NOT EXISTS idx_guidelines_source ON guidelines(source)",
        "CREATE INDEX IF NOT EXISTS idx_guidelines_status ON guidelines(status)",
        "CREATE INDEX IF NOT EXISTS idx_papers_source_database ON papers(source_database)",
        "CREATE INDEX IF NOT EXISTS idx_model_traces_input ON model_traces(input_entity_type, input_entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_recommendation_candidates_guideline ON recommendation_candidates(guideline_id)",
        "CREATE INDEX IF NOT EXISTS idx_recommendation_candidates_status ON recommendation_candidates(status)",
        "CREATE INDEX IF NOT EXISTS idx_grade_candidates_recommendation ON grade_candidates(recommendation_candidate_id)",
        "CREATE INDEX IF NOT EXISTS idx_grade_candidates_guideline ON grade_candidates(guideline_id)",
        "CREATE INDEX IF NOT EXISTS idx_recommendation_versions_guideline ON recommendation_versions(guideline_id)",
        "CREATE INDEX IF NOT EXISTS idx_recommendations_guideline_status ON recommendations(guideline_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_recommendations_current_version ON recommendations(current_version_id)",
        "CREATE INDEX IF NOT EXISTS idx_recommendation_version_review_queue_status ON recommendation_version_review_queue(review_status)",
        "CREATE INDEX IF NOT EXISTS idx_recommendation_version_review_queue_guideline ON recommendation_version_review_queue(guideline_id)",
        "CREATE INDEX IF NOT EXISTS idx_evidence_items_pico ON evidence_items(pico_id)",
        "CREATE INDEX IF NOT EXISTS idx_update_logs_guideline ON update_logs(guideline_id)",
    ]
    for sql in indexes:
        cur.execute(sql)

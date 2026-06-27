"""存储层测试文件：验证 PostgreSQL schema、contract、原子入库和完整性检查。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import os
import unittest

from src.pipeline.publish.recommendation_publisher import PublishContext, ingest_publish_bundle, publish_approved_review_item
from src.storage.connection import get_connection
from src.storage.generic_ingest import ingest_tables_atomically
from src.storage.repositories.cleaned_records import insert_guidelines
from src.storage.schema import create_schema


@unittest.skipUnless(os.getenv("RUN_POSTGRES_SMOKE") == "1", "set RUN_POSTGRES_SMOKE=1 to run live PostgreSQL smoke tests")
class PostgreSQLPublishSmokeTests(unittest.TestCase):
    def test_approved_review_publication_persists_full_closed_loop(self) -> None:
        create_schema(recreate=False)
        prefix = "smoke_publish_closed_loop"
        guideline_id = f"{prefix}_guideline"
        trace_id = f"{prefix}_trace"
        candidate_id = f"{prefix}_candidate"
        recommendation_id = f"{prefix}_recommendation"
        version_id = f"{prefix}_version"
        review_id = f"{prefix}_review"

        review_item = {
            "review_item_id": review_id,
            "review_status": "approved",
            "reviewer": "smoke-test",
            "review_reason": "Live database publication smoke test.",
            "reviewed_at": "2026-01-02T00:00:00+00:00",
            "version_payload": {
                "recommendation_version_id": version_id,
                "recommendation_id": recommendation_id,
                "recommendation_candidate_id": candidate_id,
                "guideline_id": guideline_id,
                "version_number": "v1",
                "recommendation_text": "We recommend the smoke-test intervention.",
                "quality_status": "blocked",
                "direction": "for",
                "strength": "strong",
                "certainty": "moderate",
                "change_type": "new",
                "normalized_payload": {
                    "publish_gate": {
                        "quality_status": "blocked",
                        "publishable": False,
                        "blocking_reasons": ["smoke_test_manual_approval"],
                        "warning_reasons": [],
                    }
                },
            },
        }

        with get_connection() as conn:
            try:
                insert_guidelines(
                    conn,
                    [
                        {
                            "guideline_id": guideline_id,
                            "title": "Smoke Test Guideline",
                            "disease_area": "smoke",
                            "guideline_type": "living",
                            "status": "active",
                            "source": "smoke",
                            "issuer": "smoke",
                            "publication_url": None,
                            "pdf_url": None,
                            "current_version": "v1",
                            "update_frequency": "manual",
                            "published_date": "2026-01-01",
                            "last_updated_at": "2026-01-01T00:00:00+00:00",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                        }
                    ],
                )
                conn.commit()
                ingest_tables_atomically(
                    {
                        "model_traces": [
                            {
                                "model_trace_id": trace_id,
                                "task_type": "recommendation_extraction",
                                "method": "rule",
                                "model_name": "rule",
                                "input_entity_type": "block",
                                "input_entity_id": "smoke-block",
                                "input_text": "We recommend the smoke-test intervention.",
                            }
                        ],
                        "recommendation_candidates": [
                            {
                                "candidate_id": candidate_id,
                                "model_trace_id": trace_id,
                                "statement": "We recommend the smoke-test intervention.",
                                "recommendation_text": "We recommend the smoke-test intervention.",
                                "guideline_id": guideline_id,
                                "status": "accepted",
                            }
                        ],
                    },
                    conn=conn,
                )
                bundle = publish_approved_review_item(
                    review_item,
                    context=PublishContext(published_by="smoke-test", published_at="2026-01-03T00:00:00+00:00"),
                )
                ingest_summary = ingest_publish_bundle(bundle, conn=conn)

                with conn.cursor() as cur:
                    cur.execute("SELECT current_version_id, status FROM recommendations WHERE recommendation_id = %s", (recommendation_id,))
                    recommendation_row = cur.fetchone()
                    cur.execute("SELECT quality_status FROM recommendation_versions WHERE recommendation_version_id = %s", (version_id,))
                    version_row = cur.fetchone()
                    cur.execute("SELECT COUNT(*) FROM update_logs WHERE new_recommendation_version_id = %s", (version_id,))
                    update_count = cur.fetchone()[0]
                    cur.execute("SELECT review_status, published_version_id FROM recommendation_version_review_queue WHERE review_item_id = %s", (review_id,))
                    review_row = cur.fetchone()

                self.assertEqual(ingest_summary["status"], "ingested")
                self.assertEqual(recommendation_row, (version_id, "active"))
                self.assertEqual(version_row, ("publishable",))
                self.assertEqual(update_count, 1)
                self.assertEqual(review_row, ("published", version_id))
            finally:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM update_logs WHERE new_recommendation_version_id = %s", (version_id,))
                    cur.execute("DELETE FROM recommendation_version_review_queue WHERE review_item_id = %s", (review_id,))
                    cur.execute("DELETE FROM recommendations WHERE recommendation_id = %s", (recommendation_id,))
                    cur.execute("DELETE FROM recommendation_versions WHERE recommendation_version_id = %s", (version_id,))
                    cur.execute("DELETE FROM recommendation_candidates WHERE candidate_id = %s", (candidate_id,))
                    cur.execute("DELETE FROM model_traces WHERE model_trace_id = %s", (trace_id,))
                    cur.execute("DELETE FROM guidelines WHERE guideline_id = %s", (guideline_id,))
                conn.commit()


if __name__ == "__main__":
    unittest.main()


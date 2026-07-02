"""存储层测试文件：验证 PostgreSQL schema、contract、原子入库和完整性检查。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest
from unittest.mock import patch

from src.storage.generic_ingest import (
    ingest_tables_atomically,
    partition_recommendation_versions_for_ingest,
    recommendation_version_review_row,
)


def _publishable_version(version_id: str = "version-1"):
    return {
        "recommendation_version_id": version_id,
        "quality_status": "publishable",
        "normalized_payload": {
            "publish_gate": {
                "quality_status": "publishable",
                "publishable": True,
                "blocking_reasons": [],
                "warning_reasons": [],
            }
        },
    }


class FakeConnection:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class AtomicIngestionTests(unittest.TestCase):
    def test_multi_table_bundle_commits_once(self) -> None:
        conn = FakeConnection()
        bundle = {
            "model_traces": [{"model_trace_id": "trace-1"}],
            "recommendation_versions": [_publishable_version()],
        }

        with patch("src.storage.generic_ingest._upsert_rows", side_effect=[1, 1]) as upsert:
            counts = ingest_tables_atomically(bundle, conn=conn)

        self.assertEqual(counts, {"model_traces": 1, "recommendation_versions": 1})
        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.rollbacks, 0)
        self.assertEqual(upsert.call_args_list[0].args[1], "model_traces")

    def test_recommendation_versions_publish_gate_skips_non_publishable_rows(self) -> None:
        conn = FakeConnection()
        bundle = {
            "recommendation_versions": [
                _publishable_version("version-publishable"),
                {
                    "recommendation_version_id": "version-review",
                    "quality_status": "needs_review",
                    "normalized_payload": {
                        "publish_gate": {
                            "quality_status": "needs_review",
                            "publishable": False,
                            "blocking_reasons": [],
                            "warning_reasons": ["no_accepted_grade"],
                        }
                    },
                },
                {
                    "recommendation_version_id": "version-blocked",
                    "quality_status": "blocked",
                    "normalized_payload": {
                        "publish_gate": {
                            "quality_status": "blocked",
                            "publishable": False,
                            "blocking_reasons": ["missing_source_span_coordinates"],
                            "warning_reasons": [],
                        }
                    },
                },
            ]
        }

        with patch("src.storage.generic_ingest._upsert_rows", return_value=1) as upsert:
            counts = ingest_tables_atomically(bundle, conn=conn)

        self.assertEqual(counts["recommendation_versions"], 1)
        self.assertEqual(counts["recommendation_version_review_queue"], 1)
        self.assertEqual(counts["recommendation_versions_skipped_by_publish_gate"], 2)
        self.assertEqual(upsert.call_args_list[0].args[1], "recommendation_versions")
        self.assertEqual(upsert.call_args_list[0].args[2][0]["recommendation_version_id"], "version-publishable")
        self.assertEqual(upsert.call_args_list[1].args[1], "recommendation_version_review_queue")
        review_rows = upsert.call_args_list[1].args[2]
        self.assertEqual({row["recommendation_version_id"] for row in review_rows}, {"version-review", "version-blocked"})
        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.rollbacks, 0)

    def test_review_row_preserves_publish_gate_payload(self) -> None:
        row = {
            "recommendation_version_id": "version-blocked",
            "recommendation_candidate_id": "candidate-1",
            "guideline_id": "guideline-1",
            "record_id": "record-1",
            "quality_status": "blocked",
            "normalized_payload": {
                "publish_gate": {
                    "quality_status": "blocked",
                    "publishable": False,
                    "blocking_reasons": ["missing_source_span_coordinates"],
                    "warning_reasons": ["no_linked_pico"],
                }
            },
        }

        review = recommendation_version_review_row(row)

        self.assertEqual(review["review_item_id"], "review_version-blocked")
        self.assertEqual(review["quality_status"], "blocked")
        self.assertEqual(review["priority"], "high")
        self.assertEqual(review["blocking_reasons"], ["missing_source_span_coordinates"])
        self.assertEqual(review["warning_reasons"], ["no_linked_pico"])
        self.assertEqual(review["version_payload"], row)

    def test_recommendation_version_partition_reports_reasons(self) -> None:
        publishable, report = partition_recommendation_versions_for_ingest(
            [
                _publishable_version(),
                {
                    "recommendation_version_id": "version-review",
                    "quality_status": "needs_review",
                    "normalized_payload": {
                        "publish_gate": {
                            "quality_status": "needs_review",
                            "publishable": False,
                            "warning_reasons": ["no_linked_pico"],
                        }
                    },
                },
            ]
        )

        self.assertEqual(len(publishable), 1)
        self.assertEqual(report["skipped_rows"], 1)
        self.assertEqual(report["skipped_status_counts"], {"needs_review": 1})
        self.assertEqual(report["skipped_reason_counts"], {"no_linked_pico": 1})

    def test_multi_table_bundle_rolls_back_on_any_failure(self) -> None:
        conn = FakeConnection()
        bundle = {
            "model_traces": [{"model_trace_id": "trace-1"}],
            "recommendation_versions": [_publishable_version()],
        }

        with patch("src.storage.generic_ingest._upsert_rows", side_effect=[1, ValueError("bad version")]):
            with self.assertRaisesRegex(RuntimeError, "transaction rolled back"):
                ingest_tables_atomically(bundle, conn=conn)

        self.assertEqual(conn.commits, 0)
        self.assertEqual(conn.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()


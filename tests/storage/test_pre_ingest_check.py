import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import write_jsonl
from src.storage.generic_ingest import GENERIC_TABLE_CONFIG
from src.storage.pre_ingest_check import build_pre_ingest_report
from src.storage.schema_definitions import TABLE_MIGRATIONS


class PreIngestCheckTests(unittest.TestCase):
    def test_pre_ingest_report_separates_staging_from_formal_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(run_dir / "recommendation_candidates.jsonl", [{"candidate_id": "rec_1", "status": "pending"}])
            write_jsonl(
                run_dir / "grade_candidates.jsonl",
                [
                    {
                        "grade_candidate_id": "grade_1",
                        "status": "needs_review",
                        "recommendation_candidate_id": "",
                        "normalized_payload": {"association_quality": "missing"},
                    }
                ],
            )
            write_jsonl(run_dir / "pico_questions.jsonl", [{"pico_id": "pico_1", "status": "under_review"}])
            write_jsonl(
                run_dir / "evidence_items.jsonl",
                [
                    {"evidence_id": "evidence_1", "pico_id": "pico_1", "screening_status": "pending"},
                    {"evidence_id": "evidence_2", "pico_id": "", "screening_status": "association_review"},
                ],
            )
            write_jsonl(run_dir / "recommendation_versions.jsonl", [])

            report = build_pre_ingest_report(run_dir)

            self.assertTrue(report["staging_gate"]["passed"])
            self.assertFalse(report["formal_publish_gate"]["passed"])
            self.assertIn("no_publishable_recommendation_versions", report["formal_publish_gate"]["blocking_reasons"])
            self.assertEqual(report["staging_gate"]["evidence_items_with_pico_for_staging"], 1)
            self.assertEqual(report["staging_gate"]["evidence_items_to_review_before_storage"], 1)
            self.assertEqual(report["entity_reports"]["grades"]["association_quality_counts"], {"missing": 1})

    def test_candidate_payloads_are_preserved_by_generic_ingest_config(self) -> None:
        self.assertIn("normalized_payload", GENERIC_TABLE_CONFIG["recommendation_candidates"]["json_fields"])
        self.assertIn("raw_payload", GENERIC_TABLE_CONFIG["recommendation_candidates"]["json_fields"])
        self.assertIn("normalized_payload", GENERIC_TABLE_CONFIG["grade_candidates"]["json_fields"])
        self.assertIn("raw_payload", GENERIC_TABLE_CONFIG["grade_candidates"]["json_fields"])
        self.assertIn("normalized_payload", GENERIC_TABLE_CONFIG["pico_questions"]["json_fields"])
        self.assertIn("raw_payload", GENERIC_TABLE_CONFIG["pico_questions"]["json_fields"])
        self.assertIn("normalized_payload", GENERIC_TABLE_CONFIG["evidence_items"]["json_fields"])
        self.assertIn("raw_payload", GENERIC_TABLE_CONFIG["evidence_items"]["json_fields"])
        self.assertIn(
            "ALTER TABLE grade_candidates ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
            TABLE_MIGRATIONS,
        )
        self.assertIn(
            "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
            TABLE_MIGRATIONS,
        )
        self.assertIn(
            "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
            TABLE_MIGRATIONS,
        )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import write_jsonl
from src.storage.contract_check import build_storage_contract_report
from src.storage.generic_ingest import prepare_generic_row


class StorageContractCheckTests(unittest.TestCase):
    def test_contract_accepts_reviewed_run_with_legacy_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "recommendation_candidates.jsonl",
                [
                    {
                        "candidate_id": "rec_1",
                        "model_trace_id": "trace_1",
                        "recommendation_text": "Use treatment for eligible adults.",
                        "source_text": "Use treatment for eligible adults.",
                        "normalized_payload": {},
                        "raw_payload": {},
                    }
                ],
            )
            write_jsonl(
                run_dir / "grade_candidates.jsonl",
                [
                    {
                        "grade_candidate_id": "grade_1",
                        "model_trace_id": "trace_2",
                        "source_text": "Strong recommendation, low certainty.",
                        "normalized_payload": {},
                        "raw_payload": {},
                    }
                ],
            )
            write_jsonl(
                run_dir / "pico_questions.jsonl",
                [
                    {
                        "pico_id": "pico_1",
                        "clinical_question": "In adults should treatment improve outcomes?",
                        "population": "adults",
                        "intervention": "treatment",
                        "normalized_payload": {},
                    }
                ],
            )
            write_jsonl(
                run_dir / "evidence_items.jsonl",
                [
                    {
                        "evidence_id": "evidence_1",
                        "pico_id": "pico_1",
                        "normalized_payload": {},
                    }
                ],
            )
            write_jsonl(
                run_dir / "recommendation_versions.jsonl",
                [
                    {
                        "recommendation_version_id": "version_1",
                        "recommendation_candidate_id": "rec_1",
                        "guideline_id": "guideline_1",
                        "version_number": "v1",
                        "recommendation_text": "Use treatment for eligible adults.",
                        "normalized_payload": {},
                        "raw_payload": {},
                    }
                ],
            )

            report = build_storage_contract_report(run_dir)

            self.assertTrue(report["passed"])
            self.assertEqual(report["blocking_reasons"], [])
            self.assertEqual(
                report["table_reports"]["recommendation_candidates"]["missing_required_storage_fields"]["statement"],
                0,
            )
            self.assertEqual(
                report["table_reports"]["grade_candidates"]["missing_required_storage_fields"]["source_grade_raw"],
                0,
            )

    def test_contract_blocks_missing_required_storage_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for filename in [
                "recommendation_candidates.jsonl",
                "pico_questions.jsonl",
                "evidence_items.jsonl",
                "recommendation_versions.jsonl",
            ]:
                write_jsonl(run_dir / filename, [])
            write_jsonl(
                run_dir / "grade_candidates.jsonl",
                [{"grade_candidate_id": "grade_1", "model_trace_id": "trace_1"}],
            )

            report = build_storage_contract_report(run_dir)

            self.assertFalse(report["passed"])
            self.assertIn("grade_candidates.source_grade_raw_missing:1", report["blocking_reasons"])

    def test_prepare_generic_row_uses_aliases_before_defaults(self) -> None:
        row = prepare_generic_row(
            {
                "recommendation_text": "Use treatment.",
                "source_text": "Source sentence.",
                "extraction_confidence": 0.87,
            },
            ["statement", "raw_text", "confidence"],
            set(),
            set(),
            {
                "statement": ("recommendation_text",),
                "raw_text": ("source_text",),
                "confidence": ("extraction_confidence",),
            },
        )

        self.assertEqual(row["statement"], "Use treatment.")
        self.assertEqual(row["raw_text"], "Source sentence.")
        self.assertEqual(row["confidence"], 0.87)


if __name__ == "__main__":
    unittest.main()

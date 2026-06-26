import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import read_jsonl, write_jsonl
from src.pipeline.cleaning.quality_gate import (
    NEEDS_LAYOUT_REPAIR,
    PARSE_FAILED,
    QualityGatePaths,
    READY,
    evaluate_cleaned_record,
    route_file,
)
from src.pipeline.cleaning.source_cleaner import clean_source_record


class CleaningQualityGateTests(unittest.TestCase):
    def test_normal_guideline_record_is_ready(self) -> None:
        record = clean_source_record(
            {
                "content": (
                    "Recommendations\n"
                    "Recommendation 1: In adults with insomnia, we suggest cognitive behavioral therapy.\n"
                    "The overall certainty of evidence was low due to risk of bias and imprecision."
                ),
                "title": "Demo Guideline",
                "source": "aasm",
            }
        )

        result = evaluate_cleaned_record(record)

        self.assertEqual(result.status, READY)
        self.assertGreaterEqual(result.score, 0.75)
        self.assertIn(result.record_type, {"guideline", "mixed"})
        self.assertEqual(result.record["cleaning_status"], READY)
        self.assertIn("record_type", result.record["direct_extraction"])

    def test_empty_clean_content_is_parse_failed(self) -> None:
        result = evaluate_cleaned_record(
            {
                "record_id": "empty-1",
                "source": "who",
                "title": "Empty",
                "raw_content": "",
                "clean_content": "",
                "sections": [],
            }
        )

        self.assertEqual(result.status, PARSE_FAILED)
        self.assertEqual(result.score, 0.0)
        self.assertIn("empty_raw_content", {flag["code"] for flag in result.flags})
        self.assertIn("empty_clean_content", {flag["code"] for flag in result.flags})

    def test_huge_single_section_goes_to_layout_repair(self) -> None:
        content = "Recommendation: use treatment.\n" + ("Evidence supports treatment. " * 24_000)
        result = evaluate_cleaned_record(
            {
                "record_id": "huge-1",
                "source": "nice",
                "title": "Huge Guideline",
                "raw_content": content,
                "clean_content": content,
                "content": content,
                "sections": [{"section_name": "Document", "text": content}],
                "table_count": 0,
            }
        )

        self.assertEqual(result.status, NEEDS_LAYOUT_REPAIR)
        self.assertIn("single_section_huge_document", {flag["code"] for flag in result.flags})

    def test_route_file_writes_all_gate_outputs_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cleaned.jsonl"
            ready = root / "ready.jsonl"
            repair = root / "repair.jsonl"
            failed = root / "failed.jsonl"
            skipped = root / "skipped.jsonl"
            report = root / "report.jsonl"

            good = clean_source_record(
                {
                    "content": (
                        "Recommendations\n"
                        "Recommendation 1: We suggest treatment for adults with disease.\n"
                        "The certainty of evidence is moderate."
                    ),
                    "title": "Good Guideline",
                    "source": "aasm",
                }
            )
            bad = {
                "record_id": "bad-1",
                "source": "pmc",
                "title": "Bad",
                "raw_content": "",
                "clean_content": "",
                "content": "",
                "sections": [],
            }
            write_jsonl(input_path, [good, bad])

            summary = route_file(QualityGatePaths(input_path, ready, repair, failed, skipped, report))

            self.assertEqual(summary["total_records"], 2)
            self.assertEqual(summary["status_counts"][READY], 1)
            self.assertEqual(summary["status_counts"][PARSE_FAILED], 1)
            self.assertEqual(len(read_jsonl(ready)), 1)
            self.assertEqual(len(read_jsonl(failed)), 1)
            self.assertTrue(report.exists())

    def test_route_file_rejects_overwriting_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cleaned.jsonl"
            write_jsonl(input_path, [{"record_id": "1", "raw_content": "", "clean_content": ""}])

            with self.assertRaises(ValueError):
                route_file(QualityGatePaths(input_path, input_path, root / "repair.jsonl", root / "failed.jsonl", root / "skipped.jsonl", root / "report.jsonl"))


if __name__ == "__main__":
    unittest.main()

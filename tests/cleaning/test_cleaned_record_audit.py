"""清洗阶段测试文件：验证 source_cleaner、quality_gate、offset 和记录类型路由等清洗链路行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_cleaned_records import audit_file, classify_record, garbled_text_score


class CleanedRecordAuditTests(unittest.TestCase):
    def test_empty_record_routes_to_source_reextract(self) -> None:
        route, reasons, metrics = classify_record({"raw_content": "", "clean_content": "", "metadata": {"source": "who"}})

        self.assertEqual(route, "needs_source_reextract")
        self.assertIn("empty_raw_content", reasons)
        self.assertIn("empty_clean_content", reasons)
        self.assertEqual(metrics["clean_ratio"], 0.0)

    def test_garbled_text_routes_to_source_reextract(self) -> None:
        text = '!!"# !!$% "#$%&’"*(& +",""./$001*2 ' * 80

        route, reasons, metrics = classify_record({"raw_content": text * 3, "clean_content": text, "metadata": {"source": "who"}})

        self.assertEqual(route, "needs_source_reextract")
        self.assertIn("garbled_text", reasons)
        self.assertGreaterEqual(metrics["garbled_text_score"], 0.35)
        self.assertGreaterEqual(garbled_text_score(text), 0.35)

    def test_reference_heading_warning_alone_does_not_block_ready(self) -> None:
        record = {
            "raw_content": "Recommendations\nWe recommend treatment.\nReferences\n1. Smith J. Guideline. 2020.",
            "clean_content": "Recommendations\nWe recommend treatment.",
            "references_text": "References\n1. Smith J. Guideline. 2020.",
            "recommendations": [{"text": "We recommend treatment."}],
            "cleaning_log": {"audit": {"audit_flags": ["references_contains_body_heading"]}},
        }

        route, reasons, _ = classify_record(record)

        self.assertEqual(route, "ready")
        self.assertIn("references_contains_body_heading", reasons)

    def test_deletion_budget_routes_to_review(self) -> None:
        record = {
            "raw_content": "x" * 1000,
            "clean_content": "x" * 500,
            "references_text": "",
            "cleaning_log": {"audit": {"audit_flags": ["removed_ratio_high"]}},
        }

        route, reasons, _ = classify_record(record)

        self.assertEqual(route, "needs_review")
        self.assertIn("cleaning_deletion_budget_warning", reasons)

    def test_residual_nice_boilerplate_routes_to_review(self) -> None:
        record = {
            "raw_content": "Recommendations\nWe recommend treatment.",
            "clean_content": "See www.nice.org.uk/guidance/NG106 Recommendations\nWe recommend treatment.",
            "recommendations": [{"text": "We recommend treatment."}],
            "metadata": {"source": "nice"},
        }

        route, reasons, metrics = classify_record(record)

        self.assertEqual(route, "needs_review")
        self.assertIn("residual_boilerplate", reasons)
        self.assertTrue(metrics["residual_boilerplate"])

    def test_audit_file_writes_route_outputs_and_summary(self) -> None:
        records = [
            {"raw_content": "Recommendations\nWe recommend treatment.", "clean_content": "Recommendations\nWe recommend treatment.", "recommendations": [{"text": "We recommend treatment."}]},
            {"raw_content": "x" * 1000, "clean_content": "x" * 500, "cleaning_log": {"audit": {"audit_flags": ["removed_ratio_high"]}}},
            {"raw_content": "", "clean_content": "", "metadata": {"source": "who"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_dir = Path(tmp) / "audit"
            with input_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            summary = audit_file(input_path, output_dir, sample_limit=5)

            self.assertEqual(summary["route_counts"]["ready"], 1)
            self.assertEqual(summary["route_counts"]["needs_review"], 1)
            self.assertEqual(summary["route_counts"]["needs_source_reextract"], 1)
            self.assertTrue((output_dir / "cleaned.ready.jsonl").exists())
            self.assertTrue((output_dir / "cleaned.needs_review.jsonl").exists())
            self.assertTrue((output_dir / "cleaned.needs_source_reextract.jsonl").exists())
            self.assertTrue((output_dir / "cleaning_quality_summary.json").exists())


if __name__ == "__main__":
    unittest.main()


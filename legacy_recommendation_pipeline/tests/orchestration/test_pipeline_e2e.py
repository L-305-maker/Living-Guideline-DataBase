"""编排流程测试文件：验证端到端流水线入口、公开 API 和阶段衔接是否正常。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import read_jsonl, write_jsonl
from src.pipeline.cleaning.source_cleaner import clean_file
from src.pipeline.cleaning.quality_gate import QualityGatePaths, route_file as route_cleaned_file
from src.pipeline.extraction.routing.candidate_router import route_file
from src.pipeline.extraction.grade.candidate_extractor import extract_file as extract_grade_file
from src.pipeline.extraction.recommendation.candidate_extractor import extract_file as extract_recommendation_file
from src.pipeline.llm_review.queue.builder import build_queue_file
from src.pipeline.parsing.structure_parser import parse_file


class PipelineEndToEndTests(unittest.TestCase):
    def test_guideline_record_reaches_review_queue_with_structured_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = root / "origin.jsonl"
            cleaned = root / "cleaned.jsonl"
            ready_cleaned = root / "cleaned.ready.jsonl"
            layout_repair = root / "cleaned.needs_layout_repair.jsonl"
            parse_failed = root / "cleaned.parse_failed.jsonl"
            skipped = root / "cleaned.skipped.jsonl"
            gate_report = root / "cleaned.quality_report.jsonl"
            blocks = root / "blocks.jsonl"
            routes = root / "routes"
            recs = root / "recs.jsonl"
            rec_traces = root / "rec_traces.jsonl"
            grades = root / "grades.jsonl"
            grade_traces = root / "grade_traces.jsonl"
            queue = root / "queue.jsonl"
            queue_summary = root / "queue_summary.jsonl"

            raw_content = (
                "Recommendations\n"
                "Recommendation 1: In adults with insomnia, we suggest cognitive behavioral therapy.\n"
                "The overall certainty of evidence was low due to risk of bias and imprecision.\n"
                "Table 1\nCritical outcomes\n"
                "References\n1. Smith J. Lancet. 2020."
            )
            write_jsonl(
                origin,
                [
                    {
                        "content": raw_content,
                        "tables": [],
                        "title": "Demo Guideline",
                        "source": "aasm",
                        "published_year": "2026",
                    }
                ],
            )

            clean_file(origin, cleaned)
            gate_summary = route_cleaned_file(QualityGatePaths(cleaned, ready_cleaned, layout_repair, parse_failed, skipped, gate_report))
            parse_file(ready_cleaned, blocks)
            route_file(blocks, routes, prefix="flow")
            extract_recommendation_file(routes / "flow_recommendation_blocks.jsonl", recs, rec_traces)
            extract_grade_file(routes / "flow_grade_blocks.jsonl", recs, grades, grade_traces)
            summary = build_queue_file(recs, grades, queue, queue_summary)

            cleaned_row = read_jsonl(cleaned)[0]
            ready_row = read_jsonl(ready_cleaned)[0]
            self.assertEqual(gate_summary["status_counts"]["ready"], 1)
            self.assertEqual(cleaned_row["raw_content"], raw_content)
            self.assertNotIn("Smith J.", cleaned_row["clean_content"])
            self.assertTrue(cleaned_row["tables"])
            self.assertEqual(cleaned_row["cleaning_warnings"][0]["warning_type"], "tables_empty_but_table_text_detected")
            self.assertEqual(ready_row["cleaning_status"], "ready")
            self.assertIn("record_type", ready_row)
            block = read_jsonl(blocks)[0]
            self.assertEqual(block["quality"]["record_cleaning_status"], "ready")
            self.assertEqual(block["metadata"]["record_cleaning_status"], "ready")
            self.assertIn("record_type", block["metadata"])

            recommendation = read_jsonl(recs)[0]
            self.assertEqual(recommendation["strength"], "conditional")

            grade = read_jsonl(grades)[0]
            self.assertEqual(grade["certainty"], "low")
            self.assertEqual(grade["risk_of_bias"], "serious_or_concern")
            self.assertEqual(grade["imprecision"], "serious_or_concern")
            self.assertEqual(grade["inconsistency"], "not_reported")

            self.assertGreaterEqual(summary["queue_items"], 1)
            self.assertTrue(read_jsonl(queue))


if __name__ == "__main__":
    unittest.main()


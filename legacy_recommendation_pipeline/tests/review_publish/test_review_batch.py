"""复核与发布测试文件：验证 review queue、backfill、release export 和正式发布闭包逻辑。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import iter_jsonl, write_jsonl
from src.pipeline.review.review_batch import build_review_batch, stratified_limit_rows


class ReviewBatchTests(unittest.TestCase):
    def test_build_review_batch_exports_entity_and_first_pass_queues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "recommendation_candidates.jsonl",
                [
                    {
                        "candidate_id": "rec_1",
                        "status": "pending",
                        "recommendation_text": "We suggest treatment.",
                        "direction": "for",
                        "strength": "conditional",
                        "certainty": "unclear",
                        "extraction_confidence": 0.61,
                    }
                ],
            )
            write_jsonl(
                run_dir / "grade_candidates.jsonl",
                [
                    {
                        "grade_candidate_id": "grade_1",
                        "status": "needs_review",
                        "recommendation_candidate_id": "",
                        "grade_system": "GRADE",
                        "certainty": "low",
                    }
                ],
            )
            write_jsonl(
                run_dir / "pico_questions.jsonl",
                [{"pico_id": "pico_1", "status": "under_review", "clinical_question": "Should adults use treatment?"}],
            )
            write_jsonl(
                run_dir / "evidence_items.jsonl",
                [{"evidence_id": "evidence_1", "screening_status": "pending", "study_design": "unclear"}],
            )
            write_jsonl(
                run_dir / "llm_review_queue.jsonl",
                [{"queue_id": "llm_1", "priority": "P0"}, {"queue_id": "llm_2", "priority": "P2"}],
            )

            summary = build_review_batch(run_dir)
            batch_dir = Path(summary["output_dir"])

            self.assertTrue((batch_dir / "recommendation_review_queue.jsonl").exists())
            self.assertTrue((batch_dir / "grade_review_queue.jsonl").exists())
            self.assertTrue((batch_dir / "pico_review_queue.jsonl").exists())
            self.assertTrue((batch_dir / "evidence_review_queue.jsonl").exists())
            self.assertTrue((batch_dir / "first_pass_review_queue.jsonl").exists())
            self.assertTrue((batch_dir / "recommendation_association_review_queue.jsonl").exists())
            self.assertTrue((batch_dir / "evidence_pico_review_queue.jsonl").exists())
            self.assertEqual(summary["llm_priority_records"], 1)
            self.assertIsNotNone(summary["association_review"])
            self.assertGreaterEqual(summary["first_pass_review_records"], 4)
            first_pass = list(iter_jsonl(batch_dir / "first_pass_review_queue.jsonl"))
            self.assertTrue(all(item["priority"] in {"P0", "P1"} for item in first_pass))

    def test_build_review_batch_can_override_grade_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            override_dir = run_dir / "override"
            write_jsonl(run_dir / "grade_candidates.jsonl", [])
            write_jsonl(
                override_dir / "grade_candidates.jsonl",
                [
                    {
                        "grade_candidate_id": "grade_override",
                        "status": "needs_review",
                        "recommendation_candidate_id": "rec_1",
                        "grade_system": "GRADE",
                        "certainty": "low",
                        "normalized_payload": {"association_quality": "weak"},
                    }
                ],
            )

            summary = build_review_batch(
                run_dir,
                input_overrides={"grade_candidate": override_dir / "grade_candidates.jsonl"},
            )
            batch_dir = Path(summary["output_dir"])
            grade_queue = list(iter_jsonl(batch_dir / "grade_review_queue.jsonl"))

            self.assertEqual(len(grade_queue), 1)
            self.assertIn("weak_grade_association", grade_queue[0]["review_reasons"])

    def test_first_pass_selection_is_stratified_by_record(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                {
                    "entity_id": f"a_{index}",
                    "entity_type": "evidence_item",
                    "priority": "P1",
                    "current_state": {"record_id": "record_a", "source_order": index},
                }
            )
        for index in range(2):
            rows.append(
                {
                    "entity_id": f"b_{index}",
                    "entity_type": "evidence_item",
                    "priority": "P1",
                    "current_state": {"record_id": "record_b", "source_order": index},
                }
            )

        selected = stratified_limit_rows(rows, 4)

        self.assertEqual([item["current_state"]["record_id"] for item in selected], ["record_a", "record_b", "record_a", "record_b"])


if __name__ == "__main__":
    unittest.main()


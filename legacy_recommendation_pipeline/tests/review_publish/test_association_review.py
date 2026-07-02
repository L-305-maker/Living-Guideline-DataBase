"""复核与发布测试文件：验证 review queue、backfill、release export 和正式发布闭包逻辑。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import iter_jsonl
from src.pipeline.review.association_review import AssociationReviewOptions, build_association_review_package
from src.pipeline.review.rule_assisted_backfill import BackfillInputs


class AssociationReviewTests(unittest.TestCase):
    def test_builds_recommendation_and_evidence_pico_association_batches(self) -> None:
        rec = {
            "candidate_id": "rec_1",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "status": "pending",
            "recommendation_text": "We recommend antiviral treatment for adults with severe infection.",
            "source_text": "We recommend antiviral treatment for adults with severe infection.",
            "source_order": 10,
            "source_section": "Recommendations Antiviral treatment",
            "direction": "for",
            "strength": "strong",
            "certainty": "moderate",
            "extraction_confidence": 0.91,
            "normalized_payload": {"quality_notes": [], "source_order": 10},
        }
        grade = {
            "grade_candidate_id": "grade_1",
            "recommendation_candidate_id": "",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "grade_system": "GRADE",
            "strength": "strong",
            "certainty": "moderate",
            "source_text": "Strong recommendation, moderate certainty evidence for antiviral treatment.",
            "source_order": 11,
            "source_section": "Recommendations Antiviral treatment",
            "extraction_confidence": 0.9,
            "normalized_payload": {"association_quality": "missing", "source_order": 11},
        }
        pico = {
            "pico_id": "pico_1",
            "source_record_id": "record_1",
            "guideline_id": "guideline_1",
            "status": "under_review",
            "population": "adults with severe infection",
            "intervention": "antiviral treatment",
            "clinical_question": "Should adults with severe infection use antiviral treatment?",
            "source_order": 9,
            "source_section": "Recommendations Antiviral treatment",
        }
        evidence = {
            "evidence_id": "evidence_1",
            "source_record_id": "record_1",
            "guideline_id": "guideline_1",
            "recommendation_candidate_id": "",
            "pico_id": "",
            "screening_status": "association_review",
            "study_design": "systematic_review",
            "effect_direction": "benefit",
            "source_text": "A systematic review found benefit from antiviral treatment in adults with severe infection.",
            "source_order": 12,
            "source_section": "Evidence Antiviral treatment",
            "normalized_payload": {"has_structured_evidence_signal": True, "source_order": 12},
        }

        with tempfile.TemporaryDirectory() as tmp:
            summary = build_association_review_package(
                BackfillInputs([rec], [grade], [pico], [evidence]),
                tmp,
                AssociationReviewOptions(batch_size=1),
            )
            output_dir = Path(tmp)
            recommendation_items = list(iter_jsonl(output_dir / "recommendation_association_review_queue.jsonl"))
            evidence_items = list(iter_jsonl(output_dir / "evidence_pico_review_queue.jsonl"))

        self.assertEqual(summary["recommendation_association_queue"]["records"], 1)
        self.assertEqual(summary["evidence_pico_queue"]["records"], 1)
        self.assertEqual(recommendation_items[0]["priority"], "P0")
        self.assertEqual(recommendation_items[0]["batch_index"], 1)
        self.assertEqual(recommendation_items[0]["reviewed_payload_template"]["grade_candidate_id"], "grade_1")
        self.assertEqual(recommendation_items[0]["reviewed_payload_template"]["pico_id"], "pico_1")
        self.assertEqual(recommendation_items[0]["reviewed_payload_template"]["evidence_ids"], ["evidence_1"])
        self.assertEqual(evidence_items[0]["reviewed_payload_template"]["pico_id"], "pico_1")
        self.assertIn(evidence_items[0]["priority"], {"P0", "P1"})

    def test_manually_resolved_association_is_not_requeued(self) -> None:
        rec = {
            "candidate_id": "rec_resolved",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "status": "needs_review",
            "recommendation_text": "We recommend antiviral treatment for adults with severe infection.",
            "source_order": 10,
            "source_section": "Recommendations Antiviral treatment",
            "extraction_confidence": 0.9,
            "normalized_payload": {
                "quality_notes": [],
                "source_order": 10,
                "manual_association_processing": {"review_decision": "needs_review"},
            },
        }
        grade = {
            "grade_candidate_id": "grade_1",
            "recommendation_candidate_id": "rec_resolved",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "grade_system": "GRADE",
            "strength": "strong",
            "certainty": "moderate",
            "source_text": "Strong recommendation, moderate certainty evidence for antiviral treatment.",
            "source_order": 11,
            "normalized_payload": {"association_quality": "strong", "source_order": 11},
        }

        with tempfile.TemporaryDirectory() as tmp:
            summary = build_association_review_package(BackfillInputs([rec], [grade], [], []), tmp)

        self.assertEqual(summary["recommendation_association_queue"]["records"], 0)


if __name__ == "__main__":
    unittest.main()


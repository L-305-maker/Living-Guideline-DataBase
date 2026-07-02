"""复核与发布测试文件：验证 review queue、backfill、release export 和正式发布闭包逻辑。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.pipeline.review.rule_assisted_backfill import BackfillInputs, backfill_inputs


class RuleAssistedBackfillTests(unittest.TestCase):
    def test_high_confidence_recommendation_is_accepted_and_backfilled(self) -> None:
        rec = {
            "candidate_id": "rec_1",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "recommendation_text": "We suggest treatment for adults with disease.",
            "source_text": "Intro. We suggest treatment for adults with disease. End.",
            "source_block_id": "block_1",
            "direction": "for",
            "strength": "unclear",
            "certainty": "unclear",
            "extraction_confidence": 0.9,
            "status": "pending",
            "normalized_payload": {"quality_notes": [], "source_order": 10},
            "raw_payload": {"quality": {"record_quality_flags": []}},
        }
        grade = {
            "grade_candidate_id": "grade_1",
            "recommendation_candidate_id": "rec_1",
            "grade_system": "GRADE",
            "strength": "conditional",
            "certainty": "low",
            "source_text": "Conditional recommendation, low certainty of evidence.",
            "extraction_confidence": 0.9,
            "status": "pending",
            "normalized_payload": {"association_quality": "strong"},
        }
        pico = {
            "pico_id": "pico_1",
            "guideline_id": "guideline_1",
            "source_record_id": "record_1",
            "population": "adults with disease",
            "intervention": "treatment",
            "status": "under_review",
        }
        evidence = {
            "evidence_id": "evidence_1",
            "recommendation_candidate_id": "rec_1",
            "pico_id": "pico_1",
            "screening_status": "pending",
        }

        summary, outputs = backfill_inputs(BackfillInputs([rec], [grade], [pico], [evidence]))

        self.assertEqual(summary["accepted_recommendations"], 1)
        reviewed_rec = outputs["recommendations"][0]
        self.assertEqual(reviewed_rec["status"], "accepted")
        self.assertEqual(reviewed_rec["strength"], "conditional")
        self.assertEqual(reviewed_rec["certainty"], "low")
        self.assertEqual(reviewed_rec["pico_id"], "pico_1")
        self.assertEqual(reviewed_rec["start_char"], 7)
        self.assertEqual(outputs["grades"][0]["status"], "accepted")
        self.assertEqual(outputs["picos"][0]["status"], "active")
        self.assertEqual(outputs["evidence"][0]["screening_status"], "included")

    def test_noisy_recommendation_is_left_for_review(self) -> None:
        rec = {
            "candidate_id": "rec_1",
            "guideline_id": "guideline_1",
            "recommendation_text": "Methods recommend careful grading.",
            "extraction_confidence": 0.9,
            "normalized_payload": {"quality_notes": ["methodology_statement"]},
            "raw_payload": {"quality": {"record_quality_flags": []}},
        }

        summary, outputs = backfill_inputs(BackfillInputs([rec], [], [], []))

        self.assertEqual(summary["accepted_recommendations"], 0)
        self.assertEqual(outputs["recommendations"][0].get("status"), None)
        self.assertIn("recommendation_has_noise_notes", summary["recommendation_reject_reason_counts"])

    def test_nearby_unlinked_context_can_be_repaired_before_acceptance(self) -> None:
        rec = {
            "candidate_id": "rec_1",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "recommendation_text": "We recommend antiviral treatment for adults with severe infection.",
            "source_text": "We recommend antiviral treatment for adults with severe infection.",
            "source_block_id": "block_rec",
            "source_order": 10,
            "direction": "for",
            "strength": "strong",
            "certainty": "moderate",
            "extraction_confidence": 0.9,
            "status": "pending",
            "source_section": "Recommendations Antiviral treatment",
            "normalized_payload": {"quality_notes": [], "source_order": 10},
            "raw_payload": {"quality": {"record_quality_flags": []}},
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
            "source_section": "Recommendations Antiviral treatment",
            "extraction_confidence": 0.9,
            "status": "pending",
            "normalized_payload": {"association_quality": "missing", "source_order": 11},
        }
        pico = {
            "pico_id": "pico_1",
            "guideline_id": "guideline_1",
            "source_record_id": "record_1",
            "population": "adults with severe infection",
            "intervention": "antiviral treatment",
            "status": "under_review",
            "source_order": 9,
            "source_section": "Recommendations Antiviral treatment",
        }
        evidence = {
            "evidence_id": "evidence_1",
            "source_record_id": "record_1",
            "guideline_id": "guideline_1",
            "recommendation_candidate_id": "",
            "pico_id": "",
            "study_design": "systematic_review",
            "effect_direction": "benefit",
            "source_text": "A systematic review found benefit from antiviral treatment in adults with severe infection.",
            "source_order": 12,
            "source_section": "Evidence Antiviral treatment",
            "screening_status": "association_review",
            "normalized_payload": {"has_structured_evidence_signal": True, "source_order": 12},
        }

        summary, outputs = backfill_inputs(BackfillInputs([rec], [grade], [pico], [evidence]))

        self.assertEqual(summary["accepted_recommendations"], 1)
        self.assertEqual(summary["repaired_grade_links"], 1)
        self.assertEqual(summary["repaired_evidence_links"], 1)
        self.assertEqual(outputs["recommendations"][0]["status"], "accepted")
        self.assertEqual(outputs["grades"][0]["recommendation_candidate_id"], "rec_1")
        self.assertEqual(outputs["grades"][0]["status"], "accepted")
        self.assertEqual(outputs["picos"][0]["status"], "active")
        self.assertEqual(outputs["evidence"][0]["recommendation_candidate_id"], "rec_1")
        self.assertEqual(outputs["evidence"][0]["pico_id"], "pico_1")
        self.assertEqual(outputs["evidence"][0]["screening_status"], "included")

    def test_direct_weak_grade_can_be_promoted_when_score_is_high(self) -> None:
        rec = {
            "candidate_id": "rec_weak_grade",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "recommendation_text": "We recommend antiviral treatment for adults with severe infection.",
            "source_text": "We recommend antiviral treatment for adults with severe infection.",
            "source_block_id": "block_rec",
            "source_order": 10,
            "direction": "for",
            "strength": "strong",
            "certainty": "moderate",
            "extraction_confidence": 0.9,
            "status": "pending",
            "source_section": "Recommendations Antiviral treatment",
            "normalized_payload": {"quality_notes": [], "source_order": 10},
            "raw_payload": {"quality": {"record_quality_flags": []}},
        }
        grade = {
            "grade_candidate_id": "grade_weak_direct",
            "recommendation_candidate_id": "rec_weak_grade",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "grade_system": "GRADE",
            "strength": "strong",
            "certainty": "moderate",
            "source_text": "Strong recommendation, moderate certainty evidence for antiviral treatment.",
            "source_section": "Recommendations Antiviral treatment",
            "extraction_confidence": 0.9,
            "status": "needs_review",
            "normalized_payload": {"association_quality": "weak", "source_order": 11},
        }
        pico = {
            "pico_id": "pico_1",
            "guideline_id": "guideline_1",
            "source_record_id": "record_1",
            "population": "adults with severe infection",
            "intervention": "antiviral treatment",
            "status": "active",
            "source_order": 9,
            "source_section": "Recommendations Antiviral treatment",
        }
        evidence = {
            "evidence_id": "evidence_1",
            "source_record_id": "record_1",
            "guideline_id": "guideline_1",
            "recommendation_candidate_id": "rec_weak_grade",
            "pico_id": "pico_1",
            "study_design": "systematic_review",
            "effect_direction": "benefit",
            "source_text": "A systematic review found benefit from antiviral treatment in adults with severe infection.",
            "source_order": 12,
            "source_section": "Evidence Antiviral treatment",
            "screening_status": "pending",
            "normalized_payload": {"has_structured_evidence_signal": True, "source_order": 12},
        }

        summary, outputs = backfill_inputs(BackfillInputs([rec], [grade], [pico], [evidence]))

        self.assertEqual(summary["accepted_recommendations"], 1)
        self.assertEqual(summary["association_repair_counts"]["grade_link_repaired"], 1)
        self.assertEqual(outputs["recommendations"][0]["status"], "accepted")
        self.assertEqual(outputs["grades"][0]["status"], "accepted")
        self.assertEqual(outputs["grades"][0]["normalized_payload"]["association_quality"], "strong")


if __name__ == "__main__":
    unittest.main()


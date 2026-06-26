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


if __name__ == "__main__":
    unittest.main()

import unittest

from src.pipeline.review.manual_review_gate import apply_reviews, build_queue, validate_review_items


class ManualReviewRoundTripTests(unittest.TestCase):
    def test_accepted_review_does_not_requeue_same_reason(self) -> None:
        version = {
            "recommendation_version_id": "version_1",
            "quality_status": "candidate_accepted",
            "normalized_payload": {"builder_notes": ["weak_pico_match"]},
        }
        queue, summary = build_queue([version], "recommendation_version")
        self.assertEqual(summary["queued_records"], 1)

        reviewed = dict(queue[0])
        reviewed.update(
            {
                "review_status": "reviewed",
                "review_decision": "accepted",
                "reviewer": "unit-test",
                "review_note": "Accepted.",
            }
        )
        output, apply_summary = apply_reviews([version], [reviewed], "recommendation_version")
        self.assertEqual(apply_summary["applied_reviews"], 1)

        second_queue, second_summary = build_queue(output, "recommendation_version")
        self.assertEqual(second_summary["queued_records"], 0)
        self.assertEqual(second_queue, [])

    def test_invalid_review_is_reported_and_not_applied(self) -> None:
        candidate = {"candidate_id": "rec_1", "status": "pending"}
        reviewed = {
            "review_id": "review_1",
            "entity_type": "recommendation_candidate",
            "entity_id": "rec_1",
            "review_status": "reviewed",
            "review_decision": "included",
            "reviewed_payload": {},
        }
        output, summary = apply_reviews([candidate], [reviewed], "recommendation_candidate")
        self.assertEqual(output[0]["status"], "pending")
        self.assertEqual(summary["applied_reviews"], 0)
        self.assertEqual(summary["review_validation"]["validation_error_count"], 1)

    def test_duplicate_review_entities_are_validation_errors(self) -> None:
        rows = [
            {
                "review_id": "review_1",
                "entity_type": "recommendation_version",
                "entity_id": "version_1",
                "review_status": "reviewed",
                "review_decision": "accepted",
                "reviewed_payload": {},
            },
            {
                "review_id": "review_2",
                "entity_type": "recommendation_version",
                "entity_id": "version_1",
                "review_status": "reviewed",
                "review_decision": "accepted",
                "reviewed_payload": {},
            },
        ]
        indexed, summary = validate_review_items(rows, "recommendation_version")
        self.assertEqual(list(indexed), ["version_1"])
        self.assertEqual(summary["validation_error_count"], 1)


if __name__ == "__main__":
    unittest.main()

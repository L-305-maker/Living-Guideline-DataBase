import unittest

from src.storage.generic_ingest import is_publishable_recommendation_version
from src.storage.repositories.recommendation_version_reviews import approve_review_item, reject_review_item


class RecommendationVersionReviewQueueTests(unittest.TestCase):
    def test_approved_review_item_promotes_version_payload(self) -> None:
        review_item = {
            "review_item_id": "review-version-1",
            "review_status": "pending",
            "version_payload": {
                "recommendation_version_id": "version-1",
                "quality_status": "blocked",
                "normalized_payload": {
                    "publish_gate": {
                        "quality_status": "blocked",
                        "publishable": False,
                        "blocking_reasons": ["missing_source_span_coordinates"],
                        "warning_reasons": ["no_linked_pico"],
                    }
                },
            },
        }

        reviewed, version = approve_review_item(review_item, reviewer="clinician", reason="Verified source manually.")

        self.assertEqual(reviewed["review_status"], "approved")
        self.assertEqual(version["quality_status"], "publishable")
        self.assertTrue(is_publishable_recommendation_version(version))
        gate = version["normalized_payload"]["publish_gate"]
        self.assertEqual(gate["blocking_reasons"], [])
        self.assertEqual(gate["warning_reasons"], [])
        self.assertEqual(gate["manual_override"]["reviewer"], "clinician")

    def test_rejected_review_item_does_not_promote_payload(self) -> None:
        review_item = {"review_item_id": "review-version-1", "review_status": "pending", "version_payload": {}}

        reviewed = reject_review_item(review_item, reviewer="clinician", reason="Unsupported recommendation.")

        self.assertEqual(reviewed["review_status"], "rejected")
        self.assertEqual(reviewed["review_decision"], "rejected")


if __name__ == "__main__":
    unittest.main()

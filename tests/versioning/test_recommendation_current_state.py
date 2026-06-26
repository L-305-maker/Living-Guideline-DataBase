import unittest

from src.pipeline.extraction.versioning.recommendation_state_builder import (
    build_recommendation_state,
    is_publishable_version,
)


def _publishable_version(version_id: str = "version-1") -> dict:
    return {
        "recommendation_version_id": version_id,
        "recommendation_id": "recommendation-1",
        "guideline_id": "guideline-1",
        "version_number": "v1",
        "recommendation_text": "We recommend treatment.",
        "quality_status": "publishable",
        "direction": "for",
        "strength": "strong",
        "certainty": "moderate",
        "created_at": "2026-01-01T00:00:00+00:00",
        "normalized_payload": {
            "publish_gate": {
                "quality_status": "publishable",
                "publishable": True,
                "blocking_reasons": [],
                "warning_reasons": [],
            }
        },
    }


class RecommendationCurrentStateTests(unittest.TestCase):
    def test_publishable_version_builds_current_state(self) -> None:
        state = build_recommendation_state(_publishable_version(), published_at="2026-01-02T00:00:00+00:00")

        self.assertEqual(state["recommendation_id"], "recommendation-1")
        self.assertEqual(state["current_version_id"], "version-1")
        self.assertEqual(state["status"], "active")
        self.assertEqual(state["last_published_at"], "2026-01-02T00:00:00+00:00")
        self.assertEqual(state["direction"], "for")
        self.assertEqual(state["strength"], "strong")
        self.assertEqual(state["certainty"], "moderate")

    def test_previous_recommendation_preserves_first_created_at(self) -> None:
        previous = {"first_created_at": "2025-01-01T00:00:00+00:00"}

        state = build_recommendation_state(
            _publishable_version("version-2"),
            previous_recommendation=previous,
            published_at="2026-01-02T00:00:00+00:00",
        )

        self.assertEqual(state["first_created_at"], "2025-01-01T00:00:00+00:00")
        self.assertEqual(state["current_version_id"], "version-2")

    def test_non_publishable_version_is_rejected(self) -> None:
        version = _publishable_version()
        version["quality_status"] = "needs_review"

        self.assertFalse(is_publishable_version(version))
        with self.assertRaisesRegex(ValueError, "not publishable"):
            build_recommendation_state(version)


if __name__ == "__main__":
    unittest.main()

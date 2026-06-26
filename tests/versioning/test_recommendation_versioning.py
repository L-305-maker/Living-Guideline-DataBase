import unittest

from src.pipeline.extraction.versioning.recommendation_version_builder import VersionBuildInput, build_version


class RecommendationVersioningTests(unittest.TestCase):
    def test_missing_source_span_coordinates_blocks_publish(self) -> None:
        rec = {
            "candidate_id": "candidate-1",
            "recommendation_id": "recommendation-1",
            "guideline_id": "guideline-1",
            "recommendation_text": "We recommend treatment.",
            "source_span": "We recommend treatment.",
            "status": "accepted",
        }

        version, report = build_version(VersionBuildInput(rec=rec, linked_grades=[]))

        self.assertEqual(version.quality_status, "blocked")
        self.assertFalse(report["publishable"])
        self.assertIn("missing_source_span_coordinates", report["blocking_reasons"])

    def test_missing_linked_context_requires_review(self) -> None:
        rec = {
            "candidate_id": "candidate-1",
            "recommendation_id": "recommendation-1",
            "guideline_id": "guideline-1",
            "recommendation_text": "We recommend treatment.",
            "source_span": "We recommend treatment.",
            "start_char": 0,
            "end_char": 23,
            "status": "accepted",
        }

        version, report = build_version(VersionBuildInput(rec=rec, linked_grades=[]))

        self.assertEqual(version.quality_status, "needs_review")
        self.assertFalse(report["publishable"])
        self.assertIn("no_accepted_grade", report["warning_reasons"])
        self.assertIn("no_linked_pico", report["warning_reasons"])
        self.assertIn("no_linked_evidence", report["warning_reasons"])

    def test_grade_conflict_blocks_publish(self) -> None:
        rec = {
            "candidate_id": "candidate-1",
            "recommendation_id": "recommendation-1",
            "guideline_id": "guideline-1",
            "recommendation_text": "We recommend treatment.",
            "source_span": "We recommend treatment.",
            "start_char": 0,
            "end_char": 23,
            "status": "accepted",
            "strength": "strong",
            "certainty": "high",
        }
        grade = {
            "grade_candidate_id": "grade-1",
            "recommendation_candidate_id": "candidate-1",
            "status": "accepted",
            "strength": "conditional",
            "certainty": "low",
        }

        version, report = build_version(VersionBuildInput(rec=rec, linked_grades=[grade]))

        self.assertEqual(version.quality_status, "blocked")
        self.assertIn("recommendation_grade_strength_or_certainty_conflict", report["blocking_reasons"])

    def test_previous_version_creates_incremented_immutable_version(self) -> None:
        rec = {
            "candidate_id": "candidate-2",
            "recommendation_id": "recommendation-1",
            "guideline_id": "guideline-1",
            "recommendation_text": "We recommend the updated treatment.",
            "source_text": "Source paragraph with updated treatment.",
            "source_block_id": "block-1",
            "start_char": 10,
            "end_char": 48,
            "direction": "for",
            "strength": "strong",
            "certainty": "moderate",
            "status": "accepted",
        }
        previous = {
            "recommendation_version_id": "version-1",
            "recommendation_id": "recommendation-1",
            "version_number": "v1",
            "recommendation_text": "We suggest the treatment.",
            "direction": "for",
            "strength": "conditional",
            "certainty": "low",
        }

        version, _ = build_version(
            VersionBuildInput(rec=rec, linked_grades=[], previous_version=previous)
        )

        self.assertEqual(version.version_number, "v2")
        self.assertEqual(version.previous_version_id, "version-1")
        self.assertEqual(version.change_type, "modified")
        self.assertIn("strength changed", version.change_summary or "")
        self.assertEqual(version.source_span, "Source paragraph with updated treatment.")
        self.assertEqual(version.source_span_ref, "block-1")
        self.assertEqual(version.start_char, 10)
        self.assertEqual(version.end_char, 48)

    def test_generic_recommendation_code_does_not_collapse_identity(self) -> None:
        base = {
            "guideline_id": "guideline-1",
            "recommendation_code": "for",
            "recommendation_text": "We suggest treatment.",
            "source_span": "We suggest treatment.",
            "start_char": 0,
            "end_char": 21,
            "status": "accepted",
        }
        first, _ = build_version(VersionBuildInput(rec={**base, "candidate_id": "candidate-1"}, linked_grades=[]))
        second, _ = build_version(VersionBuildInput(rec={**base, "candidate_id": "candidate-2"}, linked_grades=[]))

        self.assertNotEqual(first.recommendation_id, second.recommendation_id)
        self.assertNotEqual(first.recommendation_version_id, second.recommendation_version_id)


if __name__ == "__main__":
    unittest.main()

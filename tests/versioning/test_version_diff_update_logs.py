import unittest

from src.pipeline.update.version_diff import build_update_logs


class VersionDiffUpdateLogTests(unittest.TestCase):
    def test_build_update_logs_for_added_and_modified_versions(self) -> None:
        previous = [
            {
                "recommendation_version_id": "version-old",
                "recommendation_id": "rec-1",
                "guideline_id": "guideline-1",
                "version_number": "v1",
                "recommendation_text": "We suggest treatment.",
                "strength": "conditional",
                "certainty": "low",
            }
        ]
        current = [
            {
                "recommendation_version_id": "version-new",
                "recommendation_id": "rec-1",
                "guideline_id": "guideline-1",
                "version_number": "v2",
                "recommendation_text": "We recommend treatment.",
                "strength": "strong",
                "certainty": "moderate",
                "normalized_payload": {"linked_evidence": {"evidence_ids": ["evidence-1"]}},
            },
            {
                "recommendation_version_id": "version-added",
                "recommendation_id": "rec-2",
                "guideline_id": "guideline-1",
                "version_number": "v1",
                "recommendation_text": "We suggest CBT-I.",
            },
        ]

        logs, report = build_update_logs(previous, current)

        self.assertEqual(report["generated_update_logs"], 2)
        by_new_id = {log["new_recommendation_version_id"]: log for log in logs}
        self.assertEqual(by_new_id["version-new"]["update_type"], "text_modified")
        self.assertEqual(by_new_id["version-new"]["old_recommendation_version_id"], "version-old")
        self.assertEqual(by_new_id["version-new"]["triggering_evidence_ids"], ["evidence-1"])
        self.assertEqual(by_new_id["version-added"]["update_type"], "new_recommendation")

    def test_unchanged_version_does_not_generate_update_log(self) -> None:
        row = {
            "recommendation_version_id": "version-1",
            "recommendation_id": "rec-1",
            "guideline_id": "guideline-1",
            "version_number": "v1",
            "recommendation_text": "We suggest treatment.",
            "strength": "conditional",
            "certainty": "low",
        }

        logs, report = build_update_logs([row], [dict(row)])

        self.assertEqual(logs, [])
        self.assertEqual(report["generated_update_logs"], 0)
        self.assertEqual(report["update_type_counts"], {"unchanged": 1})


if __name__ == "__main__":
    unittest.main()

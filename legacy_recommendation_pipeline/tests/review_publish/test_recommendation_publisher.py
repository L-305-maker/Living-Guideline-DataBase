"""复核与发布测试文件：验证 review queue、backfill、release export 和正式发布闭包逻辑。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest
from unittest.mock import patch

from src.pipeline.publish.recommendation_publisher import (
    PublishContext,
    ingest_publish_bundle,
    publish_approved_review_item,
    publish_version,
)


def _publishable_version(version_id: str = "version-2") -> dict:
    return {
        "recommendation_version_id": version_id,
        "recommendation_id": "recommendation-1",
        "recommendation_candidate_id": "candidate-2",
        "guideline_id": "guideline-1",
        "version_number": "v2",
        "recommendation_text": "We recommend updated treatment.",
        "quality_status": "publishable",
        "direction": "for",
        "strength": "strong",
        "certainty": "moderate",
        "change_type": "modified",
        "normalized_payload": {
            "publish_gate": {
                "quality_status": "publishable",
                "publishable": True,
                "blocking_reasons": [],
                "warning_reasons": [],
            },
            "linked_evidence": {"linked_evidence_ids": ["evidence-1"]},
        },
    }


class RecommendationPublisherTests(unittest.TestCase):
    def test_publish_version_returns_atomic_bundle(self) -> None:
        previous = {
            "recommendation_version_id": "version-1",
            "recommendation_id": "recommendation-1",
            "guideline_id": "guideline-1",
            "version_number": "v1",
            "recommendation_text": "We suggest treatment.",
            "strength": "conditional",
            "certainty": "low",
        }

        bundle = publish_version(
            _publishable_version(),
            previous_versions=[previous],
            previous_recommendation={"first_created_at": "2025-01-01T00:00:00+00:00"},
            context=PublishContext(published_by="test", reason="New evidence", published_at="2026-01-02T00:00:00+00:00"),
        )

        self.assertEqual(len(bundle.recommendation_versions), 1)
        self.assertEqual(len(bundle.recommendations), 1)
        self.assertEqual(len(bundle.update_logs), 1)
        self.assertEqual(bundle.recommendations[0]["current_version_id"], "version-2")
        self.assertEqual(bundle.update_logs[0]["old_recommendation_version_id"], "version-1")
        self.assertEqual(bundle.update_logs[0]["triggering_evidence_ids"], ["evidence-1"])
        self.assertEqual(bundle.table_rows()["recommendations"][0]["recommendation_id"], "recommendation-1")

    def test_publish_approved_review_item_marks_review_published(self) -> None:
        review_item = {
            "review_item_id": "review-version-2",
            "review_status": "approved",
            "reviewer": "clinician",
            "review_reason": "Checked source span.",
            "reviewed_at": "2026-01-02T00:00:00+00:00",
            "version_payload": {
                **_publishable_version(),
                "quality_status": "blocked",
                "normalized_payload": {
                    "publish_gate": {
                        "quality_status": "blocked",
                        "publishable": False,
                        "blocking_reasons": ["missing_source_span_coordinates"],
                        "warning_reasons": [],
                    }
                },
            },
        }

        bundle = publish_approved_review_item(
            review_item,
            context=PublishContext(published_by="test", published_at="2026-01-03T00:00:00+00:00"),
        )

        self.assertEqual(bundle.recommendation_versions[0]["quality_status"], "publishable")
        self.assertEqual(bundle.recommendation_version_review_queue[0]["review_status"], "published")
        self.assertEqual(bundle.recommendation_version_review_queue[0]["published_version_id"], "version-2")
        self.assertEqual(bundle.summary["published_review_items"], 1)

    def test_ingest_publish_bundle_uses_atomic_storage_path(self) -> None:
        bundle = publish_version(
            _publishable_version(),
            context=PublishContext(published_by="test", published_at="2026-01-02T00:00:00+00:00"),
        )

        with patch("src.pipeline.publish.recommendation_publisher.ingest_tables_atomically", return_value={"recommendation_versions": 1}) as ingest:
            summary = ingest_publish_bundle(bundle)

        table_rows = ingest.call_args.args[0]
        self.assertIn("recommendation_versions", table_rows)
        self.assertIn("recommendations", table_rows)
        self.assertIn("update_logs", table_rows)
        self.assertEqual(summary["status"], "ingested")
        self.assertEqual(summary["table_counts"], {"recommendation_versions": 1})

    def test_non_publishable_version_cannot_publish_directly(self) -> None:
        version = _publishable_version()
        version["quality_status"] = "needs_review"

        with self.assertRaisesRegex(ValueError, "Cannot publish"):
            publish_version(version)

    def test_rejected_review_item_cannot_publish(self) -> None:
        review_item = {"review_item_id": "review-version-2", "review_status": "rejected", "version_payload": {}}

        with self.assertRaisesRegex(ValueError, "Only approved"):
            publish_approved_review_item(review_item)


if __name__ == "__main__":
    unittest.main()


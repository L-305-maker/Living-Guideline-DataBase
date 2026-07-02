"""向量化测试文件：验证 embedding queue 和后续 RAG 入队准备逻辑。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import read_jsonl, write_jsonl
from src.vectorization.embedding_queue import build_embedding_queue, iter_queue_items


class EmbeddingQueueTests(unittest.TestCase):
    def test_builds_publishable_recommendation_queue_and_skips_review_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "recommendation_versions.jsonl",
                [
                    {
                        "recommendation_version_id": "rv-1",
                        "recommendation_id": "rec-1",
                        "guideline_id": "g-1",
                        "version_number": "v1",
                        "recommendation_text": "We recommend treatment for adults with disease.",
                        "population": "adults with disease",
                        "intervention": "treatment",
                        "quality_status": "publishable",
                        "strength": "strong",
                    },
                    {
                        "recommendation_version_id": "rv-2",
                        "guideline_id": "g-1",
                        "version_number": "v1",
                        "recommendation_text": "Draft recommendation needing review.",
                        "quality_status": "needs_review",
                    },
                ],
            )

            manifest = build_embedding_queue(run_dir, run_dir / "embedding_queue.jsonl")
            rows = read_jsonl(run_dir / "embedding_queue.jsonl")

            self.assertEqual(manifest["written_rows"], 1)
            self.assertEqual(rows[0]["entity_type"], "recommendation_version")
            self.assertEqual(rows[0]["collection"], "lg_recommendations_published")
            self.assertEqual(rows[0]["metadata"]["publishable"], True)
            self.assertIn("Recommendation: We recommend treatment", rows[0]["text_for_embedding"])

    def test_review_versions_are_routed_to_review_collection_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "recommendation_versions.jsonl",
                [
                    {
                        "recommendation_version_id": "rv-review",
                        "guideline_id": "g-1",
                        "version_number": "v1",
                        "recommendation_text": "Draft recommendation needing review.",
                        "quality_status": "blocked",
                        "normalized_payload": {"publish_gate": {"publishable": False}},
                    }
                ],
            )

            rows = list(iter_queue_items(run_dir, include_review_versions=True))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["collection"], "lg_review_candidates")
            self.assertEqual(rows[0]["priority"], "review")
            self.assertEqual(rows[0]["metadata"]["publishable"], False)

    def test_source_blocks_keep_retrieval_blocks_and_skip_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "blocks.jsonl",
                [
                    {
                        "block_id": "block-rec",
                        "record_id": "record-1",
                        "guideline_id": "g-1",
                        "title": "Guideline",
                        "section_path": ["Guideline", "Recommendations"],
                        "block_type": "paragraph",
                        "text": "The panel recommends treatment for adults with disease based on moderate certainty evidence.",
                        "order": 1,
                    },
                    {
                        "block_id": "block-bg",
                        "record_id": "record-1",
                        "title": "Guideline",
                        "section_path": ["Guideline", "Background"],
                        "block_type": "paragraph",
                        "text": "This guideline describes administrative publication details and background context.",
                        "order": 2,
                    },
                ],
            )

            rows = list(iter_queue_items(run_dir, min_block_chars=20))

            self.assertEqual([row["entity_id"] for row in rows], ["block-rec"])
            self.assertEqual(rows[0]["metadata"]["block_scope"], "recommendation")
            self.assertEqual(rows[0]["collection"], "lg_source_blocks")

    def test_hash_is_stable_for_same_embedding_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_jsonl(
                run_dir / "pico_questions.jsonl",
                [
                    {
                        "pico_id": "pico-1",
                        "guideline_id": "g-1",
                        "clinical_question": "Should adults receive treatment?",
                        "population": "adults",
                        "intervention": "treatment",
                        "outcomes": [{"name": "mortality"}],
                    }
                ],
            )

            first = list(iter_queue_items(run_dir))[0]
            second = list(iter_queue_items(run_dir))[0]

            self.assertEqual(first["text_hash"], second["text_hash"])
            self.assertEqual(first["vector_id"], second["vector_id"])


if __name__ == "__main__":
    unittest.main()



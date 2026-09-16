from __future__ import annotations

import unittest
from unittest.mock import patch

from src.models import vllm_client
from src.retrieval.reranker import BgeM3ChunkReranker
from src.storage import query_embedding


class QwenModelSwitchTest(unittest.TestCase):
    def test_embedding_client_requests_fixed_dimension_and_normalizes(self) -> None:
        payloads: list[dict] = []

        def fake_post(_base_url, _path, payload, *, api_key_env):
            payloads.append(payload)
            self.assertEqual("VLLM_EMBEDDING_API_KEY", api_key_env)
            vector = [3.0, 4.0] + [0.0] * 1022
            return {"data": [{"index": 0, "embedding": vector}]}

        with patch.object(vllm_client, "helper_post_json", side_effect=fake_post):
            vectors = vllm_client.embed_texts(
                ["测试"], model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024
            )

        self.assertEqual(1024, payloads[0]["dimensions"])
        self.assertAlmostEqual(0.6, vectors[0][0])
        self.assertAlmostEqual(0.8, vectors[0][1])

    def test_query_vector_literal_routes_through_vllm(self) -> None:
        vector = [1.0] + [0.0] * 1023
        with patch("src.storage.query_embedding.embed_texts", return_value=[vector]) as embed:
            literal = query_embedding.query_vector_literal(
                "测试", model_name="Qwen/Qwen3-Embedding-8B"
            )

        self.assertTrue(literal.startswith("[1.00000000,"))
        embed.assert_called_once_with(
            ["测试"], model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024
        )

    def test_rerank_client_restores_original_document_order(self) -> None:
        response = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
        with patch.object(vllm_client, "helper_post_json", return_value=response) as post:
            scores = vllm_client.rerank_texts(
                "query", ["first", "second"], model_name="Qwen/Qwen3-Reranker-4B"
            )

        self.assertEqual([0.2, 0.9], scores)
        self.assertEqual(2, post.call_args.args[2]["top_n"])

    def test_embedding_client_rejects_wrong_dimension(self) -> None:
        response = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        with (
            patch.object(vllm_client, "helper_post_json", return_value=response),
            self.assertRaisesRegex(RuntimeError, "dimension mismatch"),
        ):
            vllm_client.embed_texts(
                ["query"], model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024
            )

    def test_chunk_reranker_uses_vllm_scores(self) -> None:
        candidates = [
            {"chunk_id": "a", "content": "first", "score": 0.2, "match_reason": {"base_rrf_score": 0.2}},
            {"chunk_id": "b", "content": "second", "score": 0.1, "match_reason": {"base_rrf_score": 0.1}},
        ]
        with patch("src.retrieval.reranker.rerank_texts", return_value=[0.1, 0.9]) as rerank:
            ranked = BgeM3ChunkReranker().rerank("query", candidates, topk=2)

        self.assertEqual("b", ranked[0]["chunk_id"])
        self.assertEqual("vllm_chunk_cross_encoder", ranked[0]["match_reason"]["reranker"])
        self.assertEqual(2, len(rerank.call_args.args[1]))

if __name__ == "__main__":
    unittest.main()

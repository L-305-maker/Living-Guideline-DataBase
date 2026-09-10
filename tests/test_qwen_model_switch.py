from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.reranker import helper_load_cross_encoder
from src.storage import query_embedding


class FakeEncodeModel:
    def __init__(self) -> None:
        self.called_kwargs: dict = {}

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.called_kwargs = dict(kwargs)
        return [[0.0] * 1024 for _ in texts]


class FakeOwner:
    def __init__(self) -> None:
        self.model_name = "Qwen/Qwen3-Reranker-4B"
        self.device = None
        self.local_files_only = True
        self.max_length = 2048
        self._model = None
        self._load_error = None


class QwenModelSwitchTest(unittest.TestCase):
    def test_encode_with_model_truncates_qwen_to_database_dimension(self) -> None:
        model = FakeEncodeModel()
        query_embedding.encode_with_model(model, ["x"], model_name="Qwen/Qwen3-Embedding-8B")

        self.assertEqual(query_embedding.EMBEDDING_DIM, model.called_kwargs["truncate_dim"])
        self.assertTrue(model.called_kwargs["normalize_embeddings"])

    def test_encode_with_model_skips_truncation_for_non_mrl(self) -> None:
        model = FakeEncodeModel()
        query_embedding.encode_with_model(model, ["x"], model_name="BAAI/bge-m3")

        self.assertNotIn("truncate_dim", model.called_kwargs)
        self.assertTrue(model.called_kwargs["normalize_embeddings"])

    def test_legacy_dimension_override_cannot_diverge_from_database_schema(self) -> None:
        model = FakeEncodeModel()
        with patch.dict(os.environ, {"PG_VECTOR_MATRYOSHKA_DIM": "512"}):
            query_embedding.encode_with_model(model, ["x"], model_name="Qwen/Qwen3-Embedding-8B")

        self.assertEqual(query_embedding.EMBEDDING_DIM, model.called_kwargs["truncate_dim"])

    def test_query_vector_literal_routes_through_encode_with_model(self) -> None:
        model = FakeEncodeModel()
        with (
            patch("src.storage.query_embedding.load_model", return_value=model),
            patch.dict(os.environ, {}, clear=False),
        ):
            literal = query_embedding.query_vector_literal("测试", model_name="Qwen/Qwen3-Embedding-8B")

        self.assertTrue(literal.startswith("["))
        self.assertEqual(query_embedding.EMBEDDING_DIM, model.called_kwargs["truncate_dim"])

    def test_cross_encoder_receives_model_kwargs_from_env(self) -> None:
        owner = FakeOwner()
        fake_cross_encoder = MagicMock(return_value=object())
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.CrossEncoder = fake_cross_encoder
        with (
            patch.dict(os.environ, {"BGE_RERANKER_DTYPE": "bfloat16", "BGE_RERANKER_ATTN": "flash_attention_2"}),
            patch.dict(sys.modules, {"sentence_transformers": fake_st}),
        ):
            loaded = helper_load_cross_encoder(owner)

        self.assertIsNotNone(loaded)
        _, kwargs = fake_cross_encoder.call_args
        self.assertEqual("bfloat16", kwargs["model_kwargs"]["torch_dtype"])
        self.assertEqual("flash_attention_2", kwargs["model_kwargs"]["attn_implementation"])

    def test_cross_encoder_without_model_kwargs_env(self) -> None:
        owner = FakeOwner()
        fake_cross_encoder = MagicMock(return_value=object())
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.CrossEncoder = fake_cross_encoder
        with (
            patch.dict(os.environ, {"BGE_RERANKER_DTYPE": "", "BGE_RERANKER_ATTN": ""}),
            patch.dict(sys.modules, {"sentence_transformers": fake_st}),
        ):
            helper_load_cross_encoder(owner)

        _, kwargs = fake_cross_encoder.call_args
        self.assertIsNone(kwargs["model_kwargs"])


    def test_load_model_passes_dtype_attn_and_padding_kwargs(self) -> None:
        fake_st = types.ModuleType("sentence_transformers")
        fake_encoder = MagicMock()
        fake_st.SentenceTransformer = fake_encoder
        query_embedding.load_model.cache_clear()
        with (
            patch.dict(os.environ, {"PG_VECTOR_MODEL_DTYPE": "bfloat16", "PG_VECTOR_MODEL_ATTN": "flash_attention_2"}),
            patch.dict(sys.modules, {"sentence_transformers": fake_st}),
        ):
            query_embedding.load_model("Qwen/Qwen3-Embedding-8B")

        _, kwargs = fake_encoder.call_args
        self.assertEqual("bfloat16", kwargs["model_kwargs"]["torch_dtype"])
        self.assertEqual("flash_attention_2", kwargs["model_kwargs"]["attn_implementation"])
        self.assertEqual("left", kwargs["tokenizer_kwargs"]["padding_side"])

if __name__ == "__main__":
    unittest.main()

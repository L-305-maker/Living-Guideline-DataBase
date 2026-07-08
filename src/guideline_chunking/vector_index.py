"""Placeholder for future vector retrieval integration."""

from __future__ import annotations


class VectorIndex:
    """Interface stub; first version intentionally ships BM25 only."""

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        raise NotImplementedError("Vector retrieval is a P2 extension.")

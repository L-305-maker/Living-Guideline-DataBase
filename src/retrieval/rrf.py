"""Reciprocal Rank Fusion."""

from __future__ import annotations


def rrf_fusion(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def weighted_rrf_fusion(rank_lists: list[tuple[list[str], float]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion with per-channel weights."""

    scores: dict[str, float] = {}
    for rank_list, weight in rank_lists:
        if weight <= 0:
            continue
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

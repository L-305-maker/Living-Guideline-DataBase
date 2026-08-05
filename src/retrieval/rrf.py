# Reciprocal Rank Fusion（倒数排名融合）。
#
# 核心思想：把多个召回通道（BM25 / dense）的 rank 名次融合为统一分数，
# 避免不同通道的原始分数（余弦距离 vs ts_rank）量纲不一致的问题。
#
# 公式：score(id) = Σ_channel 1 / (k + rank)
# - k=60 是经典常数，平滑高分项的极端权重；
# - 同分时按 item_id 字典序稳定排序，结果可复现；
# - weighted_rrf_fusion 支持每通道不同权重（适用于人工调优）。
"""Reciprocal Rank Fusion."""

from __future__ import annotations


def rrf_fusion(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """标准 RRF：多通道名次求和，按分数降序 + id 升序返回。

    行为：
    - 每个通道的名次从 1 开始；
    - 同一 item 出现在多通道时，分数累加；
    - 输出按 (-score, item_id) 排序，分数相同时 id 较小的优先。
    """
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def weighted_rrf_fusion(rank_lists: list[tuple[list[str], float]], k: int = 60) -> list[tuple[str, float]]:
    """加权 RRF：每通道可指定不同权重。

    行为：
    - 输入 (rank_list, weight) 元组列表；
    - weight ≤ 0 的通道整体跳过（等价于禁用该通道）；
    - 同分时按 item_id 字典序稳定排序。
    """
    scores: dict[str, float] = {}
    for rank_list, weight in rank_lists:
        if weight <= 0:
            continue
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
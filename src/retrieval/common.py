"""检索模块共享的无状态辅助函数。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.pipeline.cleaning.semantic_chunker import estimate_tokens, retrieval_text
from src.retrieval.document_repr.section_classifier import classify_chunk


TIME_RANGE_RE = re.compile(
    r"^\s*(\d{4})(?:-\d{2}-\d{2})?\s*[-~:]\s*(\d{4})(?:-\d{2}-\d{2})?\s*$"
)


def parse_time_range(
    time_range: str | dict[str, str] | None,
) -> tuple[str | None, str | None]:
    """把查询时间范围统一转换为可比较的起止日期。"""

    if not time_range:
        return None, None
    if isinstance(time_range, dict):
        return (
            time_range.get("start") or time_range.get("start_date"),
            time_range.get("end") or time_range.get("end_date"),
        )
    match = TIME_RANGE_RE.match(str(time_range))
    if match:
        return f"{match.group(1)}-01-01", f"{match.group(2)}-12-31"
    return str(time_range), None


def enrich_chunk_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """补齐旧版 chunk 中缺失的分类、token 数和检索文本。"""

    chunk_type = str(item.get("chunk_type") or "") or classify_chunk(item)
    item["chunk_type"] = chunk_type
    item["token_count"] = int(
        item.get("token_count") or estimate_tokens(item.get("content", ""))
    )
    item["retrieval_text"] = item.get("retrieval_text") or retrieval_text(
        item.get("title", ""),
        item.get("section_path") or [],
        chunk_type,
        item.get("content", ""),
    )
    item["is_background"] = bool(
        item.get("is_background") or chunk_type == "background"
    )
    return item


def clip_text(text: str, limit: int) -> str:
    """压缩连续空白并按字符数截断展示文本。"""

    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def source_quote_context(
    prev_content: str, content: str, next_content: str
) -> str:
    """为命中块拼接有限长度的前后文，避免返回整篇文档。"""

    parts = []
    if prev_content:
        parts.append("[previous] " + clip_text(prev_content[-500:], 500))
    parts.append("[current] " + clip_text(content, 1400))
    if next_content:
        parts.append("[next] " + clip_text(next_content[:500], 500))
    return "\n".join(parts)


QUERY_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]+"
)


def query_terms(query: str) -> list[str]:
    """按首次出现顺序提取去重的中英文查询词。"""

    seen: set[str] = set()
    terms: list[str] = []
    for token in QUERY_TOKEN_RE.findall(query or ""):
        normalized = token.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


def compact_text(text: str) -> str:
    """移除空白并统一大小写，用于跨排版形式比较文本。"""

    return re.sub(r"\s+", "", text or "").lower()


def contains_exact_phrase(text: str, query: str) -> bool:
    """同时检查原始排版和去空白排版中的完整查询短语。"""

    normalized_query = (query or "").strip().lower()
    if not normalized_query:
        return False
    if normalized_query in (text or "").lower():
        return True
    compact_query = compact_text(normalized_query)
    return bool(compact_query and compact_query in compact_text(text))


def fill_consensus_fallback(
    guidelines: list[dict[str, Any]],
    wanted: int,
    load_consensus: Callable[[int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return guideline results first, then consensus records only to fill a deficit."""

    output = [{**item, "document_kind": "guideline", "is_fallback": False} for item in guidelines]
    deficit = wanted - len(output)
    if deficit > 0:
        output.extend(
            {
                **item,
                "document_kind": "consensus",
                "is_fallback": True,
                "fallback_reason": "insufficient_guideline_results",
                "fallback_rank": rank,
            }
            for rank, item in enumerate(load_consensus(deficit), 1)
        )
    return output[:wanted]

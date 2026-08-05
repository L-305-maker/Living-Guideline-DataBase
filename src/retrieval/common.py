# 检索模块共享的无状态辅助函数。
#
# 模块职责：
# - 时间范围解析（'YYYY-YYYY' / dict / 自由字符串）；
# - chunk 元数据补齐（chunk_type / token_count / retrieval_text / is_background）；
# - 文本截断与原文片段拼接；
# - 查询词提取与精确短语匹配（含跨空白形式）；
# - consensus 兜底：先用 guideline 集合，缺额时按缺口量补充 consensus。
"""检索模块共享的无状态辅助函数。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.pipeline.cleaning.semantic_chunker import estimate_tokens, retrieval_text
from src.retrieval.document_repr.section_classifier import classify_chunk


# 时间范围正则：'YYYY'（可选月日）-[~-:] 'YYYY'（可选月日）。
TIME_RANGE_RE = re.compile(
    r"^\s*(\d{4})(?:-\d{2}-\d{2})?\s*[-~:]\s*(\d{4})(?:-\d{2}-\d{2})?\s*$"
)


def parse_time_range(
    time_range: str | dict[str, str] | None,
) -> tuple[str | None, str | None]:
    """把查询时间范围统一转换为可比较的起止日期。

    支持三种输入：
    - None / 空串：返回 (None, None)
    - dict：取 start / start_date 与 end / end_date
    - 'YYYY-YYYY' / 'YYYY:YYYY' / 'YYYY~YYYY'：自动补齐为 '-01-01' / '-12-31'
    - 其它字符串：单边 start（end=None）
    """
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
    """补齐旧版 chunk 中缺失的分类、token 数和检索文本。

    字段来源优先级：
    - chunk_type：旧记录 → 用 classify_chunk 重新分类；
    - token_count：旧记录 → 用 estimate_tokens 估算；
    - retrieval_text：缺失时按 title/section_path/chunk_type/content 重新构造；
    - is_background：缺失时由 chunk_type=='background' 推导。
    返回入参 item 同一对象（in-place 修改）。
    """
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
    """压缩连续空白并按字符数截断展示文本。

    limit ≤ 0 时返回空串；超过 limit 时截断并加 '...' 后缀。
    """
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def source_quote_context(
    prev_content: str, content: str, next_content: str
) -> str:
    """为命中块拼接有限长度的前后文，避免返回整篇文档。

    输出格式：[previous] ...\\n[current] ...\\n[next] ...
    每段都经过 clip_text 限制长度（prev 取末尾 500 / current 1400 / next 开头 500）。
    """
    parts = []
    if prev_content:
        parts.append("[previous] " + clip_text(prev_content[-500:], 500))
    parts.append("[current] " + clip_text(content, 1400))
    if next_content:
        parts.append("[next] " + clip_text(next_content[:500], 500))
    return "\n".join(parts)


# 查询词提取正则：与 text.SEARCH_TOKEN_RE 类似但保留连续的 CJK 字符（便于中文短语）。
QUERY_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]+"
)


def query_terms(query: str) -> list[str]:
    """按首次出现顺序提取去重的中英文查询词。

    注意：去重前先 lowercase + strip；CJK 段保留原顺序，便于精确短语匹配。
    """
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
    """同时检查原始排版和去空白排版中的完整查询短语。

    先按原始大小写匹配；失败时把 text 和 query 都压平去空白再匹配，
    用于容忍原文换行/缩进造成的跨行短语（如「急性心肌梗死」跨段落）。
    """
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
    """Return guideline results first, then consensus records only to fill a deficit.

    行为：
    - 现有 guidelines 全量保留并打 document_kind=guideline / is_fallback=False；
    - 缺额 = wanted - len(guidelines)，>0 时调用 load_consensus(deficit) 拉取；
    - 拉回的 consensus 打 document_kind=consensus / is_fallback=True / fallback_rank；
    - 最终切片到 wanted，避免超过检索需求。
    """
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
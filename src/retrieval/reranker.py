"""Document reranker interfaces for document-level search."""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any, Protocol
from src.retrieval.common import (
    clip_text as helper_clip_text,
    compact_text as helper_compact,
    contains_exact_phrase as helper_contains_exact_phrase,
    query_terms as helper_query_terms,
)
from src.utils.records import (
    publication_year as helper_publication_year,
    truthy as helper_truthy,
)


DEFAULT_BGE_RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"
GUIDE_RE = re.compile(r"(guideline|guidelines|consensus|recommendations?|\u6307\u5357|\u5171\u8bc6)", re.I)
OCR_UNRESOLVED_STATUSES = {"needed_unavailable", "needed_not_applied", "needed_but_disabled", "failed"}
OCR_REVIEW_STATUSES = {"applied_needs_review"}
HIGH_RISK_CLEANING_FLAGS = {"likely_ocr_failure", "low_text_signal", "noisy_ocr_lines", "pdf_text_mojibake"}


class DocumentReranker(Protocol):
    def rerank(self, query: str, candidates: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
        """Reorder already-recalled document candidates."""


class ChunkReranker(Protocol):
    def rerank(self, query: str, candidates: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
        """Reorder already-recalled chunk candidates."""


class NoopReranker:
    def rerank(self, query: str, candidates: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
        # 无操作实现保留统一接口，但明确忽略查询内容，只执行数量裁剪。
        _ = query
        return candidates[:topk]


class NoopDocumentReranker(NoopReranker):
    pass


class NoopChunkReranker(NoopReranker):
    pass


class RuleBasedDocumentReranker:
    """Small deterministic reranker used until a model reranker is wired in."""

    def __init__(self, recency_boost: bool = False) -> None:
        self.recency_boost = recency_boost

    def rerank(self, query: str, candidates: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
        # 文档重排同时考虑匹配信号与质量惩罚，最后按稳定标识处理同分候选。
        terms = helper_query_terms(query)
        ranked = []
        for candidate in candidates:
            item = dict(candidate)
            boosts = dict(item.get("match_reason", {}).get("boosts", {}))
            matched_fields = set(item.get("match_reason", {}).get("matched_fields", []))
            title = item.get("title", "")
            card_text = item.get("document_card", {}).get("card_text", "")
            view_text = "\n".join(str(view.get("text", "")) for view in item.get("matched_views", []))
            chunk_text = "\n".join(str(chunk.get("content", "")) for chunk in item.get("matched_chunks", []))

            if helper_contains_term(title, terms):
                boosts["title_term"] = 0.20
                matched_fields.add("title")
            if helper_contains_exact_phrase(title, query):
                boosts["title_exact_phrase"] = 0.35
                matched_fields.add("title")
            if helper_contains_term(card_text, terms):
                boosts["document_card_term"] = 0.12
                matched_fields.add("document_card")
            if helper_contains_term(view_text, terms):
                boosts["document_view_term"] = 0.12
                matched_fields.add("document_views")
            if helper_contains_term(chunk_text, terms):
                boosts["chunk_term"] = 0.10
                matched_fields.add("chunks")
            channel_count = len(item.get("retrieval_scores", {}))
            if channel_count >= 3:
                boosts["multi_channel_agreement"] = 0.12
            if GUIDE_RE.search(title or ""):
                boosts["guideline_title"] = 0.08
            if self.recency_boost:
                boost = helper_publication_recency_boost(item.get("publication_date"))
                if boost:
                    boosts["publication_date_recency"] = boost

            base_score = float(item.get("score", 0.0))
            quality_multiplier, quality_penalties = quality_score_multiplier(item)
            item["score"] = base_score * (1.0 + sum(boosts.values())) * quality_multiplier
            reason = dict(item.get("match_reason", {}))
            reason["boosts"] = boosts
            if quality_penalties:
                reason["quality_penalties"] = quality_penalties
                reason["quality_multiplier"] = quality_multiplier
            reason["matched_fields"] = sorted(matched_fields)
            reason["base_rrf_score"] = reason.get("base_rrf_score", base_score)
            item["match_reason"] = reason
            ranked.append(item)

        ranked.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -(helper_publication_year(item.get("publication_date")) or 0),
                item.get("doc_id", ""),
            )
        )
        return ranked[:topk]


class RuleBasedChunkReranker:
    """Deterministic chunk reranker used as a fast fallback for retrieve."""

    def rerank(self, query: str, candidates: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
        # 分块重排优先完整短语和关键字段命中，并对参考文献、OCR 风险等低质量信号降权。
        terms = helper_query_terms(query)
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            content = str(item.get("content") or "")
            section_text = " > ".join(str(part) for part in item.get("section_path") or [])
            title = str(item.get("title") or "")
            source_context = str(item.get("source_quote_context") or "")
            searchable_text = "\n".join([title, section_text, content, source_context])
            boosts: dict[str, float] = {}
            matched_fields = set(item.get("match_reason", {}).get("matched_fields", []))

            if helper_contains_exact_phrase(content, query):
                boosts["content_exact_phrase"] = 0.36
                matched_fields.add("content")
            if helper_contains_term(content, terms):
                overlap = helper_term_overlap_ratio(content, terms)
                boosts["content_term_overlap"] = min(0.26, 0.08 + overlap * 0.22)
                matched_fields.add("content")
            if helper_contains_term(section_text, terms):
                boosts["section_path_term"] = 0.12
                matched_fields.add("section_path")
            if helper_contains_term(title, terms):
                boosts["title_term"] = 0.08
                matched_fields.add("title")
            if source_context and helper_contains_term(source_context, terms):
                boosts["neighbor_context_term"] = 0.04
                matched_fields.add("source_quote_context")

            chunk_type = str(item.get("chunk_type") or "")
            type_boost = helper_chunk_type_boost(chunk_type)
            if type_boost:
                boosts[f"chunk_type_{chunk_type}"] = type_boost

            penalty = 0.0
            if item.get("is_reference_section"):
                penalty += 0.75
            if helper_looks_like_reference_section(section_text):
                penalty += 0.35

            base_score = float(item.get("score", 0.0))
            item["score"] = base_score * (1.0 + sum(boosts.values())) * max(0.05, 1.0 - min(0.9, penalty))
            reason = dict(item.get("match_reason", {}))
            reason["reranker"] = "rule_based_chunk"
            reason["matched_fields"] = sorted(matched_fields)
            reason["boosts"] = boosts
            reason["base_rrf_score"] = reason.get("base_rrf_score", base_score)
            if penalty:
                reason["penalty"] = round(penalty, 4)
            item["match_reason"] = reason
            item["rerank_text_preview"] = helper_clip_text(searchable_text, 500)
            ranked.append(item)

        ranked.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -int(item.get("token_count") or 0),
                item.get("chunk_id", ""),
            )
        )
        return ranked[:topk]


def helper_load_cross_encoder(owner: Any) -> Any:
    """按实例配置延迟加载 CrossEncoder，并缓存成功模型或失败原因。"""

    if owner._model is not None:
        return owner._model
    if owner._load_error:
        raise RuntimeError(owner._load_error)
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        # Qwen3-Reranker 等大模型可通过环境变量指定加载精度/attention 实现，避免默认 fp32 爆显存。
        model_kwargs: dict[str, Any] = {}
        dtype = os.getenv("BGE_RERANKER_DTYPE", "").strip()
        if dtype:
            model_kwargs["torch_dtype"] = dtype
        attn = os.getenv("BGE_RERANKER_ATTN", "").strip()
        if attn:
            model_kwargs["attn_implementation"] = attn
        owner._model = CrossEncoder(
            owner.model_name,
            device=owner.device,
            local_files_only=owner.local_files_only,
            max_length=owner.max_length,
            trust_remote_code=True,
            model_kwargs=model_kwargs or None,
        )
        return owner._model
    except Exception as exc:  # pragma: no cover - 依赖本地模型缓存
        owner._load_error = str(exc)
        raise RuntimeError(owner._load_error) from exc


class BgeM3DocumentReranker:
    """Embedding-first cross-encoder reranker backed by BGE reranker v2 m3."""

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_RERANKER_MODEL,
        recency_boost: bool = False,
        batch_size: int = 16,
        max_length: int = 1024,
        device: str | None = None,
        local_files_only: bool = True,
        model_weight: float = 0.86,
        base_weight: float = 0.14,
        fallback: DocumentReranker | None = None,
    ) -> None:
        self.model_name = model_name
        self.recency_boost = recency_boost
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.local_files_only = local_files_only
        self.model_weight = model_weight
        self.base_weight = base_weight
        self.fallback = fallback or RuleBasedDocumentReranker(recency_boost=recency_boost)
        self._model: Any | None = None
        self._load_error: str | None = None


    def rerank(self, query: str, candidates: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
        if not candidates:
            return []

        # 先执行规则排序，既提供模型不可用时的完整回退，也保留后续融合所需的基础分。
        fallback_ranked = self.fallback.rerank(query, candidates, len(candidates))
        try:
            model = helper_load_cross_encoder(self)
            pairs = [(query, helper_candidate_rerank_text(candidate, query)) for candidate in fallback_ranked]
            raw_scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception:
            return self.helper_mark_fallback(fallback_ranked[:topk])

        raw_values = [float(score) for score in raw_scores]
        # CrossEncoder 原始分数与 RRF 分数范围不同，先分别归一化再做加权融合。
        model_scores = helper_normalize_scores(raw_values)
        base_scores = helper_normalize_scores([float(item.get("match_reason", {}).get("base_rrf_score", item.get("score", 0.0))) for item in fallback_ranked])
        output: list[dict[str, Any]] = []
        for item, raw_score, model_score, base_score in zip(fallback_ranked, raw_values, model_scores, base_scores):
            ranked = dict(item)
            # 清洗质量只作为最终乘数，避免低质量 OCR 文档仅凭语义相似度占据首位。
            quality_multiplier, quality_penalties = quality_score_multiplier(ranked)
            combined = (self.model_weight * model_score) + (self.base_weight * base_score)
            ranked["score"] = combined * quality_multiplier
            reason = dict(ranked.get("match_reason", {}))
            reason["reranker"] = "bge_m3_cross_encoder"
            reason["reranker_model"] = self.model_name
            reason["reranker_raw_score"] = raw_score
            reason["reranker_normalized_score"] = model_score
            reason["base_normalized_score"] = base_score
            reason["quality_multiplier"] = quality_multiplier
            if quality_penalties:
                reason["quality_penalties"] = quality_penalties
            ranked["match_reason"] = reason
            output.append(ranked)

        output.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -(helper_publication_year(item.get("publication_date")) or 0),
                item.get("doc_id", ""),
            )
        )
        return output[:topk]

    def helper_mark_fallback(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for item in items:
            updated = dict(item)
            reason = dict(updated.get("match_reason", {}))
            reason["reranker"] = "rule_based_fallback"
            reason["requested_reranker_model"] = self.model_name
            if self._load_error:
                reason["reranker_error"] = self._load_error[:500]
            updated["match_reason"] = reason
            output.append(updated)
        return output


class BgeM3ChunkReranker:
    """Embedding-first cross-encoder reranker for chunk-level retrieve."""

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_RERANKER_MODEL,
        batch_size: int = 16,
        max_length: int = 1024,
        device: str | None = None,
        local_files_only: bool = True,
        model_weight: float = 0.84,
        base_weight: float = 0.10,
        rule_weight: float = 0.06,
        fallback: ChunkReranker | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.local_files_only = local_files_only
        self.model_weight = model_weight
        self.base_weight = base_weight
        self.rule_weight = rule_weight
        self.fallback = fallback or RuleBasedChunkReranker()
        self._model: Any | None = None
        self._load_error: str | None = None


    def rerank(self, query: str, candidates: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
        if not candidates:
            return []

        # 分块重排也先保留规则结果；模型加载或推理失败时可返回可解释的降级排序。
        fallback_ranked = self.fallback.rerank(query, candidates, len(candidates))
        try:
            model = helper_load_cross_encoder(self)
            pairs = [(query, helper_chunk_rerank_text(candidate)) for candidate in fallback_ranked]
            raw_scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception:
            return self.helper_mark_fallback(fallback_ranked[:topk])

        raw_values = [float(score) for score in raw_scores]
        # CrossEncoder 原始分数与 RRF 分数范围不同，先分别归一化再做加权融合。
        model_scores = helper_normalize_scores(raw_values)
        base_scores = helper_normalize_scores(
            [float(item.get("match_reason", {}).get("base_rrf_score", item.get("score", 0.0))) for item in fallback_ranked]
        )
        # 规则分与基础 RRF 分承担不同职责，分别归一化后保留为两个独立融合信号。
        rule_scores = helper_normalize_scores([float(item.get("score", 0.0)) for item in fallback_ranked])
        output: list[dict[str, Any]] = []
        for item, raw_score, model_score, base_score, rule_score in zip(
            fallback_ranked, raw_values, model_scores, base_scores, rule_scores
        ):
            ranked = dict(item)
            ranked["score"] = (
                self.model_weight * model_score
                + self.base_weight * base_score
                + self.rule_weight * rule_score
            )
            reason = dict(ranked.get("match_reason", {}))
            reason["reranker"] = "bge_m3_chunk_cross_encoder"
            reason["reranker_model"] = self.model_name
            reason["reranker_raw_score"] = raw_score
            reason["reranker_normalized_score"] = model_score
            reason["base_normalized_score"] = base_score
            reason["rule_normalized_score"] = rule_score
            ranked["match_reason"] = reason
            output.append(ranked)

        output.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -int(item.get("token_count") or 0),
                item.get("chunk_id", ""),
            )
        )
        return output[:topk]

    def helper_mark_fallback(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for item in items:
            updated = dict(item)
            reason = dict(updated.get("match_reason", {}))
            reason["reranker"] = "rule_based_chunk_fallback"
            reason["requested_reranker_model"] = self.model_name
            if self._load_error:
                reason["reranker_error"] = self._load_error[:500]
            updated["match_reason"] = reason
            output.append(updated)
        return output


def default_document_reranker(recency_boost: bool = False) -> DocumentReranker:
    mode = os.getenv("DOCUMENT_RERANKER", "bge_m3").strip().lower()
    if mode in {"none", "noop", "off"}:
        return helper_cached_noop_document_reranker()
    if mode in {"rule", "rules", "rule_based"}:
        return helper_cached_rule_document_reranker(recency_boost)
    local_only = os.getenv("BGE_RERANKER_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
    batch_size = int(os.getenv("BGE_RERANKER_BATCH_SIZE", "8"))
    max_length = int(os.getenv("BGE_RERANKER_MAX_LENGTH", "1024"))
    return helper_cached_bge_document_reranker(
        os.getenv("BGE_RERANKER_MODEL", DEFAULT_BGE_RERANKER_MODEL),
        recency_boost,
        batch_size,
        max_length,
        os.getenv("BGE_RERANKER_DEVICE") or "",
        local_only,
        helper_env_float("BGE_DOCUMENT_RERANKER_MODEL_WEIGHT", 0.86),
        helper_env_float("BGE_DOCUMENT_RERANKER_BASE_WEIGHT", 0.14),
    )


def default_chunk_reranker() -> ChunkReranker:
    mode = os.getenv("CHUNK_RERANKER", os.getenv("RERANKER", "bge_m3")).strip().lower()
    if mode in {"none", "noop", "off"}:
        return helper_cached_noop_chunk_reranker()
    if mode in {"rule", "rules", "rule_based"}:
        return helper_cached_rule_chunk_reranker()
    local_only = os.getenv("BGE_RERANKER_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
    batch_size = int(os.getenv("BGE_RERANKER_BATCH_SIZE", "8"))
    max_length = int(os.getenv("BGE_RERANKER_MAX_LENGTH", "1024"))
    return helper_cached_bge_chunk_reranker(
        os.getenv("BGE_RERANKER_MODEL", DEFAULT_BGE_RERANKER_MODEL),
        batch_size,
        max_length,
        os.getenv("BGE_RERANKER_DEVICE") or "",
        local_only,
        helper_env_float("BGE_CHUNK_RERANKER_MODEL_WEIGHT", 0.84),
        helper_env_float("BGE_CHUNK_RERANKER_BASE_WEIGHT", 0.10),
        helper_env_float("BGE_CHUNK_RERANKER_RULE_WEIGHT", 0.06),
    )


@lru_cache(maxsize=2)
def helper_cached_noop_document_reranker() -> NoopDocumentReranker:
    return NoopDocumentReranker()


@lru_cache(maxsize=4)
def helper_cached_rule_document_reranker(recency_boost: bool) -> RuleBasedDocumentReranker:
    return RuleBasedDocumentReranker(recency_boost=recency_boost)


@lru_cache(maxsize=8)
def helper_cached_bge_document_reranker(
    model_name: str,
    recency_boost: bool,
    batch_size: int,
    max_length: int,
    device: str,
    local_files_only: bool,
    model_weight: float,
    base_weight: float,
) -> BgeM3DocumentReranker:
    return BgeM3DocumentReranker(
        model_name=model_name,
        recency_boost=recency_boost,
        batch_size=batch_size,
        max_length=max_length,
        device=device or None,
        local_files_only=local_files_only,
        model_weight=model_weight,
        base_weight=base_weight,
    )


@lru_cache(maxsize=2)
def helper_cached_noop_chunk_reranker() -> NoopChunkReranker:
    return NoopChunkReranker()


@lru_cache(maxsize=2)
def helper_cached_rule_chunk_reranker() -> RuleBasedChunkReranker:
    return RuleBasedChunkReranker()


@lru_cache(maxsize=8)
def helper_cached_bge_chunk_reranker(
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    local_files_only: bool,
    model_weight: float,
    base_weight: float,
    rule_weight: float,
) -> BgeM3ChunkReranker:
    return BgeM3ChunkReranker(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device or None,
        local_files_only=local_files_only,
        model_weight=model_weight,
        base_weight=base_weight,
        rule_weight=rule_weight,
    )


def helper_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default






def helper_candidate_rerank_text(candidate: dict[str, Any], query: str = "") -> str:
    card = candidate.get("document_card") or {}
    views = candidate.get("matched_views") or []
    chunks = candidate.get("matched_chunks") or []
    terms = helper_query_terms(query)
    parts = [
        f"Title: {candidate.get('title', '')}",
        f"Abstract: {helper_clip_text(candidate.get('abstract', ''), 900)}",
        f"Source: {candidate.get('source_institution', '')}",
        f"Department: {candidate.get('clinical_department', '')}",
        f"Publication date: {candidate.get('publication_date', '')}",
    ]
    card_text = card.get("card_text", "")
    if card_text:
        parts.append("Document card: " + helper_clip_text(card_text, 1400))
    sorted_views = helper_sort_text_items_by_query(views, terms, "text")
    view_text = "\n".join(
        f"{view.get('view_type', 'view')}: {helper_clip_text(str(view.get('text', '')), 700)}" for view in sorted_views[:4]
    )
    if view_text:
        parts.append("Matched views:\n" + view_text)
    sorted_chunks = helper_sort_text_items_by_query(chunks, terms, "content")
    chunk_text = "\n".join(helper_clip_text(str(chunk.get("content", "")), 700) for chunk in sorted_chunks[:5])
    if chunk_text:
        parts.append("Matched chunks:\n" + chunk_text)
    return "\n\n".join(part for part in parts if part.strip())


def helper_chunk_rerank_text(candidate: dict[str, Any]) -> str:
    parts = [
        f"Title: {candidate.get('title', '')}",
        f"Source: {candidate.get('source_institution', '')}",
        f"Department: {candidate.get('clinical_department', '')}",
        f"Publication date: {candidate.get('publication_date', '')}",
        f"Section: {' > '.join(str(part) for part in candidate.get('section_path') or [])}",
        f"Chunk type: {candidate.get('chunk_type', '')}",
        "Content: " + helper_clip_text(str(candidate.get("content", "")), 1500),
    ]
    context = str(candidate.get("source_quote_context") or "")
    if context:
        parts.append("Neighbor context: " + helper_clip_text(context, 700))
    return "\n\n".join(part for part in parts if part.strip())


def helper_sort_text_items_by_query(items: list[dict[str, Any]], terms: list[str], text_key: str) -> list[dict[str, Any]]:
    if not terms:
        return items
    return sorted(
        items,
        key=lambda item: (
            -helper_term_overlap_ratio(str(item.get(text_key) or ""), terms),
            -float(item.get("priority") or 0.0),
        ),
    )


def helper_normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high > low:
        return [(score - low) / (high - low) for score in scores]
    return [1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, score)))) for score in scores]




def helper_contains_term(text: str, terms: list[str]) -> bool:
    lowered = (text or "").lower()
    compacted = helper_compact(text)
    return any(term in lowered or helper_compact(term) in compacted for term in terms)


def helper_term_overlap_ratio(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    lowered = (text or "").lower()
    compacted = helper_compact(text)
    hits = sum(1 for term in terms if term in lowered or helper_compact(term) in compacted)
    return hits / len(terms)




def helper_chunk_type_boost(chunk_type: str) -> float:
    return {
        "recommendation": 0.18,
        "key_message": 0.14,
        "pico": 0.12,
        "algorithm": 0.10,
        "evidence": 0.06,
        "table": 0.05,
    }.get(chunk_type, 0.0)


def helper_looks_like_reference_section(section_text: str) -> bool:
    return bool(re.search(r"(references?|bibliography|\u53c2\u8003\u6587\u732e)", section_text or "", re.I))




def helper_publication_recency_boost(publication_date: str | None) -> float:
    year = helper_publication_year(publication_date)
    if year is None:
        return 0.0
    return max(0.0, min(0.12, (year - 2012) / max(1, 2026 - 2012) * 0.12))




def helper_quality_text(item: dict[str, Any], key: str) -> str:
    direct = str(item.get(key) or "").strip().lower()
    if direct:
        return direct
    card = item.get("document_card") or {}
    return str(card.get(key) or "").strip().lower()


def helper_quality_bool(item: dict[str, Any], key: str) -> bool:
    if key in item:
        return helper_truthy(item.get(key))
    card = item.get("document_card") or {}
    return helper_truthy(card.get(key))


def quality_score_multiplier(item: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Return a conservative score multiplier and its explainable penalties."""

    penalties: dict[str, float] = {}
    cleaning_quality = helper_quality_text(item, "cleaning_quality")
    pdf_text_quality = helper_quality_text(item, "pdf_text_quality")
    source_pdf_text_quality = helper_quality_text(item, "source_pdf_text_quality")
    ocr_status = helper_quality_text(item, "ocr_status")
    cleaning_flags = {
        flag.strip()
        for flag in re.split(r"[,;]\s*", helper_quality_text(item, "cleaning_flags"))
        if flag.strip()
    }

    if cleaning_quality == "poor":
        penalties["cleaning_quality_poor"] = 0.28
    elif cleaning_quality == "warning":
        penalties["cleaning_quality_warning"] = 0.08
    if pdf_text_quality == "poor":
        penalties["pdf_text_quality_poor"] = 0.24
    elif pdf_text_quality == "warning":
        penalties["pdf_text_quality_warning"] = 0.08
    if helper_quality_bool(item, "pdf_needs_ocr"):
        penalties["pdf_needs_ocr"] = 0.18
    if ocr_status in OCR_UNRESOLVED_STATUSES:
        penalties[f"ocr_status_{ocr_status}"] = 0.24
    elif ocr_status in OCR_REVIEW_STATUSES:
        penalties[f"ocr_status_{ocr_status}"] = 0.14
    if not ocr_status and source_pdf_text_quality == "poor" and helper_quality_bool(item, "source_pdf_needs_ocr"):
        penalties["source_pdf_needs_ocr_unresolved"] = 0.14
    if HIGH_RISK_CLEANING_FLAGS & cleaning_flags:
        penalties["high_risk_cleaning_flags"] = 0.18
    if "noisy_ocr_title" in cleaning_flags:
        penalties["noisy_ocr_title"] = 0.10

    total = min(0.55, sum(penalties.values()))
    return round(1.0 - total, 4), penalties

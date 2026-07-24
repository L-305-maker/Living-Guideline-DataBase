"""Structure-first guideline chunking and hybrid retrieval.

This module deliberately uses headings, lexical rules, BM25, dense vectors, and
reranking. It does not extract PICO or depend on structured clinical metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.guideline_chunking.bm25_index import tokenize
from src.guideline_chunking.markdown_parser import parse_markdown_document
from src.guideline_chunking.models import DocumentMeta, ParsedBlock
from src.retrieval.rrf import rrf_fusion


RECOMMENDATION_RE = re.compile(
    r"\b(recommend(?:s|ing|ations?)?|suggest(?:s|ing|ions?)?|should|is advised|we recommend|we suggest)\b|"
    r"\b(?:is|are|was|were|be|being)\s+(?:not\s+)?(?:recommended|suggested)\b|"
    r"\b(?:not\s+)?recommended\s+(?:for|to|that|as)\b|"
    r"\b(?:not\s+)?suggested\s+(?:for|to|that|as)\b|"
    r"(推荐|建议|不推荐|不建议|可推荐|共识建议|专家建议|应当|应该|不应|"
    r"应在|应根据|应进行|应给予|应避免|应考虑|应首先|应仅限|应采用|应使用|应告知|应评估)",
    re.I,
)
DOSE_UNIT_PATTERN = (
    r"\b\d+(?:\.\d+)?\s*(?:[mM][cC][gG]|[mM][gG]|g|[kK][gG]|[mM][lL]|[iI][uU])\b|"
    r"\b\d+(?:\.\d+)?\s+(?:[uU][nN][iI][tT][sS]?)\b"
)
DOSE_UNIT_RE = re.compile(DOSE_UNIT_PATTERN)
CLINICAL_DETAIL_RE = re.compile(
    rf"({DOSE_UNIT_PATTERN}|(?i:\b(?:dose|dosage|regimen)\b)|\b(?:IV|PO)\b|剂量|用量|给药|口服|静脉|疗程|"
    r"(?:用药|给药|治疗|化疗|放疗)方案)"
)
SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"']))|"
    r"(?<=[。！？；])\s*|"
    r"(?<=[遥])\s*|"
    r"(?<=曰)\s*(?=[(（\dA-Z一二三四五六七八九十])"
)
CITATION_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\d{1,4}\s+(?=(?:[A-Z\"']))")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
CHINESE_MOJIBAKE_CHARS = set("窑源园圆员缘远苑愿怨援袁暂咱渊冤遥郾蚤灶则凿葬泽糟酝藻贼燥增")
QUERY_SYNONYMS = {
    "kidney transplant": ["renal transplant", "kidney transplantation", "肾脏移植", "肾移植"],
    "kidney transplantation": ["kidney transplant", "renal transplant", "肾脏移植", "肾移植"],
    "left ventricular ejection fraction": ["LVEF", "左心室射血分数", "左室射血分数"],
    "pneumonia": ["CAP", "community acquired pneumonia", "respiratory infection"],
    "antibiotic": ["antimicrobial", "antibacterial", "empiric therapy"],
    "antibiotics": ["antimicrobials", "antibacterials", "empiric therapy"],
    "hypertension": ["high blood pressure"],
    "diabetes": ["hyperglycemia"],
    "myocardial infarction": ["MI", "heart attack"],
    "acetaminophen": ["paracetamol"],
}
BILINGUAL_QUERY_KEYWORDS = {
    "left ventricular ejection fraction": "左心室射血分数",
    "ejection fraction": "射血分数",
    "below 40": "低于40%",
    "less than 40": "低于40%",
    "kidney transplant": "肾脏移植",
    "kidney transplantation": "肾脏移植",
    "renal transplant": "肾移植",
}
CHUNK_KEYWORDS = {
    "recommendation": "recommend recommendation guideline suggest should advised treatment",
    "recommendation_bundle": "recommend recommendation guideline suggest should advised treatment",
    "clinical_detail": "dose dosage regimen mg mcg ml iv po percent table",
    "general": "guideline section evidence background",
}
MAX_RECOMMENDATION_BUNDLE_CHARS = 2500
QUERY_PHRASE_STOPWORDS = {
    "guideline",
    "recommendation",
    "recommend",
    "suggest",
    "treatment",
    "therapy",
    "指南",
    "推荐",
    "建议",
    "治疗",
}


@dataclass
class GuidelineChunk:
    chunk_id: str
    chunk_type: str
    text: str
    section_path: list[str]
    source_doc_id: str
    text_for_embedding: str
    recommendation: str | None = None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""


class HashingTextEncoder:
    """Offline lexical vector fallback for tests and no-model environments."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [helper_normalize(helper_hashed_vector(text, self.dim)) for text in texts]


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", local_files_only: bool = True) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


class RuleBasedReranker:
    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_variants = expand_query(query)
        terms = sorted(set(tokenize(" ".join(query_variants))))
        phrase_terms = helper_query_phrase_terms(query_variants)
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            chunk = item["chunk"]
            searchable = " ".join(
                [
                    " > ".join(chunk.section_path),
                    chunk.chunk_type,
                    chunk.text,
                    " ".join(chunk.evidence),
                ]
            )
            searchable_lower = searchable.lower()
            overlap = helper_term_overlap(searchable, terms)
            phrase_hits = sum(1 for phrase in phrase_terms if phrase in searchable_lower)
            boost = {
                "recommendation": 0.18,
                "recommendation_bundle": 0.22,
                "clinical_detail": 0.10,
                "general": 0.0,
            }.get(chunk.chunk_type, 0.0)
            if helper_contains_phrase(searchable, query):
                boost += 0.25
            elif any(helper_contains_phrase(searchable, variant) for variant in query_variants[1:]):
                boost += 0.18
            boost += min(0.35, phrase_hits * 0.08)
            item["score"] = (
                float(item.get("score", 0.0)) * (1.0 + boost + min(0.30, overlap * 0.30))
                + min(0.04, phrase_hits * 0.01)
            )
            ranked.append(item)
        ranked.sort(key=lambda item: (-float(item["score"]), item["chunk"].chunk_id))
        return ranked[:top_k]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        local_files_only: bool = True,
        fallback: RuleBasedReranker | None = None,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.fallback = fallback or RuleBasedReranker()
        self._model: Any | None = None

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        fallback_ranked = self.fallback.rerank(query, candidates, len(candidates))
        try:
            model = self.helper_load_model()
            pairs = [(query, item["chunk"].text_for_embedding) for item in fallback_ranked]
            raw_scores = [float(score) for score in model.predict(pairs, show_progress_bar=False)]
        except Exception:
            return fallback_ranked[:top_k]
        normalized = helper_normalize_scores(raw_scores)
        output: list[dict[str, Any]] = []
        for item, model_score in zip(fallback_ranked, normalized):
            updated = dict(item)
            updated["score"] = 0.85 * model_score + 0.15 * float(item.get("score", 0.0))
            output.append(updated)
        output.sort(key=lambda item: (-float(item["score"]), item["chunk"].chunk_id))
        return output[:top_k]

    def helper_load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name, local_files_only=self.local_files_only)
        return self._model


class StructuralBM25Index:
    def __init__(self, chunks: list[GuidelineChunk]) -> None:
        self.chunks = chunks
        self.tokenized = [tokenize(helper_bm25_text(chunk)) for chunk in chunks]
        self.avgdl = sum(len(tokens) for tokens in self.tokenized) / max(1, len(self.tokenized))
        self.df: Counter[str] = Counter()
        for tokens in self.tokenized:
            self.df.update(set(tokens))

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        results: list[tuple[str, float]] = []
        n_docs = max(1, len(self.chunks))
        for chunk, tokens in zip(self.chunks, self.tokenized):
            score = helper_bm25_score(query_tokens, tokens, self.df, n_docs, self.avgdl)
            if score > 0:
                results.append((chunk.chunk_id, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]


class DenseVectorIndex:
    def __init__(self, chunks: list[GuidelineChunk], encoder: TextEncoder | None = None) -> None:
        self.chunks = chunks
        self.encoder = encoder or HashingTextEncoder()
        self.vectors = self.encoder.encode([chunk.text_for_embedding for chunk in chunks])

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        query_vector = self.encoder.encode([query])[0]
        scored = [
            (chunk.chunk_id, helper_dot(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]


class GuidelineRAGIndex:
    def __init__(
        self,
        chunks: list[GuidelineChunk],
        encoder: TextEncoder | None = None,
        reranker: RuleBasedReranker | CrossEncoderReranker | None = None,
    ) -> None:
        self.chunks = chunks
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.bm25 = StructuralBM25Index(chunks)
        self.dense = DenseVectorIndex(chunks, encoder)
        self.reranker = reranker or RuleBasedReranker()

    @classmethod
    def from_markdown_documents(
        cls,
        documents: list[tuple[str, str]],
        encoder: TextEncoder | None = None,
        reranker: RuleBasedReranker | CrossEncoderReranker | None = None,
    ) -> "GuidelineRAGIndex":
        chunks: list[GuidelineChunk] = []
        for doc_id, markdown in documents:
            chunks.extend(build_structural_chunks(markdown, doc_id))
        return cls(chunks, encoder=encoder, reranker=reranker)

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        fused_top_k: int = 100,
    ) -> list[dict[str, Any]]:
        rank_lists: list[list[str]] = []
        for expanded in expand_query(query):
            rank_lists.append([chunk_id for chunk_id, _score in self.bm25.search(expanded, bm25_top_k)])
            rank_lists.append([chunk_id for chunk_id, _score in self.dense.search(expanded, dense_top_k)])
        fused = rrf_fusion(rank_lists)[:fused_top_k]
        candidates = [{"chunk": self.by_id[chunk_id], "score": score} for chunk_id, score in fused if chunk_id in self.by_id]
        reranked = self.reranker.rerank(query, candidates, top_k)
        return [helper_result(item["chunk"], float(item["score"])) for item in reranked]


def build_structural_chunks(markdown: str, doc_id: str, title: str | None = None) -> list[GuidelineChunk]:
    meta = DocumentMeta(doc_id=doc_id, title=title)
    markdown = helper_strip_front_matter(markdown)
    blocks = parse_markdown_document(markdown, meta)
    chunks: list[GuidelineChunk] = []
    consumed_general: set[str] = set()

    # 第一遍优先抽取临床细节和推荐证据，并记录已完整消费的通用块。
    for block in sorted(blocks, key=lambda item: item.order_index):
        if helper_skip_block(block) or helper_is_reference_section(block.heading_path):
            continue
        block_text = helper_clean_block_text(block.text)
        if not block_text:
            continue
        detail_texts = helper_clinical_detail_texts(block_text)
        for detail_text in detail_texts:
            chunks.append(helper_make_chunk("clinical_detail", detail_text, block, len(chunks)))
        if len(detail_texts) == 1 and detail_texts[0] == block_text:
            consumed_general.add(block.block_id)
        for sentence in helper_recommendation_sentences(block_text):
            evidence = helper_next_evidence_paragraphs(block, blocks)
            chunks.extend(helper_make_recommendation_bundle_chunks(sentence, evidence, block, len(chunks)))

    # 第二遍只补未消费且不包含推荐句的普通内容，避免重复 chunk。
    for block in sorted(blocks, key=lambda item: item.order_index):
        if helper_skip_block(block) or helper_is_reference_section(block.heading_path) or block.block_id in consumed_general:
            continue
        block_text = helper_clean_block_text(block.text)
        if block_text and not helper_recommendation_sentences(block_text):
            chunks.append(helper_make_chunk("general", block_text, block, len(chunks)))
    if not chunks:
        # 空文档仍生成标题级兜底 chunk，保证文档在检索库中有稳定入口。
        title_text = helper_first_heading_text(markdown) or title or doc_id
        title_block = ParsedBlock(
            block_id=f"{doc_id}:title",
            doc_id=doc_id,
            block_type="title",
            heading_path=[title_text],
            heading_level=1,
            text=title_text,
            page_start=None,
            page_end=None,
            char_start=None,
            char_end=None,
            order_index=0,
            metadata={"title_only": True},
        )
        chunks.append(helper_make_chunk("general", title_text, title_block, 0))
    return chunks


def enrich_for_embedding(chunk_type: str, section_path: list[str], text: str) -> str:
    return (
        f"[GUIDELINE SECTION]: {' > '.join(section_path)}\n"
        f"[CHUNK TYPE]: {chunk_type}\n"
        "[CONTENT]:\n"
        f"{text}"
    )


def expand_query(query: str, max_queries: int = 6) -> list[str]:
    query = re.sub(r"\s+", " ", query or "").strip()
    if not query:
        return []
    expanded = [query]
    lower = query.lower()
    bilingual_terms = [value for term, value in BILINGUAL_QUERY_KEYWORDS.items() if term in lower]
    if bilingual_terms:
        expanded.append(" ".join(bilingual_terms))
        expanded.append(" ".join([*bilingual_terms, "推荐", "指南"]))
    for term, synonyms in QUERY_SYNONYMS.items():
        if term in lower:
            for synonym in synonyms:
                expanded.append(re.sub(re.escape(term), synonym, query, flags=re.I))
    expanded.extend([f"{query} guideline", f"{query} recommendation", f"{query} treatment guideline"])
    return helper_unique(expanded)[:max_queries]


def helper_strip_front_matter(markdown: str) -> str:
    if not markdown.startswith("---"):
        return markdown
    lines = markdown.splitlines()
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :]).lstrip()
    return markdown


def helper_skip_block(block: ParsedBlock) -> bool:
    text = block.text.strip()
    return bool(
        block.metadata.get("is_heading")
        or not text
        or helper_is_page_number_block(text)
        or helper_looks_like_reference_list(text)
        or (helper_looks_like_garbled_text(text) and not helper_has_table_separator(text))
    )


def helper_is_reference_section(heading_path: list[str]) -> bool:
    for heading in heading_path:
        normalized = helper_normalize_heading_text(heading)
        without_numbering = re.sub(r"^(?:\d+(?:\.\d+)*|[ivx]+)[\).]?\s+", "", normalized)
        if normalized in {"references", "reference", "bibliography", "参考文献"}:
            return True
        if without_numbering in {"references", "reference", "bibliography", "参考文献"}:
            return True
        if normalized.startswith(("references ", "bibliography ")):
            return True
        if without_numbering.startswith(("references ", "bibliography ")):
            return True
        if normalized.startswith("参考文献"):
            return True
        if without_numbering.startswith("参考文献"):
            return True
        if "references and resources" in normalized:
            return True
        if without_numbering.startswith(
            (
                "conflict of interest",
                "conflicts of interest",
                "author contribution",
                "acknowledgment",
                "acknowledgement",
                "funding",
                "research funding",
                "research support",
                "open access",
                "publisher's note",
                "copyright",
                "creative commons",
                "orcid",
            )
        ):
            return True
    return False


def helper_normalize_heading_text(heading: str) -> str:
    normalized = re.sub(r"\s+", " ", heading or "").strip().lower()
    return normalized.strip(" |-:：")


def helper_first_heading_text(markdown: str) -> str | None:
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def helper_is_page_number_block(text: str) -> bool:
    return bool(re.fullmatch(r"(?:page\s*)?\d{1,4}", text.strip(), re.I))


def helper_looks_like_reference_list(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    bracket_citations = re.findall(r"\[\d{1,4}\]", compact)
    journal_tags = re.findall(r"\[j\]", compact, flags=re.I)
    if len(compact) >= 120 and re.match(r"^\[\d{1,4}\]\s+[A-Z].{0,360}\[j\]", compact, flags=re.I):
        return True
    if len(compact) >= 150 and len(bracket_citations) >= 2 and len(journal_tags) >= 2:
        return True
    if len(compact) < 400:
        return False
    numbered_citations = re.findall(
        r"(?:^|[\s;])(?:\d{1,4}[\).]?|\[\d{1,4}\])\s+[A-Z][A-Za-zÀ-ž'’-]+(?:\s+[A-Z][A-Za-zÀ-ž'’-]+){0,4},",
        compact,
    )
    journal_markers = re.findall(r"\b(?:J|Journal|Lancet|BMJ|Cochrane|Circulation|Eur|Clin)\b", compact)
    return len(numbered_citations) >= 4 and len(journal_markers) >= 2


def helper_clean_block_text(text: str) -> str:
    lines = [line for line in (text or "").splitlines() if not helper_is_page_number_block(line.strip())]
    return helper_trim_boilerplate_tail("\n".join(lines).strip())


def helper_trim_boilerplate_tail(text: str) -> str:
    cut = len(text or "")
    for pattern in [
        r"(?im)^\s*参\s*$\s*^\s*考\s*$\s*^\s*文\s*$\s*^\s*献\s*$",
        r"(?im)^\s*参考文献\s*$",
        r"(?im)^\s*利益冲突",
        r"利\s*益\s*冲\s*突",
        r"参\s*考\s*文\s*献\s*\[",
        r"(?im)^\s*conflicts?\s+of\s+interest\b",
        r"(?im)^\s*author contributions?\b",
        r"(?im)^\s*acknowledg(?:e)?ments?\b",
        r"(?im)^\s*funding\b",
        r"(?im)^\s*open access\b",
        r"(?im)^\s*publisher'?s note\b",
        r"(?im)^\s*copyright\b",
        r"(?im)^\s*creative commons\b",
    ]:
        match = re.search(pattern, text or "")
        if match:
            cut = min(cut, match.start())
    return (text or "")[:cut].strip()


def helper_starts_boilerplate(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip().lower()
    compact = re.sub(r"\s+", "", text or "")
    return bool(
        not stripped
        or stripped.startswith(
            (
                "references",
                "bibliography",
                "conflict of interest",
                "conflicts of interest",
                "author contribution",
                "acknowledgment",
                "acknowledgement",
                "funding",
                "open access",
                "publisher's note",
                "copyright",
                "creative commons",
            )
        )
        or compact.startswith(("参考文献", "利益冲突"))
    )


def helper_looks_like_garbled_text(text: str) -> bool:
    stripped = text.strip()
    compact = re.sub(r"\s+", "", stripped)
    if len(compact) < 300:
        return False
    cjk = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    mojibake = sum(1 for char in compact if char in CHINESE_MOJIBAKE_CHARS)
    if cjk and mojibake / cjk > 0.25:
        return True
    punct = sum(1 for char in compact if not char.isalnum())
    punct_ratio = punct / max(1, len(compact))
    space_ratio = sum(1 for char in stripped if char.isspace()) / max(1, len(stripped))
    if punct_ratio > 0.35 and space_ratio < 0.08:
        return True
    ascii_letters = [char.lower() for char in compact if "a" <= char.lower() <= "z"]
    if len(ascii_letters) > 100:
        vowel_ratio = sum(1 for char in ascii_letters if char in "aeiou") / len(ascii_letters)
        return punct_ratio > 0.30 and space_ratio < 0.12 and vowel_ratio < 0.22
    return False


def helper_make_recommendation_bundle(
    recommendation: str,
    evidence: list[str],
    block: ParsedBlock,
    index: int,
) -> GuidelineChunk:
    text = recommendation
    chunk_type = "recommendation"
    if evidence:
        chunk_type = "recommendation_bundle"
        text += "\n\nEvidence:\n" + "\n\n".join(evidence)
    return helper_make_chunk(
        chunk_type,
        text,
        block,
        index,
        recommendation=recommendation,
        evidence=evidence,
    )


def helper_make_recommendation_bundle_chunks(
    recommendation: str,
    evidence: list[str],
    block: ParsedBlock,
    index: int,
) -> list[GuidelineChunk]:
    if len(helper_recommendation_bundle_text(recommendation, evidence)) <= MAX_RECOMMENDATION_BUNDLE_CHARS:
        return [helper_make_recommendation_bundle(recommendation, evidence, block, index)]
    evidence_groups = helper_split_recommendation_evidence(recommendation, evidence)
    if not evidence_groups:
        return [helper_make_recommendation_bundle(recommendation, [], block, index)]
    return [
        helper_make_recommendation_bundle(recommendation, group, block, index + offset)
        for offset, group in enumerate(evidence_groups)
    ]


def helper_recommendation_bundle_text(recommendation: str, evidence: list[str]) -> str:
    text = recommendation
    if evidence:
        text += "\n\nEvidence:\n" + "\n\n".join(evidence)
    return text


def helper_split_recommendation_evidence(recommendation: str, evidence: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for evidence_text in evidence:
        for unit in helper_evidence_bundle_units(recommendation, evidence_text):
            proposed = [*current, unit]
            if current and len(helper_recommendation_bundle_text(recommendation, proposed)) > MAX_RECOMMENDATION_BUNDLE_CHARS:
                groups.append(current)
                current = [unit]
            else:
                current = proposed
    if current:
        groups.append(current)
    return groups


def helper_evidence_bundle_units(recommendation: str, text: str) -> list[str]:
    if len(helper_recommendation_bundle_text(recommendation, [text])) <= MAX_RECOMMENDATION_BUNDLE_CHARS:
        return [text]
    units = helper_sentence_unit_candidates(text)
    if len(units) <= 1:
        units = helper_list_like_units(text)
    return units if len(units) > 1 else [text]


def helper_list_like_units(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r";\s*|；\s*|(?=\(\d+\))|(?=\b\d{1,2}[\).]\s+)|(?=•\s*)", text or "")
        if part.strip()
    ]
    if len(parts) <= 1 and re.search(r"[\u4e00-\u9fff]", text or "") and (text or "").count("+") >= 3:
        parts = [part.strip() for part in re.split(r"(?<=\+)", text or "") if part.strip()]
    if len(parts) <= 1 and len(text or "") > 900 and (text or "").count(",") >= 5:
        parts = [part.strip() for part in re.split(r",\s*", text or "") if part.strip()]
    return parts


def helper_make_chunk(
    chunk_type: str,
    text: str,
    block: ParsedBlock,
    index: int,
    recommendation: str | None = None,
    evidence: list[str] | None = None,
) -> GuidelineChunk:
    section_path = list(block.heading_path) or [str(block.metadata.get("title") or block.doc_id)]
    chunk_id = helper_chunk_id(block.doc_id, chunk_type, block.order_index, index, text)
    return GuidelineChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        text=text,
        section_path=section_path,
        source_doc_id=block.doc_id,
        text_for_embedding=enrich_for_embedding(chunk_type, section_path, text),
        recommendation=recommendation,
        evidence=evidence or [],
        metadata={
            "source_block_id": block.block_id,
            "order_index": block.order_index,
            "page_start": block.page_start,
            "page_end": block.page_end,
        },
    )


def helper_recommendation_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        candidates = helper_sentence_unit_candidates(paragraph)
        for sentence in candidates:
            if RECOMMENDATION_RE.search(sentence) and not helper_looks_like_non_detail_administrative_text(sentence):
                sentences.append(sentence)
    return helper_unique(sentences)


def helper_sentence_candidates(text: str) -> list[str]:
    lines = [line.strip(" -*\t") for line in (text or "").splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]


def helper_next_evidence_paragraphs(block: ParsedBlock, blocks: list[ParsedBlock], limit: int = 2) -> list[str]:
    # 证据扩展只沿当前章节向后进行，遇到新标题或新的推荐块必须停止，避免跨主题拼接。
    evidence: list[str] = []
    for candidate in sorted(blocks, key=lambda item: item.order_index):
        if candidate.order_index <= block.order_index:
            continue
        if candidate.metadata.get("is_heading") or candidate.heading_path != block.heading_path:
            break
        if helper_skip_block(candidate) or candidate.block_type == "table":
            continue
        text = helper_clean_block_text(candidate.text)
        if helper_looks_like_reference_list(candidate.text):
            break
        if helper_starts_boilerplate(candidate.text):
            break
        if not text:
            continue
        for unit in helper_evidence_units(text):
            evidence.append(unit)
            if len(evidence) >= limit:
                break
        if len(evidence) >= limit:
            break
    return evidence


def helper_evidence_units(text: str) -> list[str]:
    trimmed = helper_trim_boilerplate_tail(text)
    if not trimmed or helper_starts_boilerplate(trimmed):
        return []
    raw_units = [
        part.strip()
        for part in re.split(
            r"\n\s*\n|(?=^\s*(?:推荐意见|推荐说明|临床问题|问题\s*\d+|Practice Point|Recommendation:|Recommendations?:|Table\s+\d+|表\s*\d+))",
            trimmed,
            flags=re.M,
        )
        if part.strip()
    ]
    units: list[str] = []
    for raw in raw_units:
        unit = helper_trim_boilerplate_tail(raw)
        unit = helper_trim_reference_tail(unit)
        if not unit or helper_starts_boilerplate(unit) or helper_looks_like_reference_list(unit):
            break
        if helper_looks_like_non_evidence_context(unit):
            continue
        units.append(helper_compact_evidence_unit(unit))
    return helper_unique(units)


def helper_trim_reference_tail(text: str) -> str:
    match = re.search(
        r"(?is)(?:^|\s)(?:\[\d{1,4}\]|\d{1,4}[\).]?)\s*[\u4e00-\u9fffA-Z][^。\n]{0,360}\[J\]",
        text or "",
    )
    if not match:
        return (text or "").strip()
    prefix = (text or "")[: match.start()].strip()
    if not prefix or len(prefix) >= 40:
        return prefix
    return (text or "").strip()


def helper_looks_like_non_evidence_context(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip().lower()
    if not compact:
        return True
    return bool(
        re.match(r"^(?:续表|表)\s*\d+", text or "")
        or helper_looks_like_contributor_list(text)
        or compact.startswith(
            (
                "table ",
                "continued table",
                "figure ",
                "fig. ",
                "decision tree",
                "suggested questions",
            )
        )
    )


def helper_looks_like_contributor_list(text: str) -> bool:
    if len(text or "") < 80:
        return False
    chinese_affiliations = re.findall(r"[\u4e00-\u9fff]{2,4}\([^)]*(?:医院|大学|科|中心)[^)]*\)", text or "")
    english_affiliations = re.findall(
        r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3},\s*(?:MD|PhD|MBBS|MPH|MSc)\b",
        text or "",
    )
    return len(chinese_affiliations) >= 5 or len(english_affiliations) >= 5


def helper_compact_evidence_unit(text: str, max_sentence_units: int = 3) -> str:
    if len(text) <= 1200:
        return text.strip()
    candidates = helper_sentence_unit_candidates(text)
    if len(candidates) <= max_sentence_units:
        return text.strip()
    return " ".join(candidates[:max_sentence_units]).strip()


def helper_is_clinical_detail(text: str) -> bool:
    return bool(
        CLINICAL_DETAIL_RE.search(text or "")
        or helper_looks_like_table(text)
        or helper_looks_like_structured_percent_detail(text)
    )


def helper_clinical_detail_texts(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    if helper_looks_like_garbled_text(stripped) and not helper_has_table_separator(stripped):
        return []
    if helper_looks_like_symbol_noise(stripped):
        return []
    if helper_looks_like_non_detail_administrative_text(stripped):
        return []
    if helper_looks_like_affiliation_table(stripped):
        return []
    if helper_looks_like_table(stripped):
        return [stripped]
    details = [
        sentence
        for sentence in helper_clinical_detail_sentence_candidates(stripped)
        if not helper_tiny_route_fragment(sentence)
        and not helper_too_short_clinical_detail(sentence)
        and not helper_looks_like_symbol_noise(sentence)
        and not helper_looks_like_non_detail_administrative_text(sentence)
        and (CLINICAL_DETAIL_RE.search(sentence) or helper_looks_like_structured_percent_detail(sentence))
    ]
    return helper_unique(details)


def helper_clinical_detail_sentence_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for unit in helper_sentence_unit_candidates(text):
        candidates.extend(helper_clinical_detail_list_units(unit))
    return candidates


def helper_sentence_unit_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for candidate in helper_sentence_candidates(text):
        parts = helper_split_sentence_like(candidate)
        candidates.extend(parts if len(parts) > 1 else [candidate])
    return candidates


def helper_split_sentence_like(text: str) -> list[str]:
    parts = [text.strip()]
    for pattern in (CITATION_SENTENCE_SPLIT_RE, SENTENCE_SPLIT_RE):
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend(piece.strip() for piece in pattern.split(part) if piece.strip())
        parts = next_parts
    return parts


def helper_clinical_detail_list_units(text: str) -> list[str]:
    if len(text or "") < 500 or len(CLINICAL_DETAIL_RE.findall(text or "")) < 2:
        return [text]
    parts = [part.strip() for part in re.split(r";\s*", text or "") if part.strip()]
    return parts if len(parts) > 1 else [text]


def helper_looks_like_symbol_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 120:
        return False
    common_symbols = set(".,;:()[]{}<>/%+-=|#*'\"$&!?_\\/")
    controls = sum(1 for char in compact if ord(char) < 32 or 127 <= ord(char) <= 159)
    non_word = sum(
        1
        for char in compact
        if not char.isalnum() and not ("\u4e00" <= char <= "\u9fff")
    )
    unusual = sum(
        1
        for char in compact
        if not char.isalnum()
        and not ("\u4e00" <= char <= "\u9fff")
        and char not in common_symbols
    )
    length = max(1, len(compact))
    return (
        controls / length > 0.01
        or (non_word / length > 0.35 and unusual / length > 0.12)
        or (non_word / length > 0.50 and unusual / length > 0.02)
    )


def helper_looks_like_non_detail_administrative_text(text: str) -> bool:
    return helper_looks_like_abbreviation_glossary(text) or helper_looks_like_data_structure_text(text)


def helper_looks_like_abbreviation_glossary(text: str) -> bool:
    if len(text or "") < 180:
        return False
    definitions = re.findall(r"\b[A-Z][A-Z0-9\u2043-]{1,9}\s*[:：]", text or "")
    return len(definitions) >= 6


def helper_looks_like_data_structure_text(text: str) -> bool:
    text = text or ""
    if DOSE_UNIT_RE.search(text):
        return False
    marker_hits = sum((text or "").count(marker) for marker in ("数据", "信息", "模块", "结构", "标准"))
    if "治疗方案制定" in text and marker_hits >= 2:
        return True
    if re.search(r"不(?:包含|含有|涉及|提供|包括).{0,8}剂量", text):
        return True
    if len(text) < 500:
        return False
    return marker_hits >= 12


def helper_looks_like_affiliation_table(text: str) -> bool:
    if DOSE_UNIT_RE.search(text) or "%" in text:
        return False
    if not helper_looks_like_table(text):
        return False
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    affiliation_markers = re.compile(
        r"\b(?:university|hospital|department|institute|school|faculty|college|center|centre|clinic|"
        r"association|society|foundation|ministry|laborator(?:y|ies))\b",
        re.I,
    )
    marker_lines = sum(1 for line in lines if affiliation_markers.search(line))
    numbered_lines = sum(1 for line in lines if re.search(r"(?:^|\|)\s*\d{1,3}[A-Z]", line))
    credential_hits = len(re.findall(r"\b(?:MD|PhD|MBBS|MSc|MPH|FRCP|FRCPC)\b", text))
    return (
        marker_lines >= max(5, len(lines) // 4) and numbered_lines >= 2
    ) or (
        credential_hits >= 5 and marker_lines >= 2
    )


def helper_tiny_route_fragment(text: str) -> bool:
    return text.strip().lower().strip(".:;") in {"iv", "po"}


def helper_too_short_clinical_detail(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip().strip("。.;")
    if DOSE_UNIT_RE.search(compact):
        return False
    if len(compact) >= 8 and re.search(r"\d", compact) and re.search(
        r"方案|口服|静脉|疗程|给药|regimen|dose|dosage|\biv\b|\bpo\b",
        compact,
        re.I,
    ):
        return False
    return len(compact) < 16


def helper_looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) >= 2 and helper_has_table_separator(text):
        return True
    return sum(1 for line in lines if line.count("|") >= 2) >= 2


def helper_has_table_separator(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return any(TABLE_SEPARATOR_RE.match(line) for line in lines)


def helper_looks_like_structured_percent_detail(text: str) -> bool:
    if "%" not in (text or ""):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and any("%" in line and re.search(r"\s{2,}|\t|[:：]", line) for line in lines):
        return True
    compact = re.sub(r"\s+", " ", text).strip()
    return len(compact) <= 240 and bool(re.match(r"(?:[-*]\s*)?(?:[\w /-]{1,50}[:：]|\d+[\).])", compact))


def helper_bm25_text(chunk: GuidelineChunk) -> str:
    return "\n".join(
        [
            " > ".join(chunk.section_path),
            chunk.chunk_type,
            CHUNK_KEYWORDS.get(chunk.chunk_type, ""),
            chunk.text,
        ]
    )


def helper_bm25_score(
    query_tokens: list[str],
    tokens: list[str],
    df: Counter[str],
    n_docs: int,
    avgdl: float,
) -> float:
    tf = Counter(tokens)
    dl = len(tokens) or 1
    score = 0.0
    for token in query_tokens:
        if token not in tf:
            continue
        idf = math.log(1 + (n_docs - df[token] + 0.5) / (df[token] + 0.5))
        freq = tf[token]
        score += idf * (freq * 2.2) / (freq + 1.2 * (1 - 0.75 + 0.75 * dl / max(avgdl, 1)))
    return score


def helper_hashed_vector(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return vector


def helper_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def helper_dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def helper_normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _score in scores]
    return [(score - low) / (high - low) for score in scores]


def helper_term_overlap(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    token_set = set(tokenize(text))
    return sum(1 for term in terms if term in token_set) / len(terms)


def helper_query_phrase_terms(query_variants: list[str]) -> list[str]:
    phrases: set[str] = set()
    for variant in query_variants:
        normalized = re.sub(r"\s+", " ", variant.lower()).strip()
        for part in normalized.split():
            if part in QUERY_PHRASE_STOPWORDS:
                continue
            if re.search(r"[\u4e00-\u9fff]", part):
                if len(part) >= 3:
                    phrases.add(part)
    return sorted(phrases)


def helper_contains_phrase(text: str, query: str) -> bool:
    return bool(query and query.lower() in (text or "").lower())


def helper_chunk_id(doc_id: str, chunk_type: str, order_index: int, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{chunk_type}:{order_index}:{index}:{text[:120]}".encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}_{chunk_type}_{digest}"


def helper_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def helper_result(chunk: GuidelineChunk, score: float) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
        "section_path": chunk.section_path,
        "score": score,
    }


def chunk_to_dict(chunk: GuidelineChunk) -> dict[str, Any]:
    return asdict(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Markdown guideline files.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    documents = [(Path(path).stem, Path(path).read_text(encoding="utf-8", errors="replace")) for path in args.input]
    index = GuidelineRAGIndex.from_markdown_documents(documents)
    print(json.dumps(index.search(args.query, top_k=args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

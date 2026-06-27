"""向量化文件：为后续 embedding/RAG 流程准备可入队、可追踪的知识对象。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.common.data_artifacts import utc_now
from src.common.process_jsonl import iter_jsonl, write_jsonl


JsonDict = Dict[str, Any]

QUEUE_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_VERSION = "dense-v1"

RUN_FILES = {
    "recommendation_versions": "recommendation_versions.jsonl",
    "blocks": "blocks.jsonl",
    "pico_questions": "pico_questions.jsonl",
    "evidence_items": "evidence_items.jsonl",
}

BLOCK_SCOPE_KEYWORDS = {
    "recommendation": (
        "recommend",
        "recommendation",
        "should",
        "suggest",
        "offer",
        "avoid",
        "indicated",
    ),
    "evidence": (
        "evidence",
        "trial",
        "study",
        "systematic review",
        "meta-analysis",
        "risk ratio",
        "confidence interval",
    ),
    "grade": (
        "grade",
        "certainty",
        "quality of evidence",
        "strong",
        "conditional",
        "weak",
    ),
    "pico": (
        "population",
        "intervention",
        "comparator",
        "outcome",
        "clinical question",
    ),
}


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def json_fragment(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def labeled_parts(parts: Sequence[tuple[str, Any]]) -> str:
    lines = []
    for label, value in parts:
        if isinstance(value, (dict, list)):
            text = json_fragment(value)
        else:
            text = clean_text(value)
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload(row: JsonDict) -> JsonDict:
    value = row.get("normalized_payload")
    return value if isinstance(value, dict) else {}


def publish_gate(row: JsonDict) -> JsonDict:
    gate = payload(row).get("publish_gate")
    return gate if isinstance(gate, dict) else {}


def is_publishable_version(row: JsonDict) -> bool:
    gate = publish_gate(row)
    if gate and gate.get("publishable") is not True:
        return False
    return str(row.get("quality_status") or gate.get("quality_status") or "") == "publishable"


def common_metadata(row: JsonDict) -> JsonDict:
    return {
        key: row.get(key)
        for key in (
            "guideline_id",
            "recommendation_id",
            "recommendation_version_id",
            "recommendation_candidate_id",
            "pico_id",
            "evidence_id",
            "record_id",
            "source_record_id",
            "paper_id",
            "source",
            "issuer",
            "title",
            "source_url",
            "source_section",
            "published_at",
            "quality_status",
            "status",
            "strength",
            "certainty",
            "direction",
        )
        if row.get(key) not in (None, "")
    }


def queue_item(
    *,
    entity_type: str,
    entity_id: str,
    source_file: str,
    text: str,
    metadata: JsonDict,
    collection: str,
    priority: str,
    embedding_model: str,
    embedding_version: str,
) -> JsonDict:
    digest = text_hash(text)
    return {
        "queue_version": QUEUE_VERSION,
        "vector_id": f"{entity_type}:{entity_id}:{digest[:16]}",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source_file": source_file,
        "collection": collection,
        "priority": priority,
        "text_for_embedding": text,
        "text_hash": digest,
        "embedding_model": embedding_model,
        "embedding_version": embedding_version,
        "metadata": metadata,
        "needs_embedding": True,
    }


def recommendation_version_text(row: JsonDict) -> str:
    return labeled_parts(
        [
            ("Recommendation", row.get("recommendation_text")),
            ("Population", row.get("population")),
            ("Intervention", row.get("intervention")),
            ("Comparator", row.get("comparator")),
            ("Outcome", row.get("outcome_summary")),
            ("Direction", row.get("direction")),
            ("Strength", row.get("strength")),
            ("Certainty", row.get("certainty")),
            ("Rationale", row.get("rationale")),
            ("Remarks", row.get("remarks")),
            ("Source section", row.get("source_section")),
        ]
    )


def recommendation_version_item(
    row: JsonDict,
    source_file: str,
    embedding_model: str,
    embedding_version: str,
    *,
    include_review: bool,
) -> JsonDict | None:
    entity_id = clean_text(row.get("recommendation_version_id"))
    text = recommendation_version_text(row)
    if not entity_id or not text:
        return None
    published = is_publishable_version(row)
    if not published and not include_review:
        return None
    metadata = common_metadata(row)
    metadata["publishable"] = published
    return queue_item(
        entity_type="recommendation_version",
        entity_id=entity_id,
        source_file=source_file,
        text=text,
        metadata=metadata,
        collection="lg_recommendations_published" if published else "lg_review_candidates",
        priority="high" if published else "review",
        embedding_model=embedding_model,
        embedding_version=embedding_version,
    )


def block_scope(row: JsonDict) -> str:
    haystack = " ".join(
        [
            clean_text(row.get("heading")),
            clean_text(row.get("text")),
            " ".join(clean_text(part) for part in row.get("section_path") or []),
            " ".join(clean_text(hint) for hint in row.get("candidate_hints") or []),
        ]
    ).lower()
    for scope, keywords in BLOCK_SCOPE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return scope
    return "background"


def source_block_text(row: JsonDict) -> str:
    section_path = row.get("section_path") or []
    return labeled_parts(
        [
            ("Title", row.get("title")),
            ("Section", " > ".join(clean_text(part) for part in section_path if clean_text(part))),
            ("Heading", row.get("heading")),
            ("Block type", row.get("block_type")),
            ("Text", row.get("text")),
        ]
    )


def source_block_item(
    row: JsonDict,
    source_file: str,
    embedding_model: str,
    embedding_version: str,
    *,
    include_background_blocks: bool,
    min_block_chars: int,
) -> JsonDict | None:
    entity_id = clean_text(row.get("block_id"))
    text = source_block_text(row)
    scope = block_scope(row)
    if not entity_id or len(clean_text(row.get("text"))) < min_block_chars:
        return None
    if scope == "background" and not include_background_blocks:
        return None
    metadata = common_metadata(row)
    metadata.update(
        {
            "block_scope": scope,
            "block_type": row.get("block_type"),
            "section_path": row.get("section_path") or [],
            "order": row.get("order"),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
        }
    )
    return queue_item(
        entity_type="source_block",
        entity_id=entity_id,
        source_file=source_file,
        text=text,
        metadata=metadata,
        collection="lg_source_blocks",
        priority="high" if scope in {"recommendation", "evidence", "grade"} else "normal",
        embedding_model=embedding_model,
        embedding_version=embedding_version,
    )


def pico_question_text(row: JsonDict) -> str:
    return labeled_parts(
        [
            ("Clinical question", row.get("clinical_question")),
            ("Population", row.get("population")),
            ("Intervention", row.get("intervention")),
            ("Comparator", row.get("comparator")),
            ("Outcomes", row.get("outcomes")),
            ("Source", row.get("source_text") or row.get("source_span")),
        ]
    )


def pico_question_item(row: JsonDict, source_file: str, embedding_model: str, embedding_version: str) -> JsonDict | None:
    entity_id = clean_text(row.get("pico_id"))
    text = pico_question_text(row)
    if not entity_id or not text:
        return None
    return queue_item(
        entity_type="pico_question",
        entity_id=entity_id,
        source_file=source_file,
        text=text,
        metadata=common_metadata(row),
        collection="lg_pico_questions",
        priority="high",
        embedding_model=embedding_model,
        embedding_version=embedding_version,
    )


def evidence_item_text(row: JsonDict) -> str:
    return labeled_parts(
        [
            ("Population", row.get("population_extracted")),
            ("Intervention", row.get("intervention_extracted")),
            ("Comparator", row.get("comparator_extracted")),
            ("Outcomes", row.get("outcomes_extracted")),
            ("Study design", row.get("study_design")),
            ("Sample size", row.get("sample_size")),
            ("Effect size", row.get("effect_size")),
            ("Confidence interval", row.get("confidence_interval")),
            ("Effect direction", row.get("effect_direction")),
            ("Source", row.get("source_text") or row.get("source_span")),
        ]
    )


def evidence_item(row: JsonDict, source_file: str, embedding_model: str, embedding_version: str) -> JsonDict | None:
    entity_id = clean_text(row.get("evidence_id"))
    text = evidence_item_text(row)
    if not entity_id or not text:
        return None
    return queue_item(
        entity_type="evidence_item",
        entity_id=entity_id,
        source_file=source_file,
        text=text,
        metadata=common_metadata(row),
        collection="lg_evidence_items",
        priority="high",
        embedding_model=embedding_model,
        embedding_version=embedding_version,
    )


def iter_queue_items(
    run_dir: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_version: str = DEFAULT_EMBEDDING_VERSION,
    include_review_versions: bool = False,
    include_background_blocks: bool = False,
    min_block_chars: int = 80,
) -> Iterable[JsonDict]:
    root = Path(run_dir)
    version_path = root / RUN_FILES["recommendation_versions"]
    if version_path.exists():
        for row in iter_jsonl(version_path):
            item = recommendation_version_item(
                row,
                RUN_FILES["recommendation_versions"],
                embedding_model,
                embedding_version,
                include_review=include_review_versions,
            )
            if item:
                yield item

    block_path = root / RUN_FILES["blocks"]
    if block_path.exists():
        for row in iter_jsonl(block_path):
            item = source_block_item(
                row,
                RUN_FILES["blocks"],
                embedding_model,
                embedding_version,
                include_background_blocks=include_background_blocks,
                min_block_chars=min_block_chars,
            )
            if item:
                yield item

    pico_path = root / RUN_FILES["pico_questions"]
    if pico_path.exists():
        for row in iter_jsonl(pico_path):
            item = pico_question_item(row, RUN_FILES["pico_questions"], embedding_model, embedding_version)
            if item:
                yield item

    evidence_path = root / RUN_FILES["evidence_items"]
    if evidence_path.exists():
        for row in iter_jsonl(evidence_path):
            item = evidence_item(row, RUN_FILES["evidence_items"], embedding_model, embedding_version)
            if item:
                yield item


def build_embedding_queue(
    run_dir: str | Path,
    output: str | Path,
    *,
    manifest_output: str | Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_version: str = DEFAULT_EMBEDDING_VERSION,
    include_review_versions: bool = False,
    include_background_blocks: bool = False,
    min_block_chars: int = 80,
) -> JsonDict:
    items = list(
        iter_queue_items(
            run_dir,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            include_review_versions=include_review_versions,
            include_background_blocks=include_background_blocks,
            min_block_chars=min_block_chars,
        )
    )
    written = write_jsonl(output, items)
    entity_counts = Counter(item["entity_type"] for item in items)
    collection_counts = Counter(item["collection"] for item in items)
    manifest: JsonDict = {
        "queue_version": QUEUE_VERSION,
        "created_at": utc_now(),
        "run_dir": str(Path(run_dir)),
        "output": str(Path(output)),
        "embedding_model": embedding_model,
        "embedding_version": embedding_version,
        "include_review_versions": include_review_versions,
        "include_background_blocks": include_background_blocks,
        "min_block_chars": min_block_chars,
        "written_rows": written,
        "entity_type_counts": dict(entity_counts),
        "collection_counts": dict(collection_counts),
    }
    if manifest_output:
        write_jsonl(manifest_output, [manifest])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stable embedding queue from a processed Living-Guideline run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-version", default=DEFAULT_EMBEDDING_VERSION)
    parser.add_argument("--include-review-versions", action="store_true")
    parser.add_argument("--include-background-blocks", action="store_true")
    parser.add_argument("--min-block-chars", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_embedding_queue(
        args.run_dir,
        args.output,
        manifest_output=args.manifest_output,
        embedding_model=args.embedding_model,
        embedding_version=args.embedding_version,
        include_review_versions=args.include_review_versions,
        include_background_blocks=args.include_background_blocks,
        min_block_chars=args.min_block_chars,
    )
    print(
        "embedding_queue_written={rows} output={output} collections={collections}".format(
            rows=manifest["written_rows"],
            output=manifest["output"],
            collections=manifest["collection_counts"],
        )
    )


if __name__ == "__main__":
    main()


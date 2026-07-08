"""Optional FAISS vector indexes for document cards, document views, and chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Iterable, Iterator

from src.retrieval.bm25_store import _card_records, _view_records
from src.retrieval.chunk_normalizer import iter_normalized_chunks
from src.utils.io import DATA_DIR, ensure_dir, ensure_parent, read_jsonl


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_LOCAL_FILES_ONLY = os.getenv("BGE_VECTOR_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_FP16 = os.getenv("BGE_VECTOR_FP16", "1").strip().lower() not in {"0", "false", "no"}


def _load_vector_dependencies():
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"Vector dependencies unavailable: {exc}") from exc
    return faiss, np, SentenceTransformer


@lru_cache(maxsize=4)
def _cached_model(model_name: str, local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY):
    _faiss, _np, SentenceTransformer = _load_vector_dependencies()
    return SentenceTransformer(model_name, local_files_only=local_files_only)


def _batched(records: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _limited(records: Iterable[dict[str, Any]], limit: int | None) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        yield record


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _allowed_doc_ids(documents_path: str | Path | None) -> set[str] | None:
    if documents_path is None:
        return None
    path = Path(documents_path)
    if not path.exists():
        return None
    return {str(record["doc_id"]) for record in read_jsonl(path) if record.get("doc_id")}


def _encode(model: Any, texts: list[str], normalize_embeddings: bool = True):
    return model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")


def _maybe_half(model: Any, device: str | None, fp16: bool) -> None:
    if not fp16 or not (device or "").startswith("cuda"):
        return
    try:
        model.half()
    except Exception:
        return


def _record_text(record: dict[str, Any], text_field: str) -> str:
    return str(record.get(text_field) or record.get("content") or "")


def _build_index_stream(
    records: Iterable[dict[str, Any]],
    text_field: str,
    id_field: str,
    index_path: Path,
    mapping_path: Path,
    model_name: str,
    batch_size: int,
    limit: int | None = None,
    device: str | None = None,
    max_seq_length: int | None = None,
    progress_every: int = 64,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    fp16: bool = DEFAULT_FP16,
) -> dict[str, Any]:
    faiss, _np, SentenceTransformer = _load_vector_dependencies()
    model_kwargs = {"device": device} if device else {}
    model_kwargs["local_files_only"] = local_files_only
    model = SentenceTransformer(model_name, **model_kwargs)
    _maybe_half(model, device, fp16)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    ensure_dir(index_path.parent)
    ensure_parent(mapping_path)
    tmp_index_path = index_path.with_name(index_path.name + ".tmp")
    tmp_mapping_path = mapping_path.with_name(mapping_path.name + ".tmp")

    index = None
    count = 0
    dim = None
    with Path(tmp_mapping_path).open("w", encoding="utf-8", newline="\n") as mapping_handle:
        for batch in _batched(_limited(records, limit), batch_size):
            texts = [_record_text(record, text_field) for record in batch]
            vectors = _encode(model, texts)
            if index is None:
                dim = int(vectors.shape[1])
                index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            for record in batch:
                mapping_handle.write(json.dumps({"vector_id": count, id_field: record[id_field]}, ensure_ascii=False) + "\n")
                count += 1
            if count % progress_every == 0:
                print(json.dumps({"index": str(index_path.name), "encoded": count}, ensure_ascii=False), flush=True)

    if index is None:
        index = faiss.IndexFlatIP(1)
        dim = 1
    faiss.write_index(index, str(tmp_index_path))
    os.replace(tmp_index_path, index_path)
    os.replace(tmp_mapping_path, mapping_path)
    return {"count": count, "dim": dim}


def _build_index_shards(
    records: Iterable[dict[str, Any]],
    text_field: str,
    id_field: str,
    shard_dir: Path,
    manifest_path: Path,
    model_name: str,
    batch_size: int,
    shard_size: int,
    limit: int | None = None,
    device: str | None = None,
    max_seq_length: int | None = None,
    progress_every: int = 64,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    fp16: bool = DEFAULT_FP16,
) -> dict[str, Any]:
    faiss, _np, SentenceTransformer = _load_vector_dependencies()
    model_kwargs = {"device": device} if device else {}
    model_kwargs["local_files_only"] = local_files_only
    model = SentenceTransformer(model_name, **model_kwargs)
    _maybe_half(model, device, fp16)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    ensure_dir(shard_dir)
    ensure_parent(manifest_path)

    manifest: dict[str, Any] = {
        "model": model_name,
        "id_field": id_field,
        "text_field": text_field,
        "dim": None,
        "shard_size": shard_size,
        "shards": [],
    }
    total = 0
    current_shard: list[dict[str, Any]] = []
    shard_index = 0

    def flush_shard(shard_records: list[dict[str, Any]], shard_no: int, offset: int) -> dict[str, Any]:
        index_path = shard_dir / f"shard_{shard_no:05d}.index"
        mapping_path = shard_dir / f"shard_{shard_no:05d}_mapping.jsonl"
        expected = len(shard_records)
        if index_path.exists() and mapping_path.exists() and _count_jsonl(mapping_path) == expected:
            expected_ids = [record[id_field] for record in shard_records]
            existing_ids = [record.get(id_field) for record in read_jsonl(mapping_path)]
            if existing_ids == expected_ids:
                return {
                    "index": str(index_path),
                    "mapping": str(mapping_path),
                    "offset": offset,
                    "count": expected,
                    "skipped_existing": True,
                }

        tmp_index_path = index_path.with_name(index_path.name + ".tmp")
        tmp_mapping_path = mapping_path.with_name(mapping_path.name + ".tmp")
        index = None
        local_count = 0
        dim = None
        with tmp_mapping_path.open("w", encoding="utf-8", newline="\n") as mapping_handle:
            for batch in _batched(shard_records, batch_size):
                texts = [_record_text(record, text_field) for record in batch]
                vectors = _encode(model, texts)
                if index is None:
                    dim = int(vectors.shape[1])
                    index = faiss.IndexFlatIP(dim)
                index.add(vectors)
                for record in batch:
                    mapping_handle.write(
                        json.dumps({"vector_id": local_count, id_field: record[id_field]}, ensure_ascii=False) + "\n"
                    )
                    local_count += 1
                if local_count % progress_every == 0 or local_count == expected:
                    print(
                        json.dumps(
                            {"shard": shard_no, "encoded_in_shard": local_count, "global_encoded": offset + local_count},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        if index is None:
            index = faiss.IndexFlatIP(1)
            dim = 1
        faiss.write_index(index, str(tmp_index_path))
        os.replace(tmp_index_path, index_path)
        os.replace(tmp_mapping_path, mapping_path)
        if manifest["dim"] is None:
            manifest["dim"] = dim
        return {"index": str(index_path), "mapping": str(mapping_path), "offset": offset, "count": local_count}

    for record in _limited(records, limit):
        current_shard.append(record)
        if len(current_shard) >= shard_size:
            shard = flush_shard(current_shard, shard_index, total)
            manifest["shards"].append(shard)
            total += int(shard["count"])
            shard_index += 1
            current_shard = []
            manifest["count"] = total
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if current_shard:
        shard = flush_shard(current_shard, shard_index, total)
        manifest["shards"].append(shard)
        total += int(shard["count"])
    manifest["count"] = total
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"count": total, "dim": manifest["dim"], "shards": len(manifest["shards"]), "manifest": str(manifest_path)}


def build_vector_indexes(
    clean_dir: str | Path = DATA_DIR / "markdown_clean",
    chunks_path: str | Path = DATA_DIR / "chunks" / "all_chunks.jsonl",
    output_dir: str | Path = DATA_DIR / "index",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    target: str = "all",
    limit: int | None = None,
    device: str | None = None,
    max_seq_length: int | None = None,
    shard_size: int | None = None,
    progress_every: int = 64,
    documents_path: str | Path | None = DATA_DIR / "documents.jsonl",
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    fp16: bool = DEFAULT_FP16,
) -> dict[str, Any]:
    if target not in {"all", "document_cards", "document_views", "chunks"}:
        raise ValueError("target must be one of: all, document_cards, document_views, chunks")
    out = ensure_dir(output_dir)
    allowed_doc_ids = _allowed_doc_ids(documents_path)
    data_dir = Path(output_dir).parent
    needs_cards = target in {"all", "document_cards"}
    needs_views = target in {"all", "document_views"}
    cards = _card_records(clean_dir, data_dir / "document_cards.jsonl") if needs_cards else []
    views = _view_records(clean_dir, data_dir / "document_views.jsonl") if needs_views else []
    if allowed_doc_ids is not None:
        cards = [record for record in cards if record.get("doc_id") in allowed_doc_ids]
        views = [record for record in views if record.get("doc_id") in allowed_doc_ids]
    chunks_count = 0
    if Path(chunks_path).exists():
        for chunk in iter_normalized_chunks(data_dir, chunks_path):
            if allowed_doc_ids is None or chunk.get("doc_id") in allowed_doc_ids:
                chunks_count += 1
    result: dict[str, Any] = {
        "document_cards": len(cards),
        "document_views": len(views),
        "chunks": chunks_count,
        "model": model_name,
        "batch_size": batch_size,
    }
    try:
        if target in {"all", "document_cards"} and cards:
            if shard_size:
                result["document_card_index"] = _build_index_shards(
                    cards,
                    "card_text",
                    "doc_id",
                    out / "faiss_document_cards_shards",
                    out / "faiss_document_cards_shards.json",
                    model_name,
                    batch_size,
                    shard_size,
                    limit,
                    device,
                    max_seq_length,
                    progress_every,
                    local_files_only,
                    fp16,
                )
            else:
                result["document_card_index"] = _build_index_stream(
                    cards,
                    "card_text",
                    "doc_id",
                    out / "faiss_document_cards.index",
                    out / "faiss_document_cards_mapping.jsonl",
                    model_name,
                    batch_size,
                    limit,
                    device,
                    max_seq_length,
                    progress_every,
                    local_files_only,
                    fp16,
                )
        if target in {"all", "document_views"} and views:
            if shard_size:
                result["document_view_index"] = _build_index_shards(
                    views,
                    "text",
                    "view_id",
                    out / "faiss_document_views_shards",
                    out / "faiss_document_views_shards.json",
                    model_name,
                    batch_size,
                    shard_size,
                    limit,
                    device,
                    max_seq_length,
                    progress_every,
                    local_files_only,
                    fp16,
                )
            else:
                result["document_view_index"] = _build_index_stream(
                    views,
                    "text",
                    "view_id",
                    out / "faiss_document_views.index",
                    out / "faiss_document_views_mapping.jsonl",
                    model_name,
                    batch_size,
                    limit,
                    device,
                    max_seq_length,
                    progress_every,
                    local_files_only,
                    fp16,
                )
        if target in {"all", "chunks"} and chunks_count:
            if shard_size:
                result["chunk_index"] = _build_index_shards(
                    (
                        chunk
                        for chunk in iter_normalized_chunks(data_dir, chunks_path)
                        if allowed_doc_ids is None or chunk.get("doc_id") in allowed_doc_ids
                    ),
                    "retrieval_text",
                    "chunk_id",
                    out / "faiss_chunks_shards",
                    out / "faiss_chunks_shards.json",
                    model_name,
                    batch_size,
                    shard_size,
                    limit,
                    device,
                    max_seq_length,
                    progress_every,
                    local_files_only,
                    fp16,
                )
            else:
                result["chunk_index"] = _build_index_stream(
                    (
                        chunk
                        for chunk in iter_normalized_chunks(data_dir, chunks_path)
                        if allowed_doc_ids is None or chunk.get("doc_id") in allowed_doc_ids
                    ),
                    "retrieval_text",
                    "chunk_id",
                    out / "faiss_chunks.index",
                    out / "faiss_chunks_mapping.jsonl",
                    model_name,
                    batch_size,
                    limit,
                    device,
                    max_seq_length,
                    progress_every,
                    local_files_only,
                    fp16,
                )
        metadata_path = out / "vector_indexes_metadata.json"
        if target != "all" and metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            existing.update(result)
            result = existing
        result["built"] = True
        (metadata_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except RuntimeError as exc:
        result.update({"built": False, "error": str(exc)})
        (out / "vector_disabled.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _vector_search_shards(
    query: str,
    manifest_path: Path,
    id_field: str,
    top_n: int,
    model_name: str,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
) -> list[str]:
    if not manifest_path.exists():
        return []
    try:
        faiss, _np, _SentenceTransformer = _load_vector_dependencies()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = _cached_model(model_name, local_files_only)
        vector = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vector)
        candidates: list[tuple[float, str]] = []
        for shard in manifest.get("shards", []):
            index_file = Path(shard["index"])
            mapping_file = Path(shard["mapping"])
            if not index_file.exists() or not mapping_file.exists():
                continue
            mappings = list(read_jsonl(mapping_file))
            index = faiss.read_index(str(index_file))
            scores, ids = index.search(vector, min(top_n, len(mappings)))
            for score, item_index in zip(scores[0], ids[0]):
                if 0 <= int(item_index) < len(mappings):
                    candidates.append((float(score), mappings[int(item_index)][id_field]))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        seen: set[str] = set()
        results = []
        for _score, item_id in candidates:
            if item_id in seen:
                continue
            seen.add(item_id)
            results.append(item_id)
            if len(results) >= top_n:
                break
        return results
    except Exception:
        return []


def vector_search(
    query: str,
    index_path: str | Path,
    mapping_path: str | Path,
    id_field: str,
    top_n: int = 50,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
) -> list[str]:
    index_path = Path(index_path)
    mapping_path = Path(mapping_path)
    manifest_path = index_path.with_name(index_path.stem + "_shards.json")
    if manifest_path.exists():
        return _vector_search_shards(query, manifest_path, id_field, top_n, model_name, local_files_only)
    if not index_path.exists() or not mapping_path.exists():
        return []
    try:
        faiss, _np, _SentenceTransformer = _load_vector_dependencies()
        mappings = list(read_jsonl(mapping_path))
        index = faiss.read_index(str(index_path))
        model = _cached_model(model_name, local_files_only)
        vector = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vector)
        _scores, ids = index.search(vector, min(top_n, len(mappings)))
        return [mappings[int(i)][id_field] for i in ids[0] if 0 <= int(i) < len(mappings)]
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default=str(DATA_DIR / "markdown_clean"))
    parser.add_argument("--chunks", default=str(DATA_DIR / "chunks" / "all_chunks.jsonl"))
    parser.add_argument("--output-dir", default=str(DATA_DIR / "index"))
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target", choices=["all", "document_cards", "document_views", "chunks"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--documents", default=str(DATA_DIR / "documents.jsonl"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_vector_indexes(
                args.clean_dir,
                args.chunks,
                args.output_dir,
                args.model,
                args.batch_size,
                args.target,
                args.limit,
                args.device,
                args.max_seq_length,
                args.shard_size,
                args.progress_every,
                args.documents,
                not args.allow_download,
                not args.fp32,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Optional FAISS vector indexes for document cards, document views, and chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Iterable, Iterator

from src.retrieval.bm25_store import helper_card_records, helper_view_records
from src.retrieval.chunk_normalizer import iter_normalized_chunks
from src.utils.io import DATA_DIR, ensure_dir, ensure_parent, read_jsonl


DEFAULT_EMBEDDING_MODEL = os.getenv("BGE_VECTOR_MODEL", "BAAI/bge-m3")
DEFAULT_LOCAL_FILES_ONLY = os.getenv("BGE_VECTOR_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_FP16 = os.getenv("BGE_VECTOR_FP16", "1").strip().lower() not in {"0", "false", "no"}


def vector_retrieval_required(required: bool | None = None) -> bool:
    if required is not None:
        return required
    return os.getenv("VECTOR_RETRIEVAL_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "on"}


def helper_load_vector_dependencies():
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"Vector dependencies unavailable: {exc}") from exc
    return faiss, np, SentenceTransformer


@lru_cache(maxsize=4)
def helper_cached_model(model_name: str, local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY):
    _faiss, _np, SentenceTransformer = helper_load_vector_dependencies()
    return SentenceTransformer(model_name, local_files_only=local_files_only)


def helper_batched(records: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def helper_limited(records: Iterable[dict[str, Any]], limit: int | None) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        yield record


def helper_count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def helper_allowed_doc_ids(documents_path: str | Path | None) -> set[str] | None:
    if documents_path is None:
        return None
    path = Path(documents_path)
    if not path.exists():
        return None
    return {str(record["doc_id"]) for record in read_jsonl(path) if record.get("doc_id")}


def helper_encode(model: Any, texts: list[str], normalize_embeddings: bool = True):
    return model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")


def helper_maybe_half(model: Any, device: str | None, fp16: bool) -> None:
    if not fp16 or not (device or "").startswith("cuda"):
        return
    try:
        model.half()
    except Exception:
        return


def helper_record_text(record: dict[str, Any], text_field: str) -> str:
    return str(record.get(text_field) or record.get("content") or "")


def helper_build_index_stream(
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
    faiss, _np, SentenceTransformer = helper_load_vector_dependencies()
    model_kwargs = {"device": device} if device else {}
    model_kwargs["local_files_only"] = local_files_only
    model = SentenceTransformer(model_name, **model_kwargs)
    helper_maybe_half(model, device, fp16)
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
        for batch in helper_batched(helper_limited(records, limit), batch_size):
            texts = [helper_record_text(record, text_field) for record in batch]
            vectors = helper_encode(model, texts)
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


def helper_build_index_shards(
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
    faiss, _np, SentenceTransformer = helper_load_vector_dependencies()
    model_kwargs = {"device": device} if device else {}
    model_kwargs["local_files_only"] = local_files_only
    model = SentenceTransformer(model_name, **model_kwargs)
    helper_maybe_half(model, device, fp16)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    ensure_dir(shard_dir)
    ensure_parent(manifest_path)

    # 清单记录模型、字段和分片顺序；检索端只依赖该清单即可遍历全部分片。
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
        if index_path.exists() and mapping_path.exists() and helper_count_jsonl(mapping_path) == expected:
            # 不能只比较条数：相同数量但顺序不同会让 FAISS 行号指向错误业务 ID。
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

        # 索引和映射先写临时文件，再分别原子替换，避免中断时留下半写入文件。
        tmp_index_path = index_path.with_name(index_path.name + ".tmp")
        tmp_mapping_path = mapping_path.with_name(mapping_path.name + ".tmp")
        index = None
        local_count = 0
        dim = None
        with tmp_mapping_path.open("w", encoding="utf-8", newline="\n") as mapping_handle:
            for batch in helper_batched(shard_records, batch_size):
                texts = [helper_record_text(record, text_field) for record in batch]
                vectors = helper_encode(model, texts)
                if index is None:
                    # 向量维度由第一批真实编码结果确定，避免配置值与模型实际输出不一致。
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

    for record in helper_limited(records, limit):
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
    # target 同时控制数据读取和产物命名；尽早拒绝拼写错误，避免生成不完整索引。
    if target not in {"all", "document_cards", "document_views", "chunks"}:
        raise ValueError("target must be one of: all, document_cards, document_views, chunks")
    out = ensure_dir(output_dir)
    allowed_doc_ids = helper_allowed_doc_ids(documents_path)
    data_dir = Path(output_dir).parent
    needs_cards = target in {"all", "document_cards"}
    needs_views = target in {"all", "document_views"}
    cards = helper_card_records(clean_dir, data_dir / "document_cards.jsonl") if needs_cards else []
    views = helper_view_records(clean_dir, data_dir / "document_views.jsonl") if needs_views else []
    if allowed_doc_ids is not None:
        # documents.jsonl 是可服务文档的权威集合，孤立卡片或视图不能进入检索索引。
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
                result["document_card_index"] = helper_build_index_shards(
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
                result["document_card_index"] = helper_build_index_stream(
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
                result["document_view_index"] = helper_build_index_shards(
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
                result["document_view_index"] = helper_build_index_stream(
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
                result["chunk_index"] = helper_build_index_shards(
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
                result["chunk_index"] = helper_build_index_stream(
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
            # 单目标增量构建只覆盖对应条目，保留其他目标上一次成功构建的元数据。
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


def helper_vector_search_shards(
    query: str,
    manifest_path: Path,
    id_field: str,
    top_n: int,
    model_name: str,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    required: bool = False,
) -> list[str]:
    if not manifest_path.exists():
        return []
    try:
        faiss, _np, _SentenceTransformer = helper_load_vector_dependencies()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = manifest.get("shards", [])
        if required and not shards:
            raise RuntimeError(f"Vector retrieval required but shard manifest is empty: {manifest_path}")
        model = helper_cached_model(model_name, local_files_only)
        vector = model.encode([query], convert_to_numpy=True).astype("float32")
        # 建库向量同样做过 L2 归一化，因此内积可直接作为余弦相似度使用。
        faiss.normalize_L2(vector)
        candidates: list[tuple[float, str]] = []
        for shard in shards:
            index_file = Path(shard["index"])
            mapping_file = Path(shard["mapping"])
            if not index_file.exists() or not mapping_file.exists():
                if required:
                    raise RuntimeError(
                        f"Vector retrieval required but shard artifacts are missing: {index_file}, {mapping_file}"
                    )
                continue
            mappings = list(read_jsonl(mapping_file))
            index = faiss.read_index(str(index_file))
            # FAISS 行号必须与映射文件逐行对应；强制模式下任何数量偏差都应立即失败。
            if required and int(index.ntotal) != len(mappings):
                raise RuntimeError(
                    f"Vector index/mapping size mismatch: {index_file} has {index.ntotal}, {mapping_file} has {len(mappings)}"
                )
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
    except Exception as exc:
        # 可选模式保持历史兼容并返回空通道；强制模式保留失败原因，禁止静默退化。
        if required:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Vector retrieval failed for {manifest_path}: {exc}") from exc
        return []


def vector_search(
    query: str,
    index_path: str | Path,
    mapping_path: str | Path,
    id_field: str,
    top_n: int = 50,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    required: bool | None = None,
) -> list[str]:
    index_path = Path(index_path)
    mapping_path = Path(mapping_path)
    required = vector_retrieval_required(required)
    manifest_path = index_path.with_name(index_path.stem + "_shards.json")
    # 同时存在单文件和分片产物时优先分片清单，因为它代表较新的可扩展布局。
    if manifest_path.exists():
        return helper_vector_search_shards(query, manifest_path, id_field, top_n, model_name, local_files_only, required)
    if not index_path.exists() or not mapping_path.exists():
        if required:
            raise RuntimeError(
                f"Vector retrieval required but index artifacts are missing: {index_path}, {mapping_path}"
            )
        return []
    try:
        faiss, _np, _SentenceTransformer = helper_load_vector_dependencies()
        mappings = list(read_jsonl(mapping_path))
        index = faiss.read_index(str(index_path))
        # FAISS 行号必须与映射文件逐行对应；强制模式下任何数量偏差都应立即失败。
        if required and int(index.ntotal) != len(mappings):
            raise RuntimeError(
                f"Vector index/mapping size mismatch: {index_path} has {index.ntotal}, {mapping_path} has {len(mappings)}"
            )
        model = helper_cached_model(model_name, local_files_only)
        vector = model.encode([query], convert_to_numpy=True).astype("float32")
        # 建库向量同样做过 L2 归一化，因此内积可直接作为余弦相似度使用。
        faiss.normalize_L2(vector)
        _scores, ids = index.search(vector, min(top_n, len(mappings)))
        return [mappings[int(i)][id_field] for i in ids[0] if 0 <= int(i) < len(mappings)]
    except Exception as exc:
        # 可选模式保持历史兼容并返回空通道；强制模式保留失败原因，禁止静默退化。
        if required:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Vector retrieval failed for {index_path}: {exc}") from exc
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

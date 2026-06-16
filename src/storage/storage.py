from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer


COLLECTION_NAME = "medical_guidelines"
DOCUMENT_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
VECTOR_SIZE = 768
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_UPSERT_BATCH_SIZE = 64
DOCUMENT_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 64


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DOC_TOKENIZER: Optional[Any] = None
_DOC_MODEL: Optional[torch.nn.Module] = None
_QUERY_TOKENIZER: Optional[Any] = None
_QUERY_MODEL: Optional[torch.nn.Module] = None


def get_qdrant_client() -> QdrantClient:
    # 创建 Qdrant client，并把连接错误转成更清晰的提示.
    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        return client
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"无法连接 Qdrant: {QDRANT_URL}。请确认服务已启动，或设置 QDRANT_URL。原始错误: {exc}"
        ) from exc


def _load_document_encoder() -> tuple[Any, torch.nn.Module]:
    # 加载文档编码器。MedCPT 双塔结构中，入库必须使用 Article-Encoder。
    global _DOC_TOKENIZER, _DOC_MODEL
    if _DOC_TOKENIZER is None or _DOC_MODEL is None:
        print(f"Loading document encoder on {_DEVICE}: {DOCUMENT_MODEL_NAME}", file=sys.stderr)
        _DOC_TOKENIZER = AutoTokenizer.from_pretrained(DOCUMENT_MODEL_NAME)
        _DOC_MODEL = AutoModel.from_pretrained(DOCUMENT_MODEL_NAME).to(_DEVICE).eval()
    return _DOC_TOKENIZER, _DOC_MODEL


def _load_query_encoder() -> tuple[Any, torch.nn.Module]:
    # 加载查询编码器。检索查询必须使用 Query-Encoder，不能和文档编码器混用。
    global _QUERY_TOKENIZER, _QUERY_MODEL
    if _QUERY_TOKENIZER is None or _QUERY_MODEL is None:
        print(f"Loading query encoder on {_DEVICE}: {QUERY_MODEL_NAME}", file=sys.stderr)
        _QUERY_TOKENIZER = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        _QUERY_MODEL = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(_DEVICE).eval()
    return _QUERY_TOKENIZER, _QUERY_MODEL


def _encode_texts(texts: Sequence[str],tokenizer: Any,model: torch.nn.Module,max_length: int,batch_size: int,) -> np.ndarray:
    vectors: List[np.ndarray] = []
    clean_texts = [str(text or "") for text in texts]

    with torch.no_grad():
        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(_DEVICE)
            outputs = model(**encoded)
            cls_vec = outputs.last_hidden_state[:, 0, :]
            cls_vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)
            vectors.append(cls_vec.cpu().numpy().astype(np.float32))

    if not vectors:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32)
    return np.vstack(vectors)


def encode_documents(texts: Sequence[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> np.ndarray:
    # 使用 MedCPT Article-Encoder 编码待入库 chunk。
    tokenizer, model = _load_document_encoder()
    return _encode_texts(texts, tokenizer, model, DOCUMENT_MAX_LENGTH, batch_size)


def encode_query(text: str) -> np.ndarray:
    # 使用 MedCPT Query-Encoder 编码检索查询。
    tokenizer, model = _load_query_encoder()
    return _encode_texts([text], tokenizer, model, QUERY_MAX_LENGTH, 1)[0]


def create_collection(recreate: bool = False) -> None:
    # 创建集合和 payload 索引；recreate=True 时会先删除旧集合。
    client = get_qdrant_client()
    try:
        exists = client.collection_exists(COLLECTION_NAME)
        if exists and recreate:
            client.delete_collection(COLLECTION_NAME)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Created collection: {COLLECTION_NAME}")

        index_fields = {
            "year": models.PayloadSchemaType.INTEGER,
            "issuer": models.PayloadSchemaType.KEYWORD,
            "source": models.PayloadSchemaType.KEYWORD,
            "effective_date": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, schema in index_fields.items():
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if "already exists" not in message and "exists" not in message:
                    raise
        print("Payload indexes ready: year, issuer, source, effective_date")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"创建 Qdrant 集合失败: {exc}") from exc


def load_chunks(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    # 流式读取 jsonl，坏行会跳过并打印行号。
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过 JSON 解析失败行 {line_no}: {exc}", file=sys.stderr)
                continue
            if not isinstance(item, dict):
                print(f"跳过非对象行 {line_no}", file=sys.stderr)
                continue
            yield item


def _extract_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    if not match:
        return None
    return int(match.group(0))


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_medical_topic(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _make_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") or {}
    text = str(item.get("page_content") or "")
    published_date = str(metadata.get("published_date") or "")
    effective_date = _first_non_empty(metadata.get("effective_date"), published_date)

    return {
        "text": text,
        "title": str(metadata.get("title") or ""),
        "url": str(metadata.get("url") or ""),
        "issuer": str(metadata.get("issuer") or ""),
        "source": str(metadata.get("source") or ""),
        "published_date": published_date,
        "effective_date": effective_date,
        "year": _extract_year(effective_date),
        "medical_topic": _normalize_medical_topic(metadata.get("medical_topic")),
        "chunk_index": metadata.get("chunk_index"),
    }


def _make_point_id(payload: Dict[str, Any]) -> str:
    # 用溯源字段生成稳定 UUID，重复入库同一 chunk 会覆盖。
    text_digest = hashlib.sha1(str(payload.get("text") or "").encode("utf-8")).hexdigest()
    raw = "|".join(
        [
            str(payload.get("url") or ""),
            str(payload.get("title") or ""),
            str(payload.get("chunk_index") or ""),
            text_digest,
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _iter_batches(items: Iterable[Dict[str, Any]], batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest(jsonl_path: str | Path, batch_size: int = DEFAULT_UPSERT_BATCH_SIZE) -> None:
    # 读取 chunks，使用文档编码器向量化，并批量 upsert 到 Qdrant。
    create_collection(recreate=False)
    client = get_qdrant_client()
    total = 0

    for batch in _iter_batches(load_chunks(jsonl_path), batch_size):
        texts = [str(item.get("page_content") or "") for item in batch]
        valid_rows = [(item, text) for item, text in zip(batch, texts) if text.strip()]
        if not valid_rows:
            continue

        valid_items = [item for item, _ in valid_rows]
        valid_texts = [text for _, text in valid_rows]
        vectors = encode_documents(valid_texts)
        points = []

        for item, vector in zip(valid_items, vectors):
            payload = _make_payload(item)
            points.append(
                models.PointStruct(
                    id=_make_point_id(payload),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Qdrant upsert 失败，已处理 {total} 条后中断: {exc}") from exc

        total += len(points)
        print(f"已入库 {total} chunks")

    print(f"入库完成，总计 {total} chunks")


def _build_filter(
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    issuer: Optional[str] = None,
) -> Optional[models.Filter]:
    conditions: List[models.FieldCondition] = []
    if year is not None:
        conditions.append(
            models.FieldCondition(
                key="year",
                match=models.MatchValue(value=year),
            )
        )
    elif year_from is not None or year_to is not None:
        conditions.append(
            models.FieldCondition(
                key="year",
                range=models.Range(gte=year_from, lte=year_to),
            )
        )
    if issuer:
        conditions.append(
            models.FieldCondition(
                key="issuer",
                match=models.MatchValue(value=issuer),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def _qdrant_search(
    client: QdrantClient,
    query_vector: Sequence[float],
    query_filter: Optional[models.Filter],
    top_k: int,
) -> Any:
    """兼容 qdrant-client 新旧检索 API。"""
    if hasattr(client, "search"):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return result.points


def search(
    query: str,
    top_k: int = 5,
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    issuer: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # 查询向量化后在 Qdrant 中检索，返回 score 和 payload。
    client = get_qdrant_client()
    query_vector = encode_query(query).tolist()
    query_filter = _build_filter(year=year, year_from=year_from, year_to=year_to, issuer=issuer)

    try:
        hits = _qdrant_search(client, query_vector, query_filter, top_k)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qdrant 检索失败: {exc}") from exc

    results: List[Dict[str, Any]] = []
    for hit in hits:
        results.append(
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "payload": hit.payload or {},
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MedCPT 向量化与 Qdrant 入库/检索")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="读取 JSONL chunks 并写入 Qdrant")
    ingest_parser.add_argument("--input", default="outputs/data.jsonl", help="chunk JSONL 路径")
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_UPSERT_BATCH_SIZE, help="Qdrant upsert 批大小")
    ingest_parser.add_argument("--recreate", action="store_true", help="重建集合后再入库")

    search_parser = subparsers.add_parser("search", help="检索医学指南 chunks")
    search_parser.add_argument("query", help="查询文本")
    search_parser.add_argument("--top-k", type=int, default=5, help="返回条数")
    search_parser.add_argument("--year", type=int, default=None, help="仅检索指定 published_date 年份的数据")
    search_parser.add_argument("--year-from", type=int, default=None, help="仅检索该年份及之后的数据")
    search_parser.add_argument("--year-to", type=int, default=None, help="仅检索该年份及之前的数据")
    search_parser.add_argument("--issuer", default=None, help="按 issuer 精确过滤")

    args = parser.parse_args()

    if args.command == "ingest":
        if args.recreate:
            create_collection(recreate=True)
        ingest(args.input, batch_size=args.batch_size)
        return

    if args.command == "search":
        results = search(
            args.query,
            top_k=args.top_k,
            year=args.year,
            year_from=args.year_from,
            year_to=args.year_to,
            issuer=args.issuer,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

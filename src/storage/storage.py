from __future__ import annotations

"""Qdrant 入库与检索入口。

改造后主链路读取 `*.children.jsonl` 与 `*.parents.jsonl`：children collection 用于向量检索，
parents collection 用于按 parent_id 回查上下文。旧 `outputs/data.jsonl` 仅作为兼容输入保留。
"""

import argparse
import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = Any  # type: ignore[misc, assignment]
    models = None  # type: ignore[assignment]

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

CHILD_COLLECTION_NAME = os.getenv("QDRANT_CHILD_COLLECTION", "medical_guideline_children")
PARENT_COLLECTION_NAME = os.getenv("QDRANT_PARENT_COLLECTION", "medical_guideline_parents")
COLLECTION_NAME = CHILD_COLLECTION_NAME

DOCUMENT_MODEL_NAME = os.getenv("DOCUMENT_MODEL_NAME", "ncbi/MedCPT-Article-Encoder")
QUERY_MODEL_NAME = os.getenv("QUERY_MODEL_NAME", "ncbi/MedCPT-Query-Encoder")
VECTOR_SIZE = 768
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_UPSERT_BATCH_SIZE = 64
DOCUMENT_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 64

_DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
_DOC_TOKENIZER: Optional[Any] = None
_DOC_MODEL: Optional[Any] = None
_QUERY_TOKENIZER: Optional[Any] = None
_QUERY_MODEL: Optional[Any] = None


def _require_qdrant() -> None:
    """确认 qdrant-client 可用，否则给出安装提示。"""

    if models is None:
        raise RuntimeError("缺少 qdrant-client，请先安装依赖：pip install qdrant-client")


def _require_embedding_deps() -> None:
    """确认向量化依赖可用，否则给出安装提示。"""

    if torch is None or AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("缺少 torch/transformers，请先安装项目依赖后再入库或检索")


def get_qdrant_client() -> QdrantClient:
    """创建 Qdrant client，并把连接错误转成更清晰的提示。"""

    _require_qdrant()
    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        return client
    except Exception as exc:
        raise RuntimeError(f"无法连接 Qdrant: {QDRANT_URL}。请确认服务已启动，或设置 QDRANT_URL。原始错误: {exc}") from exc


def _load_document_encoder() -> tuple[Any, Any]:
    """加载 MedCPT Article-Encoder，入库向量必须使用文档编码器。"""

    global _DOC_TOKENIZER, _DOC_MODEL
    _require_embedding_deps()
    if _DOC_TOKENIZER is None or _DOC_MODEL is None:
        logger.info("Loading document encoder on %s: %s", _DEVICE, DOCUMENT_MODEL_NAME)
        _DOC_TOKENIZER = AutoTokenizer.from_pretrained(DOCUMENT_MODEL_NAME)
        _DOC_MODEL = AutoModel.from_pretrained(DOCUMENT_MODEL_NAME).to(_DEVICE).eval()
    return _DOC_TOKENIZER, _DOC_MODEL


def _load_query_encoder() -> tuple[Any, Any]:
    """加载 MedCPT Query-Encoder，检索查询必须使用查询编码器。"""

    global _QUERY_TOKENIZER, _QUERY_MODEL
    _require_embedding_deps()
    if _QUERY_TOKENIZER is None or _QUERY_MODEL is None:
        logger.info("Loading query encoder on %s: %s", _DEVICE, QUERY_MODEL_NAME)
        _QUERY_TOKENIZER = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        _QUERY_MODEL = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(_DEVICE).eval()
    return _QUERY_TOKENIZER, _QUERY_MODEL


def _encode_texts(texts: Sequence[str], tokenizer: Any, model: Any, max_length: int, batch_size: int) -> np.ndarray:
    """批量编码文本并做 L2 归一化。"""

    _require_embedding_deps()
    vectors: List[np.ndarray] = []
    clean_texts = [str(text or "") for text in texts]
    with torch.no_grad():
        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(_DEVICE)
            outputs = model(**encoded)
            cls_vec = outputs.last_hidden_state[:, 0, :]
            cls_vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)
            vectors.append(cls_vec.cpu().numpy().astype(np.float32))
    if not vectors:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32)
    return np.vstack(vectors)


def encode_documents(texts: Sequence[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> np.ndarray:
    """使用文档编码器编码待入库 chunk。"""

    tokenizer, model = _load_document_encoder()
    return _encode_texts(texts, tokenizer, model, DOCUMENT_MAX_LENGTH, batch_size)


def encode_query(text: str) -> np.ndarray:
    """使用查询编码器编码检索问题。"""

    tokenizer, model = _load_query_encoder()
    return _encode_texts([text], tokenizer, model, QUERY_MAX_LENGTH, 1)[0]


def create_collection(collection_name: str, recreate: bool = False) -> None:
    """创建单个 Qdrant collection 与常用 payload 索引。"""

    client = get_qdrant_client()
    try:
        exists = client.collection_exists(collection_name)
        if exists and recreate:
            client.delete_collection(collection_name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            logger.info("Created collection: %s", collection_name)

        index_fields = {
            "chunk_id": models.PayloadSchemaType.KEYWORD,
            "doc_id": models.PayloadSchemaType.KEYWORD,
            "parent_id": models.PayloadSchemaType.KEYWORD,
            "chunk_type": models.PayloadSchemaType.KEYWORD,
            "year": models.PayloadSchemaType.INTEGER,
            "issuer": models.PayloadSchemaType.KEYWORD,
            "source": models.PayloadSchemaType.KEYWORD,
            "effective_date": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, schema in index_fields.items():
            try:
                client.create_payload_index(collection_name=collection_name, field_name=field_name, field_schema=schema)
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" not in message and "exists" not in message:
                    raise
    except Exception as exc:
        raise RuntimeError(f"创建 Qdrant collection 失败: {exc}") from exc


def create_collections(recreate: bool = False) -> None:
    """创建 children 与 parents 两个 collection。"""

    create_collection(CHILD_COLLECTION_NAME, recreate=recreate)
    create_collection(PARENT_COLLECTION_NAME, recreate=recreate)


def load_chunks(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    """流式读取 JSONL，坏行会跳过并记录日志。"""

    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("跳过 JSON 解析失败行 %s: %s", line_no, exc)
                continue
            if not isinstance(item, dict):
                logger.warning("跳过非对象行 %s", line_no)
                continue
            yield item


def _extract_year(value: Any) -> Optional[int]:
    """从日期或任意字符串中提取年份。"""

    if value is None:
        return None
    match = re.search(r"(?:19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def _first_non_empty(*values: Any) -> str:
    """返回第一个非空字符串。"""

    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_medical_topic(value: Any) -> List[str]:
    """把 medical_topic 统一成字符串列表。"""

    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _payload_from_new_chunk(item: Dict[str, Any]) -> Dict[str, Any]:
    """从 schema.Chunk JSON 生成 Qdrant payload。"""

    provenance = item.get("provenance") or {}
    clinical = item.get("clinical") or {}
    published_date = str(provenance.get("published_date") or "")
    effective_date = _first_non_empty(provenance.get("effective_date"), published_date)
    return {
        "chunk_id": str(item.get("chunk_id") or ""),
        "doc_id": str(item.get("doc_id") or provenance.get("doc_id") or ""),
        "parent_id": item.get("parent_id"),
        "level": item.get("level"),
        "chunk_type": str(item.get("chunk_type") or ""),
        "text": str(item.get("text") or ""),
        "context": str(item.get("context") or ""),
        "clinical": clinical,
        "strength": clinical.get("strength"),
        "evidence_level": clinical.get("evidence_level"),
        "title": str(provenance.get("title") or ""),
        "url": str(provenance.get("url") or ""),
        "issuer": str(provenance.get("issuer") or ""),
        "source": str(provenance.get("source") or ""),
        "published_date": published_date,
        "effective_date": effective_date,
        "year": _extract_year(effective_date),
        "page": provenance.get("page"),
        "char_start": provenance.get("char_start"),
        "char_end": provenance.get("char_end"),
        "embedding_model": str(item.get("embedding_model") or DOCUMENT_MODEL_NAME),
    }


def _payload_from_old_chunk(item: Dict[str, Any]) -> Dict[str, Any]:
    """从旧 page_content/metadata JSON 生成兼容 payload。"""

    metadata = item.get("metadata") or {}
    text = str(item.get("page_content") or "")
    published_date = str(metadata.get("published_date") or "")
    effective_date = _first_non_empty(metadata.get("effective_date"), published_date)
    return {
        "chunk_id": "",
        "doc_id": str(metadata.get("doc_id") or ""),
        "parent_id": None,
        "level": None,
        "chunk_type": "legacy_chunk",
        "text": text,
        "context": "",
        "clinical": {
            "quality": metadata.get("quality"),
            "recommendation": metadata.get("recommendation"),
        },
        "strength": None,
        "evidence_level": metadata.get("quality"),
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


def _make_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    """兼容新 Chunk JSON 与旧 page_content/metadata JSON。"""

    if "text" in item and "chunk_type" in item:
        return _payload_from_new_chunk(item)
    return _payload_from_old_chunk(item)


def _make_point_id(payload: Dict[str, Any]) -> str:
    """用稳定字段生成 UUID，重复入库同一 chunk 会覆盖。"""

    primary = str(payload.get("chunk_id") or "")
    if primary:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, primary))
    text_digest = hashlib.sha1(str(payload.get("text") or "").encode("utf-8")).hexdigest()
    raw = "|".join([str(payload.get("url") or ""), str(payload.get("title") or ""), str(payload.get("chunk_index") or ""), text_digest])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _text_for_embedding(payload: Dict[str, Any]) -> str:
    """组合用于向量化的文本，child 携带 parent context，parent 使用自身文本。"""

    text = str(payload.get("text") or "")
    context = str(payload.get("context") or "")
    if payload.get("chunk_type") == "section_parent":
        return text or context
    return f"{context}\n{text}".strip() if context else text


def _iter_batches(items: Iterable[Dict[str, Any]], batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    """按固定大小切分迭代器，供批量 upsert 使用。"""

    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_file(jsonl_path: str | Path, collection_name: str, batch_size: int = DEFAULT_UPSERT_BATCH_SIZE) -> int:
    """读取一个 JSONL 文件，向量化后批量 upsert 到指定 collection。"""

    create_collection(collection_name, recreate=False)
    client = get_qdrant_client()
    total = 0

    for batch in _iter_batches(load_chunks(jsonl_path), batch_size):
        payloads = [_make_payload(item) for item in batch]
        valid_payloads = [payload for payload in payloads if str(payload.get("text") or "").strip()]
        if not valid_payloads:
            continue

        vectors = encode_documents([_text_for_embedding(payload) for payload in valid_payloads])
        points = [
            models.PointStruct(id=_make_point_id(payload), vector=vector.tolist(), payload=payload)
            for payload, vector in zip(valid_payloads, vectors)
        ]
        try:
            client.upsert(collection_name=collection_name, points=points)
        except Exception as exc:
            raise RuntimeError(f"Qdrant upsert 失败，已处理 {total} 条后中断: {exc}") from exc

        total += len(points)
        logger.info("已入库 %s -> %s: %s chunks", jsonl_path, collection_name, total)
    return total


def ingest(
    children_path: str | Path,
    parents_path: str | Path = "",
    batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    recreate: bool = False,
) -> Dict[str, int]:
    """主链路入库：children 入检索 collection，parents 入上下文 collection。"""

    if recreate:
        create_collections(recreate=True)
    counts: Dict[str, int] = {}
    if parents_path:
        counts["parents"] = ingest_file(parents_path, PARENT_COLLECTION_NAME, batch_size=batch_size)
    counts["children"] = ingest_file(children_path, CHILD_COLLECTION_NAME, batch_size=batch_size)
    return counts


def _build_filter(
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    issuer: Optional[str] = None,
) -> Optional[models.Filter]:
    """根据可选条件构造 Qdrant payload filter。"""

    conditions: List[models.FieldCondition] = []
    if year is not None:
        conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))
    elif year_from is not None or year_to is not None:
        conditions.append(models.FieldCondition(key="year", range=models.Range(gte=year_from, lte=year_to)))
    if issuer:
        conditions.append(models.FieldCondition(key="issuer", match=models.MatchValue(value=issuer)))
    return models.Filter(must=conditions) if conditions else None


def _qdrant_search(client: QdrantClient, query_vector: Sequence[float], query_filter: Optional[models.Filter], top_k: int) -> Any:
    """兼容 qdrant-client 新旧检索 API。"""

    if hasattr(client, "search"):
        return client.search(
            collection_name=CHILD_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
    result = client.query_points(
        collection_name=CHILD_COLLECTION_NAME,
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
    """在 children collection 中检索，返回 score 与 payload。"""

    client = get_qdrant_client()
    query_vector = encode_query(query).tolist()
    query_filter = _build_filter(year=year, year_from=year_from, year_to=year_to, issuer=issuer)
    try:
        hits = _qdrant_search(client, query_vector, query_filter, top_k)
    except Exception as exc:
        raise RuntimeError(f"Qdrant 检索失败: {exc}") from exc

    return [{"id": str(hit.id), "score": float(hit.score), "payload": hit.payload or {}} for hit in hits]


def get_parent(parent_id: str) -> Optional[Dict[str, Any]]:
    """按 parent chunk_id 从 parents collection 回查上下文。"""

    client = get_qdrant_client()
    query_filter = models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=parent_id))])
    records, _ = client.scroll(collection_name=PARENT_COLLECTION_NAME, scroll_filter=query_filter, limit=1, with_payload=True)
    if not records:
        return None
    return records[0].payload or {}


def parse_args() -> argparse.Namespace:
    """解析 storage CLI 参数。"""

    parser = argparse.ArgumentParser(description="MedCPT 向量化与 Qdrant 入库/检索")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="读取 parent/child JSONL 并写入 Qdrant")
    ingest_parser.add_argument("--children", default="", help="children JSONL 路径，主链路必填")
    ingest_parser.add_argument("--parents", default="", help="parents JSONL 路径")
    ingest_parser.add_argument("--input", default="", help="DEPRECATED: 旧 page_content/metadata JSONL 路径")
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_UPSERT_BATCH_SIZE)
    ingest_parser.add_argument("--recreate", action="store_true", help="重建两个 collection 后再入库")

    search_parser = subparsers.add_parser("search", help="检索医学指南 child chunks")
    search_parser.add_argument("query", help="查询文本")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--year", type=int, default=None)
    search_parser.add_argument("--year-from", type=int, default=None)
    search_parser.add_argument("--year-to", type=int, default=None)
    search_parser.add_argument("--issuer", default=None)

    parent_parser = subparsers.add_parser("parent", help="按 parent_id 回查 parent payload")
    parent_parser.add_argument("parent_id", help="parent chunk_id")

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI 入口：执行入库、检索或 parent 回查。"""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")

    if args.command == "ingest":
        children = args.children or args.input
        if not children:
            raise ValueError("请提供 --children；旧格式可临时使用 --input")
        counts = ingest(children, parents_path=args.parents, batch_size=args.batch_size, recreate=args.recreate)
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        return

    if args.command == "search":
        results = search(args.query, top_k=args.top_k, year=args.year, year_from=args.year_from, year_to=args.year_to, issuer=args.issuer)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.command == "parent":
        print(json.dumps(get_parent(args.parent_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

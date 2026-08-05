# chunk 构建脚本共用的 JSONL 与元数据 IO 工具。
#
# 模块职责：
# - append_jsonl：原子追加（无重试，但 ensure_parent 保证父目录存在）；
# - read_document_meta：从 metadata_dir/<stem>.json 读 DocumentMeta，缺字段回退到 markdown 路径；
# - group_by_doc：按 doc_id 把 records 分组（常用于 section→atomic/table 的归属）；
# - default_warnings_path：在 output 旁边创建 chunk_build_warnings.jsonl，便于人工 review。
"""Small JSONL and metadata helpers for chunk build scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.guideline_chunking.models import DocumentMeta
from src.utils.io import ensure_parent, read_jsonl, write_jsonl


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """追加 JSONL 记录（ensure_parent 兜底创建父目录）。"""
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_document_meta(md_path: Path, metadata_dir: str | Path | None = None) -> DocumentMeta:
    """从 metadata_dir/<md_path.stem>.json 读 DocumentMeta；缺字段回退到 md_path.stem。

    - 缺 metadata_dir / 文件不存在 → payload 为空，doc_id 退化为 md_path.stem；
    - payload 字段缺值时使用空 / None，由 DocumentMeta 自行决定默认值。
    """
    payload: dict[str, Any] = {}
    if metadata_dir:
        meta_path = Path(metadata_dir) / f"{md_path.stem}.json"
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    return DocumentMeta(
        doc_id=str(payload.get("doc_id") or md_path.stem),
        title=payload.get("title"),
        publisher=payload.get("publisher"),
        language=payload.get("language"),
        document_type=payload.get("document_type"),
        publication_date=payload.get("publication_date"),
        last_updated_date=payload.get("last_updated_date"),
        source_url=payload.get("source_url"),
    )


def group_by_doc(records: Iterable[Any]) -> dict[str, list[Any]]:
    """按 record.doc_id 分组，便于跨 chunk 类型归属。"""
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.doc_id, []).append(record)
    return grouped


def default_warnings_path(output_path: str | Path) -> Path:
    """警告日志默认位置：output 同目录下 chunk_build_warnings.jsonl。"""
    return Path(output_path).parent / "chunk_build_warnings.jsonl"
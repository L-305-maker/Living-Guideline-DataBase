# 在不改变 chunk 边界的前提下，按最新文档科室刷新 chunk 自身的科室标签。
#
# 关键约束：
# - chunk 标签必须是文档标签的子集（不能跳出父级分类层级）；
# - 默认只重新标 unknown 的 chunk；force_documents 集合强制全量重算。
# - 输出聚合文件 all_chunks.jsonl 在 chunks 全部改完后原子替换。
"""Refresh chunk department labels without changing chunk boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.clinical_department import UNKNOWN_DEPARTMENT, classify_chunk_departments
from src.utils.io import DATA_DIR, read_jsonl


def load_document_departments(path: str | Path) -> dict[str, list[str]]:
    """读 documents.jsonl 返回 doc_id → labels 列表。

    兼容两种来源：
    - clinical_departments（list，preferred）
    - clinical_department（str，fallback）
    """
    departments: dict[str, list[str]] = {}
    for record in read_jsonl(path):
        labels = [str(item).strip() for item in record.get("clinical_departments") or [] if str(item).strip()]
        if not labels:
            labels = [str(record.get("clinical_department") or UNKNOWN_DEPARTMENT)]
        departments[str(record["doc_id"])] = labels
    return departments


def relabel_chunk(record: dict[str, Any], parent: list[str], force: bool = False) -> tuple[dict[str, Any], bool, bool]:
    """对单个 chunk 计算新标签。

    决策表：
    - force=True：忽略旧值，按 classify_chunk_departments 重算；
    - fallback（旧值=[未知] 且父级不是未知）：直接取父级第一个科室，避免空白；
    - 旧值已是父级子集：保持原状；
    - 其它：调 classify_chunk_departments 重新收敛。

    返回 (新记录, 是否发生变化, 是否走了 fallback 路径)。
    """
    current = [str(item).strip() for item in record.get("clinical_departments") or [] if str(item).strip()]
    fallback = current == [UNKNOWN_DEPARTMENT] and UNKNOWN_DEPARTMENT not in parent
    if force:
        result = classify_chunk_departments(
            record.get("section_path") or [],
            str(record.get("content") or ""),
            parent,
            str(record.get("chunk_type") or ""),
        )
        labels = list(result["clinical_departments"])
    elif fallback:
        labels = [parent[0]]
    elif current and set(current).issubset(parent):
        labels = current
    else:
        result = classify_chunk_departments(
            record.get("section_path") or [],
            str(record.get("content") or ""),
            parent,
            str(record.get("chunk_type") or ""),
        )
        labels = list(result["clinical_departments"])

    # 兼容老记录里有 clinical_department / department_scope 单数字段，重写时移除。
    changed = current != labels or "clinical_department" in record or "department_scope" in record
    record = dict(record)
    record["clinical_departments"] = labels
    record.pop("clinical_department", None)
    record.pop("department_scope", None)
    return record, changed, fallback


def relabel_all(
    document_manifest: str | Path = DATA_DIR / "documents.jsonl",
    chunks_dir: str | Path = DATA_DIR / "chunks",
    force_documents: set[str] | None = None,
) -> dict[str, int | str]:
    """逐文件重标 chunk 标签，并统一重建 all_chunks.jsonl。

    关键设计：
    - 每个 chunk 文件用 .tmp 临时写入，原文件保留到 rename 成功；
    - 同步往 aggregate_tmp 写一份聚合，循环结束后再原子替换；
    - 校验 chunk 标签是 doc 标签的子集（防漂移）：
      if not labels.issubset(documents[doc_id]) → ValueError；
    - 任意一步出错 unlink 临时文件并 raise，保证可重入。
    """
    documents = load_document_departments(document_manifest)
    directory = Path(chunks_dir)
    chunk_files = sorted(path for path in directory.glob("*.jsonl") if path.name != "all_chunks.jsonl")
    aggregate_path = directory / "all_chunks.jsonl"
    aggregate_tmp = directory / "all_chunks.jsonl.tmp"
    chunks = changed = fallback = files = 0
    documents_with_chunks: set[str] = set()

    try:
        with aggregate_tmp.open("w", encoding="utf-8", newline="\n") as aggregate:
            for path in chunk_files:
                file_tmp = path.with_suffix(path.suffix + ".tmp")
                try:
                    with path.open("r", encoding="utf-8-sig") as source, file_tmp.open(
                        "w", encoding="utf-8", newline="\n"
                    ) as target:
                        for line_no, line in enumerate(source, start=1):
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                            doc_id = str(record.get("doc_id") or "")
                            if doc_id not in documents:
                                raise KeyError(f"Chunk {record.get('chunk_id')} has no document metadata")
                            record, was_changed, used_fallback = relabel_chunk(
                                record, documents[doc_id], doc_id in (force_documents or set())
                            )
                            labels = set(record["clinical_departments"])
                            # 约束：chunk 标签是文档标签的子集。
                            if not labels.issubset(documents[doc_id]):
                                raise ValueError(f"Chunk {record.get('chunk_id')} has labels outside its document")
                            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                            target.write(encoded)
                            aggregate.write(encoded)
                            chunks += 1
                            changed += was_changed
                            fallback += used_fallback
                            documents_with_chunks.add(doc_id)
                    file_tmp.replace(path)
                except Exception:
                    file_tmp.unlink(missing_ok=True)
                    raise
                files += 1
                if files % 500 == 0:
                    print(json.dumps({"files": files, "chunks": chunks, "changed": changed}), flush=True)
        aggregate_tmp.replace(aggregate_path)
    except Exception:
        aggregate_tmp.unlink(missing_ok=True)
        raise

    return {
        "documents": len(documents),
        "documents_with_chunks": len(documents_with_chunks),
        "chunk_files": files,
        "chunks": chunks,
        "changed_chunks": changed,
        "fallback_to_primary": fallback,
        "forced_documents": len(force_documents or set()),
        "chunks_dir": str(directory),
    }


def load_force_documents(path: str | Path | None) -> set[str]:
    """读 reclassification 报告（来自 reclassify_unknown_departments）取出 identified doc_ids。

    返回空集合表示不强制；非空集合的 doc_id 在重标时走 force=True 路径。
    """
    if not path:
        return set()
    return {
        str(record["doc_id"])
        for record in read_jsonl(path)
        if record.get("status") == "identified"
    }


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default=str(DATA_DIR / "documents.jsonl"))
    parser.add_argument("--chunks-dir", default=str(DATA_DIR / "chunks"))
    parser.add_argument("--force-documents", help="Reclassification report JSONL; identified doc_ids are forced.")
    args = parser.parse_args()
    force_documents = load_force_documents(args.force_documents)
    print(json.dumps(relabel_all(args.documents, args.chunks_dir, force_documents), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
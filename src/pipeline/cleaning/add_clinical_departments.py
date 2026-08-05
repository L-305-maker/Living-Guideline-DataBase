# 把 clinical_department 字段回填到 Markdown front-matter 与 JSONL 派生产物。
#
# 应用场景：早期文档的 front-matter 缺少 clinical_department，本模块用 doc 级
# 规则引擎（classify_clinical_department）补齐；同时同步 documents.jsonl /
# documents_raw.jsonl / sections/*.jsonl / chunks/*.jsonl，使所有派生产物一致。
#
# 安全策略：
# - 默认 force=True，重写已有 clinical_department；--no-force 时保留原值；
# - --reuse-existing：用 markdown 已有的分类作为种子，避免每次跑都重算；
# - 临时文件 + os.replace 原子替换，避免中断留下半成品。
"""Backfill clinical department metadata into Markdown and JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.clinical_department import classify_clinical_department
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.io import DATA_DIR, read_jsonl
from src.utils.metadata import extract_abstract


def helper_write_text_atomic(path: Path, text: str) -> None:
    """原子写文本文件：先写 tmp，再 os.replace。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def helper_write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> int:
    """原子写整个 JSONL：先把 records 全部序列化到 tmp，再 os.replace。"""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    return len(records)


def helper_rewrite_jsonl_stream(path: Path, department_by_doc: dict[str, str]) -> tuple[int, int]:
    """流式重写 JSONL（不一次性加载全部记录），按 doc_id 改写 clinical_department。

    返回 (总行数, 改写行数)；其它字段原样保留。
    """
    if not path.exists():
        return 0, 0
    tmp = path.with_name(path.name + ".tmp")
    total = 0
    changed = 0
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            department = department_by_doc.get(rec.get("doc_id", ""), rec.get("clinical_department") or "未分类")
            if rec.get("clinical_department") != department:
                rec["clinical_department"] = department
                changed += 1
            dst.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            total += 1
    os.replace(tmp, path)
    return total, changed


def helper_read_front_matter_metadata(path: Path) -> dict[str, str]:
    """只解析 front-matter 不读正文，避免大文件 IO。

    手动逐行读取，仅在 front-matter 闭合后再 parse_front_matter。
    """
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        if first.strip() != "---":
            return {}
        for line in handle:
            if line.strip() == "---":
                break
            lines.append(line)
    metadata, _body = parse_front_matter("---\n" + "".join(lines) + "---\n")
    return metadata


def load_existing_departments(markdown_dirs: list[Path]) -> dict[str, str]:
    """从已有 Markdown front-matter 抽取 doc_id → clinical_department 映射。

    用作 --reuse-existing 的种子来源；旧文档已有标签时直接复用，避免重分类带来的抖动。
    """
    department_by_doc: dict[str, str] = {}
    for directory in markdown_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            metadata = helper_read_front_matter_metadata(path)
            doc_id = metadata.get("id")
            department = metadata.get("clinical_department")
            if doc_id and department:
                department_by_doc[doc_id] = department
    return department_by_doc


def classify_markdown(
    path: Path,
    force: bool = True,
    seed_departments: dict[str, str] | None = None,
) -> tuple[str | None, str | None, bool]:
    """对单个 Markdown 决定是否需要重写 clinical_department。

    决策：
    - 不传 id 的文档直接跳过（无 doc_id 无法定位）；
    - 已有分类且 not force：保留；
    - 已有分类且 force，但与种子一致：保留；
    - 其它情况：调 classify_clinical_department 重算，写回 front-matter。

    返回 (doc_id, department, changed)。
    """
    fast_metadata = helper_read_front_matter_metadata(path)
    doc_id = fast_metadata.get("id")
    if not doc_id:
        return None, None, False
    existing = fast_metadata.get("clinical_department")
    seeded = (seed_departments or {}).get(doc_id)
    if existing and not force:
        return doc_id, existing, False
    if seeded and existing == seeded:
        return doc_id, seeded, False

    markdown = path.read_text(encoding="utf-8", errors="replace")
    metadata, body = parse_front_matter(markdown)
    doc_id = metadata.get("id")
    if not doc_id:
        return None, None, False
    existing = metadata.get("clinical_department")
    if existing and not force:
        department = existing
    elif seeded:
        department = seeded
    else:
        department = classify_clinical_department(
            metadata.get("title", ""),
            extract_abstract(body),
            body,
        )
    changed = metadata.get("clinical_department") != department
    if changed:
        metadata["clinical_department"] = department
        helper_write_text_atomic(path, dump_front_matter(metadata, body))
    return doc_id, department, changed


def update_markdown_dirs(
    markdown_dirs: list[Path],
    force: bool = True,
    seed_departments: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """扫描 markdown_raw/ 和 markdown_clean/ 两个目录，输出 doc_id → department 与统计。

    注意 markdown_clean 写回会覆盖清洗后的产物，所以 force=False 时尽量不动它；
    仅当需要补齐缺失字段时才改写。
    """
    department_by_doc: dict[str, str] = {}
    stats: dict[str, Any] = {"markdown_files": 0, "markdown_changed": 0, "markdown_skipped": 0}
    for directory in markdown_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            doc_id, department, changed = classify_markdown(path, force=force, seed_departments=seed_departments)
            if not doc_id or not department:
                stats["markdown_skipped"] += 1
                continue
            department_by_doc[doc_id] = department
            stats["markdown_files"] += 1
            if changed:
                stats["markdown_changed"] += 1
    return department_by_doc, stats


def update_manifest(path: Path, department_by_doc: dict[str, str]) -> tuple[int, int]:
    """重写单个 manifest JSONL，按 doc_id 改写 clinical_department。"""
    if not path.exists():
        return 0, 0
    records: list[dict[str, Any]] = []
    changed = 0
    for rec in read_jsonl(path):
        department = department_by_doc.get(rec.get("doc_id", ""), rec.get("clinical_department") or "未分类")
        if rec.get("clinical_department") != department:
            rec["clinical_department"] = department
            changed += 1
        records.append(rec)
    helper_write_jsonl_atomic(path, records)
    return len(records), changed


def update_partitioned_jsonl(directory: Path, department_by_doc: dict[str, str]) -> tuple[int, int, int]:
    """处理 sections/ 与 chunks/ 这类按文档分片的 JSONL。

    跳过 chunks/all_chunks.jsonl 聚合文件（由 relabel_chunk_departments 单独处理）。
    返回 (文件数, 总行数, 改写行数)。
    """
    if not directory.exists():
        return 0, 0, 0
    files = 0
    total = 0
    changed = 0
    for path in sorted(directory.glob("*.jsonl")):
        if path.name == "all_chunks.jsonl":
            continue
        rows, updates = helper_rewrite_jsonl_stream(path, department_by_doc)
        files += 1
        total += rows
        changed += updates
    return files, total, changed


def backfill(data_dir: str | Path = DATA_DIR, force: bool = True, reuse_existing: bool = False) -> dict[str, Any]:
    """主入口：把 clinical_department 回填到所有 Markdown 与 JSONL 派生产物。

    关键场景：
    - 旧产物有 clinical_department 但与新规则冲突 → force=True 时重写；
    - 旧产物字段缺失 → 一律补齐；
    - all_chunks.jsonl 在被占用时（PermissionError on Windows）会记为 pending_replace，
      由后续脚本手动处理。
    """
    data_path = Path(data_dir)
    markdown_dirs = [data_path / "markdown_raw", data_path / "markdown_clean"]
    seed_departments = load_existing_departments(markdown_dirs) if reuse_existing else {}
    department_by_doc, stats = update_markdown_dirs(markdown_dirs, force=force, seed_departments=seed_departments)

    manifest_total, manifest_changed = update_manifest(data_path / "documents.jsonl", department_by_doc)
    raw_manifest_total, raw_manifest_changed = update_manifest(data_path / "documents_raw.jsonl", department_by_doc)
    section_files, section_rows, section_changed = update_partitioned_jsonl(data_path / "sections", department_by_doc)
    chunk_files, chunk_rows, chunk_changed = update_partitioned_jsonl(data_path / "chunks", department_by_doc)
    all_chunks_pending_replace = False
    try:
        all_chunk_rows, all_chunk_changed = helper_rewrite_jsonl_stream(
            data_path / "chunks" / "all_chunks.jsonl", department_by_doc
        )
    except PermissionError:
        # Windows 上 chunks/all_chunks.jsonl 可能被并发进程占用；
        # 记为 pending_replace 让用户知道聚合文件未刷新，需要后续单独跑。
        all_chunks_pending_replace = True
        tmp_path = data_path / "chunks" / "all_chunks.jsonl.tmp"
        all_chunk_rows = 0
        all_chunk_changed = 0
        if tmp_path.exists():
            with tmp_path.open("r", encoding="utf-8") as handle:
                all_chunk_rows = sum(1 for line in handle if line.strip())

    stats.update(
        {
            "documents": len(department_by_doc),
            "documents_manifest_rows": manifest_total,
            "documents_manifest_changed": manifest_changed,
            "raw_documents_manifest_rows": raw_manifest_total,
            "raw_documents_manifest_changed": raw_manifest_changed,
            "section_files": section_files,
            "section_rows": section_rows,
            "section_changed": section_changed,
            "chunk_files": chunk_files,
            "chunk_rows": chunk_rows,
            "chunk_changed": chunk_changed,
            "all_chunk_rows": all_chunk_rows,
            "all_chunk_changed": all_chunk_changed,
            "all_chunks_pending_replace": all_chunks_pending_replace,
            "department_distribution": dict(Counter(department_by_doc.values()).most_common()),
        }
    )
    return stats


def main() -> None:
    """CLI 入口：
    --no-force 保留已有 clinical_department；
    --reuse-existing 用 Markdown 已有的分类作为种子，避免每次重算。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--no-force", action="store_true", help="Keep existing clinical_department values.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing Markdown classifications by doc_id.")
    args = parser.parse_args()
    print(json.dumps(backfill(args.data_dir, force=not args.no_force, reuse_existing=args.reuse_existing), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
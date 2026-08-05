# 文件系统与 JSONL I/O 辅助。
#
# 模块职责：
# - WORKSPACE_ROOT / DATA_DIR：项目根与默认数据目录常量；
# - ensure_dir / ensure_parent：递归创建目录（含父目录）；
# - read_jsonl / write_jsonl / append_jsonl：流式 JSONL 读写；
# - relative_to_workspace：把绝对路径转成相对项目根的字符串；
# - iter_markdown_files：扫描目录下所有 .md 文件（排序）。
"""Filesystem and JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


# 项目根（src/utils/io.py 的父级父级 = 项目根）+ 默认数据目录。
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data" / "evidence"


def ensure_dir(path: str | Path) -> Path:
    """递归创建目录（含父目录），返回路径本身。"""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent(path: str | Path) -> Path:
    """仅创建父目录，常用于 JSONL 写入前的目录准备。"""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """流式读取 JSONL：UTF-8-sig（容忍 BOM）、跳过空行、JSON 错误抛 ValueError 含行号。"""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """写 JSONL：ensure_ascii=False 保留中文；separators=(",", ":") 节省体积。"""
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """追加单行 JSONL 到已有文件（无去重逻辑）。"""
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def relative_to_workspace(path: str | Path) -> str:
    """把绝对路径转成相对 WORKSPACE_ROOT 的字符串；不在项目根下时返回绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def iter_markdown_files(directory: str | Path) -> Iterator[Path]:
    """排序遍历目录下所有 .md 文件，便于可复现的处理顺序。"""
    yield from sorted(Path(directory).glob("*.md"))
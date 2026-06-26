from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List


logger = logging.getLogger(__name__)


# 流式读取 JSONL，适合 data.jsonl 这类大文件。
def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSONL，适合 origin、blocks、candidates 这类可能很大的流水线文件。"""
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %s JSON parse error in %s: %s", line_no, jsonl_path, exc)
                continue
            if isinstance(item, dict):
                yield item


# 小文件一次性读取，大文件优先使用 iter_jsonl。
def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """一次性读取小型 JSONL；大文件应优先使用 iter_jsonl，避免内存占用过高。"""
    return list(iter_jsonl(path))


# 写入单条 JSONL 记录。
def write_jsonl_obj(handle: Any, item: Dict[str, Any]) -> None:
    """向已打开的文件句柄写入一条 JSONL 记录，供流式写入场景复用。"""
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    """写入一组字典记录到 JSONL 文件，并返回实际写出的记录数。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_obj(handle, row)
            count += 1
    return count


# 保存旧版 Document-like 对象，兼容现有清洗/切分代码。
def save_jsonl(result: Iterable[Any], output_path: str | Path) -> None:
    """保存旧版 Document-like 对象为 page_content/metadata 结构，兼容历史切分脚本。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for doc in result:
            write_jsonl_obj(
                f,
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )

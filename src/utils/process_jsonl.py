from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List


logger = logging.getLogger(__name__)


# 流式读取 JSONL，适合 data.jsonl 这类大文件。
def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """Stream JSONL records without loading large files into memory."""
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as f:
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
    """Read a JSONL file into memory. Prefer iter_jsonl for large files."""
    return list(iter_jsonl(path))


# 写入单条 JSONL 记录。
def write_jsonl_obj(handle: Any, item: Dict[str, Any]) -> None:
    """Write one JSON-serializable dict to an opened JSONL file handle."""
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")


# 保存旧版 Document-like 对象，兼容现有清洗/切分代码。
def save_jsonl(result: Iterable[Any], output_path: str | Path) -> None:
    """Save Document-like objects as the existing page_content/metadata JSONL shape."""
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

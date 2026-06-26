from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skip malformed JSONL line %s in %s: %s", line_no, jsonl_path, exc)
                continue
            if isinstance(item, dict):
                yield item


def append_jsonl(path: str | Path, item: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


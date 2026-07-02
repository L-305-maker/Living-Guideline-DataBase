"""存储层文件：定义 PostgreSQL schema、行映射、入库流程和完整性检查。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


JsonDict = Dict[str, Any]
DEFAULT_BATCH_SIZE = 500


def sql_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Illegal SQL identifier: {name}")
    return name


def json_obj(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def json_list(value: Any) -> str:
    return json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_jsonl(path: str | Path) -> Iterator[JsonDict]:
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON at {jsonl_path}:{line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                yield item


def batches(items: Iterable[JsonDict], batch_size: int) -> Iterator[List[JsonDict]]:
    batch: List[JsonDict] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def execute_many(conn: Any, sql: str, rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(sql, list(rows))
    return len(rows)


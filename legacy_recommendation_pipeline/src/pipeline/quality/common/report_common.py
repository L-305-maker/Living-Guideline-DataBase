"""质量评估文件：提供 goldset、审计指标和质量报告能力，用于评估候选结果是否可靠。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

from src.common.extraction_common import JsonDict
from src.common.process_jsonl import write_jsonl


def counts_by(rows: Iterable[JsonDict], field: str, default: str = "unknown") -> dict[str, int]:
    return dict(Counter(str(row.get(field) or default) for row in rows))


def truncate(value: object, max_chars: int) -> str:
    return str(value or "")[:max_chars]


def append_limited(groups: dict[str, list[JsonDict]], label: str, row: JsonDict, limit: int) -> None:
    if len(groups[label]) < limit:
        groups[label].append(row)


def flatten_groups(groups: dict[str, list[JsonDict]]) -> list[JsonDict]:
    samples: list[JsonDict] = []
    for label in sorted(groups):
        samples.extend(groups[label])
    return samples


def limited_groups() -> dict[str, list[JsonDict]]:
    return defaultdict(list)


def write_report(path: str | Path, report: JsonDict) -> None:
    write_jsonl(path, [report])


def sample_by_labels(
    rows: Iterable[JsonDict],
    labeler: Callable[[JsonDict], list[str]],
    builder: Callable[[JsonDict, str], JsonDict],
    per_group: int,
) -> list[JsonDict]:
    groups = limited_groups()
    for row in rows:
        for label in labeler(row):
            append_limited(groups, label, builder(row, label), per_group)
    return flatten_groups(groups)


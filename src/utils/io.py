"""Filesystem and JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data" / "evidence"


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
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
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def relative_to_workspace(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(resolved)



def iter_markdown_files(directory: str | Path) -> Iterator[Path]:
    yield from sorted(Path(directory).glob("*.md"))

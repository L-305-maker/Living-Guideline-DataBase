"""Rewrite retrieval JSONL artifacts to their fixed minimal contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from src.pipeline.cleaning.chunker import compact_chunk_record
from src.retrieval.document_repr.builder import compact_document_card, compact_document_view
from src.utils.io import DATA_DIR


def helper_rewrite(path: Path, projector: Callable[[dict[str, Any]], dict[str, Any]]) -> tuple[int, int, int]:
    before = path.stat().st_size
    temporary = path.with_suffix(path.suffix + ".compact.tmp")
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig") as source, temporary.open("w", encoding="utf-8", newline="\n") as target:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                target.write(json.dumps(projector(record), ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count, before, path.stat().st_size


def compact_artifacts(data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    # 所有 JSONL 先写临时文件再原子替换，任一步失败都要清理临时产物并保留原文件。
    root = Path(data_dir)
    card_count, card_before, card_after = helper_rewrite(root / "document_cards.jsonl", compact_document_card)
    view_count, view_before, view_after = helper_rewrite(root / "document_views.jsonl", compact_document_view)

    chunks_dir = root / "chunks"
    aggregate = chunks_dir / "all_chunks.jsonl"
    aggregate_tmp = aggregate.with_suffix(aggregate.suffix + ".compact.tmp")
    chunk_count = chunk_before = chunk_after = files = 0
    try:
        with aggregate_tmp.open("w", encoding="utf-8", newline="\n") as combined:
            for path in sorted(item for item in chunks_dir.glob("*.jsonl") if item.name != aggregate.name):
                before = path.stat().st_size
                temporary = path.with_suffix(path.suffix + ".compact.tmp")
                local_count = 0
                try:
                    with path.open("r", encoding="utf-8-sig") as source, temporary.open("w", encoding="utf-8", newline="\n") as target:
                        for line_no, line in enumerate(source, start=1):
                            if not line.strip():
                                continue
                            try:
                                record = compact_chunk_record(json.loads(line))
                            except json.JSONDecodeError as exc:
                                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                            target.write(encoded)
                            combined.write(encoded)
                            local_count += 1
                    temporary.replace(path)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
                chunk_count += local_count
                chunk_before += before
                chunk_after += path.stat().st_size
                files += 1
                if files % 500 == 0:
                    print(json.dumps({"chunk_files": files, "chunks": chunk_count}), flush=True)
        aggregate_tmp.replace(aggregate)
    except Exception:
        aggregate_tmp.unlink(missing_ok=True)
        raise

    return {
        "document_cards": {"count": card_count, "before_bytes": card_before, "after_bytes": card_after},
        "document_views": {"count": view_count, "before_bytes": view_before, "after_bytes": view_after},
        "chunks": {"count": chunk_count, "files": files, "before_bytes": chunk_before, "after_bytes": chunk_after},
        "aggregate_bytes": aggregate.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    print(json.dumps(compact_artifacts(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

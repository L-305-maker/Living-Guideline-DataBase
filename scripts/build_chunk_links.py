from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guideline_chunking.chunk_linker import build_chunk_links
from src.guideline_chunking.io_utils import read_jsonl, write_jsonl
from src.guideline_chunking.models import AtomicChunk, SectionChunk, TableChunk, from_dict, to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", required=True)
    parser.add_argument("--atomic", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sections = [from_dict(SectionChunk, record) for record in read_jsonl(args.sections)]
    atomic = [from_dict(AtomicChunk, record) for record in read_jsonl(args.atomic)]
    tables = [from_dict(TableChunk, record) for record in read_jsonl(args.tables)]
    links = build_chunk_links(sections, atomic, tables)
    write_jsonl(args.output, [to_dict(link) for link in links])
    print(f"wrote {len(links)} chunk links to {args.output}")


if __name__ == "__main__":
    main()

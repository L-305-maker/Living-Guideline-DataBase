from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guideline_chunking.io_utils import append_jsonl, default_warnings_path, group_by_doc, read_jsonl, write_jsonl
from src.guideline_chunking.models import ChunkBuildWarning, ParsedBlock, SectionChunk, from_dict, meta_from_block, to_dict
from src.guideline_chunking.table_chunker import build_table_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--sections", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warnings-output")
    args = parser.parse_args()

    blocks = [from_dict(ParsedBlock, record) for record in read_jsonl(args.blocks)]
    sections = [from_dict(SectionChunk, record) for record in read_jsonl(args.sections)]
    sections_by_doc = group_by_doc(sections)
    warnings: list[ChunkBuildWarning] = []
    chunks = []
    for doc_blocks in group_by_doc(blocks).values():
        chunks.extend(build_table_chunks(doc_blocks, sections_by_doc.get(doc_blocks[0].doc_id, []), meta_from_block(doc_blocks[0]), warnings))
    write_jsonl(args.output, [to_dict(chunk) for chunk in chunks])
    if warnings:
        append_jsonl(args.warnings_output or default_warnings_path(args.output), [to_dict(warning) for warning in warnings])
    print(f"wrote {len(chunks)} table chunks to {args.output}")


if __name__ == "__main__":
    main()

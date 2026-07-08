from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guideline_chunking.io_utils import group_by_doc, read_jsonl, write_jsonl
from src.guideline_chunking.models import ParsedBlock, from_dict, meta_from_block, to_dict
from src.guideline_chunking.section_chunker import build_section_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    blocks = [from_dict(ParsedBlock, record) for record in read_jsonl(args.blocks)]
    sections = []
    for doc_blocks in group_by_doc(blocks).values():
        sections.extend(build_section_chunks(doc_blocks, meta_from_block(doc_blocks[0])))
    write_jsonl(args.output, [to_dict(section) for section in sections])
    print(f"wrote {len(sections)} section chunks to {args.output}")


if __name__ == "__main__":
    main()

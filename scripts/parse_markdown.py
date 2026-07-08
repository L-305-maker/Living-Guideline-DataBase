from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guideline_chunking.io_utils import default_warnings_path, read_document_meta, write_jsonl
from src.guideline_chunking.markdown_parser import parse_markdown_document
from src.guideline_chunking.models import to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-dir", required=True)
    parser.add_argument("--metadata-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    for md_path in sorted(Path(args.markdown_dir).glob("*.md")):
        meta = read_document_meta(md_path, args.metadata_dir)
        blocks = parse_markdown_document(md_path.read_text(encoding="utf-8-sig"), meta)
        records.extend(to_dict(block) for block in blocks)
    write_jsonl(args.output, records)
    write_jsonl(default_warnings_path(args.output), [])
    print(f"wrote {len(records)} parsed blocks to {args.output}")


if __name__ == "__main__":
    main()

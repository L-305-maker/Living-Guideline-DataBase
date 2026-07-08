from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guideline_chunking.bm25_index import build_chunk_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_chunk_index(args.atomic, args.tables, args.index_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

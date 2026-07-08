from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guideline_chunking.chunk_retrieve_service import ChunkRetrieveService
from src.guideline_chunking.models import to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    service = ChunkRetrieveService(args.index_dir)
    results = [to_dict(result) for result in service.retrieve(args.query, top_k=args.top_k)]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

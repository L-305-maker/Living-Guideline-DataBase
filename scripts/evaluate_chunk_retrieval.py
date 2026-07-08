from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guideline_chunking.evaluation import evaluate_chunk_retrieval, format_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    print(format_metrics(evaluate_chunk_retrieval(args.index_dir, args.queries, args.top_k)))


if __name__ == "__main__":
    main()

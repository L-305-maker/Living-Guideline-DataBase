"""本地脚本入口：把 src 中的项目能力包装成命令行工具，方便运行、审计或质量检查。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.connection import DATABASE_URL, get_connection, mask_database_url
from src.storage.schema import SCHEMA_TABLES
from src.storage.utils import sql_identifier


RUN_FILES = {
    "recommendation_candidates": "recommendation_candidates.jsonl",
    "grade_candidates": "grade_candidates.jsonl",
    "pico_questions": "pico_questions.jsonl",
    "evidence_items": "evidence_items.jsonl",
    "recommendation_versions": "recommendation_versions.jsonl",
    "recommendation_versions_needs_review": "recommendation_versions.needs_review.jsonl",
}


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def db_counts() -> dict[str, Any]:
    result: dict[str, Any] = {
        "database_url": mask_database_url(DATABASE_URL),
        "table_counts": {},
        "missing_tables": [],
    }
    with get_connection() as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            for table in SCHEMA_TABLES:
                cur.execute(
                    "select exists ("
                    "select 1 from information_schema.tables "
                    "where table_schema = current_schema() and table_name = %s"
                    ")",
                    (table,),
                )
                if not cur.fetchone()[0]:
                    result["missing_tables"].append(table)
                    continue
                cur.execute(f"select count(*) from {sql_identifier(table)}")
                result["table_counts"][table] = cur.fetchone()[0]
    return result


def run_dir_counts(run_dir: Path) -> dict[str, int | None]:
    return {name: count_jsonl(run_dir / filename) for name, filename in RUN_FILES.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only PostgreSQL status check for storage ingest verification.")
    parser.add_argument("--run-dir", help="Optional reviewed run directory to compare with DB counts.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    report: dict[str, Any] = {"db": db_counts()}
    if args.run_dir:
        report["run_dir"] = str(Path(args.run_dir))
        report["run_dir_counts"] = run_dir_counts(Path(args.run_dir))

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


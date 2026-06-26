from __future__ import annotations

import argparse
import json

from src.common.process_jsonl import write_jsonl
from src.storage.connection import DATABASE_URL, connection, get_connection
from src.storage.contract_check import build_storage_contract_report
from src.storage.generic_ingest import GENERIC_TABLE_CONFIG, ingest_table, ingest_tables_atomically, integrity_report, table_counts
from src.storage.repositories.cleaned_records import (
    ingest_cleaned_records,
    insert_cleaned_records,
    insert_guidelines,
    insert_papers,
)
from src.storage.row_mappers import cleaned_record_row, guideline_row, paper_row
from src.storage.schema import SCHEMA_TABLES, create_indexes as _create_indexes, create_schema
from src.storage.utils import (
    DEFAULT_BATCH_SIZE,
    as_int as _as_int,
    batches as _batches,
    execute_many as _execute_many,
    iter_jsonl,
    json_list as _json_list,
    json_obj as _json,
    sql_identifier as _sql_identifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PostgreSQL storage for the Living-Guideline knowledge base.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create Living-Guideline PostgreSQL schema")
    init_parser.add_argument("--recreate", action="store_true", help="Drop and recreate all Living-Guideline tables")

    cleaned_parser = subparsers.add_parser("ingest-cleaned", help="Ingest source_cleaner JSONL output")
    cleaned_parser.add_argument("--input", required=True, help="Cleaned JSONL path")
    cleaned_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    table_parser = subparsers.add_parser("ingest-table", help="Ingest a schema_v2 table JSONL")
    table_parser.add_argument("--table", required=True, choices=sorted(GENERIC_TABLE_CONFIG))
    table_parser.add_argument("--input", required=True)
    table_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    validate_parser = subparsers.add_parser("validate-run", help="Validate a reviewed run directory against storage ingest contract")
    validate_parser.add_argument("--run-dir", required=True, help="Reviewed run directory containing storage JSONL files")
    validate_parser.add_argument("--output", help="Optional JSONL report path")

    subparsers.add_parser("counts", help="Print row counts for all Living-Guideline tables")
    subparsers.add_parser("integrity-report", help="Print storage integrity counts and orphan checks")

    args = parser.parse_args()
    if args.command == "init":
        create_schema(recreate=args.recreate)
    elif args.command == "ingest-cleaned":
        total = ingest_cleaned_records(args.input, batch_size=args.batch_size)
        print(f"Done. Ingested {total} cleaned records.")
    elif args.command == "ingest-table":
        total = ingest_table(args.table, args.input, batch_size=args.batch_size)
        print(f"Done. Ingested {total} rows into {args.table}.")
    elif args.command == "validate-run":
        report = build_storage_contract_report(args.run_dir)
        if args.output:
            write_jsonl(args.output, [report])
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "counts":
        print(json.dumps(table_counts(), ensure_ascii=False, indent=2))
    elif args.command == "integrity-report":
        print(json.dumps(integrity_report(), ensure_ascii=False, indent=2))


__all__ = [
    "DATABASE_URL",
    "DEFAULT_BATCH_SIZE",
    "GENERIC_TABLE_CONFIG",
    "SCHEMA_TABLES",
    "_as_int",
    "_batches",
    "_create_indexes",
    "_execute_many",
    "_json",
    "_json_list",
    "_sql_identifier",
    "cleaned_record_row",
    "connection",
    "create_schema",
    "build_storage_contract_report",
    "get_connection",
    "guideline_row",
    "ingest_cleaned_records",
    "ingest_table",
    "ingest_tables_atomically",
    "integrity_report",
    "insert_cleaned_records",
    "insert_guidelines",
    "insert_papers",
    "iter_jsonl",
    "main",
    "paper_row",
    "table_counts",
]


if __name__ == "__main__":
    main()

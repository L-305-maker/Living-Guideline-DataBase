"""Run PostgreSQL ingestion and embedding vectorization on a remote host over SSH."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


REMOTE_SCRIPT = r"""
set -euo pipefail

project_dir=$1
python_bin=$2
data_dir=$3
env_file=$4
model_name=$5
hf_home=$6
ingest_batch_size=$7
vector_batch_size=$8
page_size=$9
skip_ingest=${10}

cd "$project_dir"
test -x "$python_bin" || { echo "Python is not executable: $project_dir/$python_bin" >&2; exit 2; }
test -r "$env_file" || { echo "Environment file is not readable: $env_file" >&2; exit 2; }
test -d "$hf_home" || { echo "Hugging Face cache is missing: $hf_home" >&2; exit 2; }
for relative_path in documents.jsonl document_cards.jsonl document_views.jsonl chunks/all_chunks.jsonl; do
    test -f "$data_dir/$relative_path" || { echo "Missing data artifact: $data_dir/$relative_path" >&2; exit 2; }
done
for relative_path in markdown_clean markdown_raw sections; do
    test -d "$data_dir/$relative_path" || { echo "Missing data directory: $data_dir/$relative_path" >&2; exit 2; }
done

set -a
. "$env_file"
set +a
: "${POSTGRES_DSN:?POSTGRES_DSN is not set in the remote environment file}"
export PG_VECTOR_MODEL="$model_name"
export PG_VECTOR_LOCAL_ONLY=1
# 8B 模型默认 bf16 加载(需 Ampere+ GPU;不支持时 export PG_VECTOR_MODEL_DTYPE=float16)
export PG_VECTOR_MODEL_DTYPE="${PG_VECTOR_MODEL_DTYPE:-bfloat16}"
export HF_HOME="$hf_home"
export HF_HUB_CACHE="$hf_home/hub"
export SENTENCE_TRANSFORMERS_HOME="$hf_home"

"$python_bin" -m src.storage.postgres_store init --with-vector
if [ "$skip_ingest" = "0" ]; then
    "$python_bin" -m src.storage.postgres_store ingest --data-dir "$data_dir" --batch-size "$ingest_batch_size"
fi
for target in document_cards document_views chunks; do
    "$python_bin" -m src.storage.vectorize "$target" \
        --model "$model_name" \
        --batch-size "$vector_batch_size" \
        --page-size "$page_size"
done
"$python_bin" -m src.storage.postgres_store index-vectors
"$python_bin" -m src.storage.postgres_store verify --model "$model_name"
"""


def build_ssh_command(args: argparse.Namespace) -> list[str]:
    remote_args = [
        args.project_dir,
        args.python,
        args.data_dir,
        args.env_file,
        args.model,
        args.hf_home,
        str(args.ingest_batch_size),
        str(args.vector_batch_size),
        str(args.page_size),
        "1" if args.skip_ingest else "0",
    ]
    remote_command = "bash -s -- " + " ".join(shlex.quote(value) for value in remote_args)
    command = ["ssh"]
    if args.port:
        command.extend(["-p", str(args.port)])
    if args.identity_file:
        command.extend(["-i", str(Path(args.identity_file))])
    command.extend([args.ssh_target, remote_command])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use local SSH configuration to run remote PostgreSQL ingestion and embedding vectorization."
    )
    parser.add_argument("ssh_target", help="SSH config alias or user@host")
    parser.add_argument("--project-dir", default="/home/lhj/project/evidence_generation")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--data-dir", default="data/evidence")
    parser.add_argument("--env-file", default="/etc/pdf-markdown-rag/postgres.env")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--hf-home", default="/data/lhj/huggingface")
    parser.add_argument("--ingest-batch-size", type=int, default=500)
    parser.add_argument("--vector-batch-size", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--port", type=int)
    parser.add_argument("--identity-file")
    parser.add_argument("--skip-ingest", action="store_true", help="Resume missing vectorization without replacing PG data.")
    parser.add_argument("--dry-run", action="store_true", help="Print the SSH command without connecting.")
    args = parser.parse_args()
    if min(args.ingest_batch_size, args.vector_batch_size, args.page_size) < 1:
        parser.error("batch sizes and page size must be positive")
    return args


def main() -> None:
    args = parse_args()
    command = build_ssh_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return
    remote_script = REMOTE_SCRIPT.replace("\r\n", "\n").replace("\r", "\n")
    subprocess.run(command, input=remote_script, text=True, check=True)


if __name__ == "__main__":
    main()

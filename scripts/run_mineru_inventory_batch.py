from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "data/evidence_candidate/candidate_inventory.csv"
DEFAULT_OUTPUT = ROOT / "data/evidence_candidate/mineru_inventory_batch_001"
DEFAULT_DEEPSEEK = ROOT / "data/evidence_candidate/deepseek_batch_001/markdown_clean"
MINERU_EXE = ROOT / ".venv-mineru-gpu/Scripts/mineru.exe"
MINERU_CONFIG = ROOT / "data/mineru/mineru.json"
MODEL_CACHE = ROOT / "data/mineru/modelscope"


def valid_markdown(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def load_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["candidate_type"] == "repair_document"
        and row["action"] == "pending_baidu_ocr"
        and row["source_resolved"].lower() == "true"
    ]
    missing = [row["source_file"] for row in selected if not Path(row["source_file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source PDFs; first: {missing[0]}")
    doc_ids = [row["doc_id"] for row in selected]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate doc_id found in selected inventory rows")
    return selected


def markdown_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in directory.rglob("*.md") if valid_markdown(path)}


def assign_shards(rows: list[dict[str, str]], shard_count: int) -> list[list[dict[str, str]]]:
    shards: list[list[dict[str, str]]] = [[] for _ in range(min(shard_count, len(rows)))]
    page_totals = [0] * len(shards)
    for row in sorted(rows, key=lambda item: int(item["pdf_page_count"]), reverse=True):
        index = page_totals.index(min(page_totals))
        shards[index].append(row)
        page_totals[index] += int(row["pdf_page_count"])
    return shards


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_manifest(
    candidates: list[dict[str, str]], output_dir: Path, deepseek_dir: Path, shard_count: int
) -> tuple[list[list[dict[str, str]]], list[dict[str, object]]]:
    deepseek = markdown_index(deepseek_dir)
    mineru = markdown_index(output_dir / "output")
    remaining = [row for row in candidates if row["doc_id"] not in deepseek and row["doc_id"] not in mineru]
    shards = assign_shards(remaining, shard_count)
    shard_by_doc = {
        row["doc_id"]: index
        for index, shard in enumerate(shards, start=1)
        for row in shard
    }
    manifest: list[dict[str, object]] = []
    for row in candidates:
        doc_id = row["doc_id"]
        if doc_id in deepseek:
            status, result = "completed_deepseek", deepseek[doc_id]
        elif doc_id in mineru:
            status, result = "completed_mineru", mineru[doc_id]
        else:
            status, result = "pending_mineru", ""
        manifest.append(
            {
                "doc_id": doc_id,
                "title": row["title"],
                "source_file": row["source_file"],
                "pdf_page_count": row["pdf_page_count"],
                "reasons": row["reasons"],
                "inventory_action": row["action"],
                "shard": f"shard_{shard_by_doc[doc_id]:02d}" if doc_id in shard_by_doc else "",
                "status": status,
                "output_markdown": str(result),
            }
        )
    fields = list(manifest[0])
    write_csv(output_dir / "manifest.csv", manifest, fields)
    return shards, manifest


def stage_shard(shard: list[dict[str, str]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for row in shard:
        source = Path(row["source_file"])
        target = directory / f"{row['doc_id']}.pdf"
        if target.exists():
            if target.stat().st_size != source.stat().st_size:
                raise FileExistsError(f"Stale staged file: {target}")
            continue
        os.link(source, target)


def run_shards(shards: list[list[dict[str, str]]], output_dir: Path) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env.update(
        {
            "MINERU_TOOLS_CONFIG_JSON": str(MINERU_CONFIG),
            "MINERU_MODEL_SOURCE": "local",
            "MODELSCOPE_CACHE": str(MODEL_CACHE),
            "CUDA_VISIBLE_DEVICES": "0",
            "MINERU_PROCESSING_WINDOW_SIZE": "1",
            "MINERU_PDF_RENDER_THREADS": "1",
            "MINERU_API_MAX_CONCURRENT_REQUESTS": "1",
            "MINERU_INTRA_OP_NUM_THREADS": "8",
            "MINERU_INTER_OP_NUM_THREADS": "2",
        }
    )
    for index, shard in enumerate(shards, start=1):
        if not shard:
            continue
        staged = output_dir / "run_inputs" / run_id / f"shard_{index:02d}"
        stage_shard(shard, staged)
        pages = sum(int(row["pdf_page_count"]) for row in shard)
        log_path = output_dir / "logs" / f"{run_id}_shard_{index:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"START shard_{index:02d}: {len(shard)} documents, {pages} pages", flush=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            result = subprocess.run(
                [
                    str(MINERU_EXE),
                    "-p",
                    str(staged),
                    "-o",
                    str(output_dir / "output"),
                    "-b",
                    "pipeline",
                    "-m",
                    "ocr",
                    "-l",
                    "ch",
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"END shard_{index:02d}: exit={result.returncode}, log={log_path}", flush=True)


def write_quality_report(output_dir: Path) -> Counter[str]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "output").rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "doc_id": path.stem,
                "characters": len(text),
                "headings": len(re.findall(r"(?m)^#{1,6}\\s+", text)),
                "replacement_characters": text.count("\ufffd"),
                "control_characters": len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text)),
                "output_markdown": str(path),
            }
        )
    if rows:
        write_csv(output_dir / "ocr_quality.csv", rows, list(rows[0]))
    return Counter("nonempty" if int(row["characters"]) > 0 else "empty" for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable local MinerU OCR for inventory candidates")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deepseek-dir", type=Path, default=DEFAULT_DEEPSEEK)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be at least 1")
    for required in (args.inventory, MINERU_EXE, MINERU_CONFIG, MODEL_CACHE):
        if not required.exists():
            parser.error(f"Required path not found: {required}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.inventory.resolve())
    shards, manifest = prepare_manifest(candidates, output_dir, args.deepseek_dir.resolve(), args.shards)
    counts = Counter(str(row["status"]) for row in manifest)
    pending_pages = sum(int(row["pdf_page_count"]) for row in manifest if row["status"] == "pending_mineru")
    print(f"Inventory selected: {len(candidates)}", flush=True)
    print(f"Status: {dict(counts)}", flush=True)
    print(f"Pending: {counts['pending_mineru']} documents, {pending_pages} pages", flush=True)
    print(f"Manifest: {output_dir / 'manifest.csv'}", flush=True)
    if args.run and counts["pending_mineru"]:
        run_shards(shards, output_dir)
        _, manifest = prepare_manifest(candidates, output_dir, args.deepseek_dir.resolve(), args.shards)
        counts = Counter(str(row["status"]) for row in manifest)
    quality = write_quality_report(output_dir)
    print(f"Final status: {dict(counts)}", flush=True)
    print(f"MinerU quality rows: {dict(quality)}", flush=True)
    return 0 if not args.run or counts["pending_mineru"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

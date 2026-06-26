from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


JsonDict = Dict[str, Any]

DEFAULT_CATEGORIES = {
    "raw_pdf": {
        "type": "raw_download",
        "git_policy": "exclude",
        "retention": "keep source manifest; raw files may be regenerated from crawler inputs",
    },
    "origin": {
        "type": "origin_jsonl",
        "git_policy": "exclude by default",
        "retention": "keep reproducible snapshots only when tied to a regression run",
    },
    "processed": {
        "type": "pipeline_output",
        "git_policy": "exclude by default",
        "retention": "keep current and named regression baselines; remove tmp runs after review",
    },
    "reference": {
        "type": "reference_input",
        "git_policy": "case by case",
        "retention": "keep if manually curated or external-standard input",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def classify_processed_child(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("tmp") or "tmp" in name:
        return "temporary_run"
    if name.startswith("regression") or "baseline" in name:
        return "regression_baseline"
    if name in {"current", "latest"} or name.endswith("_latest"):
        return "current_output"
    return "pipeline_output"


def summarize_dir(path: Path, base: Path, include_hashes: bool, sample_limit: int) -> JsonDict:
    files = list(iter_files(path))
    total_bytes = sum(file.stat().st_size for file in files)
    samples: List[JsonDict] = []
    for file in files[:sample_limit]:
        item: JsonDict = {
            "path": file.relative_to(base).as_posix(),
            "bytes": file.stat().st_size,
            "modified_at": datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat(),
        }
        if include_hashes:
            item["sha256"] = file_sha256(file)
        samples.append(item)
    return {
        "path": path.relative_to(base).as_posix(),
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sample_files": samples,
    }


def build_manifest(data_root: str | Path = "data", include_hashes: bool = False, sample_limit: int = 20) -> JsonDict:
    root = Path(data_root)
    manifest: JsonDict = {
        "manifest_version": 1,
        "created_at": utc_now(),
        "data_root": root.as_posix(),
        "categories": [],
        "processed_runs": [],
        "notes": [
            "Large generated artifacts should remain outside Git.",
            "Stable small samples belong in tests/fixtures.",
            "Temporary processed runs should be removed or promoted to named regression baselines.",
        ],
    }
    for name, policy in DEFAULT_CATEGORIES.items():
        path = root / name
        summary = summarize_dir(path, root, include_hashes, sample_limit)
        summary.update(policy)
        manifest["categories"].append(summary)

    processed = root / "processed"
    if processed.exists():
        for child in sorted(processed.iterdir()):
            if not child.is_dir():
                continue
            summary = summarize_dir(child, root, include_hashes, sample_limit=5)
            summary["type"] = classify_processed_child(child)
            summary["keep"] = summary["type"] in {"regression_baseline", "current_output"}
            manifest["processed_runs"].append(summary)
    return manifest


def write_manifest(output: str | Path, manifest: JsonDict) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a data artifact manifest for generated pipeline data.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/MANIFEST.json")
    parser.add_argument("--include-hashes", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.data_root, include_hashes=args.include_hashes, sample_limit=args.sample_limit)
    write_manifest(args.output, manifest)
    print(
        "manifest_written={output} categories={categories} processed_runs={processed_runs}".format(
            output=args.output,
            categories=len(manifest["categories"]),
            processed_runs=len(manifest["processed_runs"]),
        )
    )


if __name__ == "__main__":
    main()

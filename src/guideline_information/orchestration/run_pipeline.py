"""Run guideline information steps from existing evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.guideline_information.extraction.pipeline import build_candidates_from_sections
from src.guideline_information.paths import InformationRunPaths
from src.guideline_information.models import SCHEMA_VERSION
from src.guideline_information.repository import write_models
from src.utils.io import DATA_DIR, ensure_parent


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_information_pipeline(
    *,
    evidence_data_dir: str | Path = DATA_DIR,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    paths = InformationRunPaths.create(output_root or DATA_DIR.parent / "guideline_information", run_id)
    sections_dir = Path(evidence_data_dir) / "sections"
    input_hashes = {str(path): hash_file(path) for path in sorted(sections_dir.glob("*.jsonl"))}
    candidates = build_candidates_from_sections(sections_dir, paths.run_id)
    write_models(paths.candidates, candidates)
    manifest = {
        "pipeline_version": "guideline_information_v1",
        "run_id": paths.run_id,
        "input_sections_dir": str(sections_dir),
        "input_file_hashes": input_hashes,
        "code_version": "working_tree",
        "model_versions": [],
        "prompt_versions": [],
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(candidates),
        "extracts_with_model": False,
    }
    ensure_parent(paths.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

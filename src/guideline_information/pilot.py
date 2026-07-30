"""Prepare small reproducible pilot datasets from existing evidence sections."""

from __future__ import annotations

import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.guideline_chunking.block_classifier import classify_block_type
from src.guideline_information.paths import DEFAULT_INFORMATION_DIR, PilotPaths
from src.utils.ids import sha256_file
from src.utils.io import DATA_DIR, read_jsonl, write_jsonl


def hash_file(path: Path) -> str:
    return sha256_file(path)


def prepare_pilot(
    *,
    evidence_data_dir: str | Path = DATA_DIR,
    output_root: str | Path = DEFAULT_INFORMATION_DIR,
    organization: str = "IDSA",
    pilot_id: str = "idsa_pilot_v1",
    documents: int = 3,
    seed: int = 42,
    max_sections: int = 150,
) -> dict[str, Any]:
    data_dir = Path(evidence_data_dir)
    docs_path = data_dir / "documents.jsonl"
    sections_dir = data_dir / "sections"
    if not docs_path.is_file() or not sections_dir.is_dir():
        raise FileNotFoundError(f"Missing evidence documents or sections under {data_dir}")
    all_docs = list(read_jsonl(docs_path))
    docs = [row for row in all_docs if _matches_organization(row, organization)]
    metadata_issues: list[str] = []
    fallback_docs = [row for row in docs if organization.lower() not in str(row.get("source_institution") or "").lower()]
    if fallback_docs:
        metadata_issues.append("organization identified from source_file/title/doc_id because source_institution did not contain organization")
    if not docs:
        raise ValueError(f"No documents found for organization={organization}")
    sections_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_hashes = {str(docs_path): hash_file(docs_path)}
    for path in sorted(sections_dir.glob("*.jsonl")):
        input_hashes[str(path)] = hash_file(path)
        for section in read_jsonl(path):
            doc_id = str(section.get("doc_id") or "")
            if doc_id:
                sections_by_doc[doc_id].append(section)
    eligible = [doc for doc in docs if sections_by_doc.get(str(doc.get("doc_id") or ""))]
    if not eligible:
        raise ValueError(f"No section-complete documents found for organization={organization}")
    rng = random.Random(seed)
    eligible.sort(key=lambda row: (str(row.get("publication_date") or ""), str(row.get("doc_id") or "")))
    rng.shuffle(eligible)
    selected_docs = eligible[: max(1, documents)]
    selected_doc_ids = {str(doc.get("doc_id") or "") for doc in selected_docs}
    selected_sections = [section for doc_id in selected_doc_ids for section in sections_by_doc[doc_id]]
    selected_sections.sort(key=lambda row: (str(row.get("doc_id") or ""), int(row.get("char_start") or 0)))
    selected_sections = _balanced_sections(selected_sections, max_sections)
    profile_counts = Counter(classify_block_type(str(row.get("content") or ""), [str(item) for item in row.get("section_path") or []]) for row in selected_sections)
    paths = PilotPaths.create(output_root, pilot_id)
    write_jsonl(paths.selected_documents, selected_docs)
    write_jsonl(paths.selected_sections, selected_sections)
    coverage = {
        "candidate_profile_counts": dict(profile_counts),
        "hard_negative_sections": sum(profile_counts.get(name, 0) for name in ["method", "rationale_candidate", "evidence_candidate", "reference"]),
    }
    paths.expected_coverage.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    git_info = _git_info()
    manifest = {
        "pilot_id": pilot_id,
        "organization": organization,
        "selected_doc_ids": sorted(selected_doc_ids),
        "selected_section_ids": [
            f"{row.get('doc_id')}:{row.get('char_start', 0)}:{row.get('char_end', 0)}" for row in selected_sections
        ],
        "selection_seed": seed,
        "selection_strategy": "metadata organization filter plus balanced recommendation/hard-negative sections",
        "requested_documents": documents,
        "selected_documents": len(selected_docs),
        "selected_sections": len(selected_sections),
        "input_file_hashes": input_hashes,
        "input_manifest_hash": input_hashes.get(str(docs_path), ""),
        "metadata_issues": metadata_issues,
        "code_commit": git_info["code_commit"],
        "working_tree_dirty": git_info["working_tree_dirty"],
        "working_tree_status": git_info["working_tree_status"],
        "expected_coverage": coverage,
    }
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _balanced_sections(sections: list[dict[str, Any]], max_sections: int) -> list[dict[str, Any]]:
    if len(sections) <= max_sections:
        return sections
    recs = []
    hard = []
    other = []
    for row in sections:
        label = classify_block_type(str(row.get("content") or ""), [str(item) for item in row.get("section_path") or []])
        if label == "recommendation_candidate":
            recs.append(row)
        elif label in {"method", "rationale_candidate", "evidence_candidate", "reference"}:
            hard.append(row)
        else:
            other.append(row)
    quota_rec = min(len(recs), max_sections // 2)
    quota_hard = min(len(hard), max_sections // 4)
    remaining = max_sections - quota_rec - quota_hard
    return (recs[:quota_rec] + hard[:quota_hard] + other[:remaining])[:max_sections]



def _matches_organization(row: dict[str, Any], organization: str) -> bool:
    needle = organization.lower()
    haystack = " ".join(
        str(row.get(key) or "") for key in ["source_institution", "organization", "title", "source_file", "document_kind", "doc_id"]
    ).lower()
    return needle in haystack or (needle == "idsa" and "infectious diseases society of america" in haystack)


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], text=True).splitlines()
        return {"code_commit": commit, "working_tree_dirty": bool(status), "working_tree_status": status[:100]}
    except Exception as exc:  # pragma: no cover
        return {"code_commit": "unknown", "working_tree_dirty": True, "working_tree_status": [type(exc).__name__]}

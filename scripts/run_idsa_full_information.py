from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.guideline_information.extraction.candidate_builder import CandidateBuilder
from src.guideline_information.orchestration.pilot_pipeline import run_pilot
from src.guideline_information.repository import write_models
from src.pipeline.cleaning.chunker import chunk_file
from src.pipeline.cleaning.cleaner import clean_all
from src.pipeline.cleaning.encoder import encode_file
from src.pipeline.cleaning.pdf_to_md import convert_all
from src.utils.front_matter import parse_front_matter
from src.utils.io import read_jsonl, write_jsonl


DEFAULT_SOURCE_DIR = Path("data/raw_pdf/idsa_pdf")
DEFAULT_OUTPUT_ROOT = Path("information/IDSA")
DEFAULT_RUN_ID = "idsa_full_information_v1"


def run(
    *,
    source_dir: Path,
    output_root: Path,
    run_id: str,
    workers: int,
    skip_model: bool,
    model_only: bool,
    max_items: int | None,
    max_model_calls: int | None,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    evidence_dir = output_root / "evidence"
    summary_path = output_root / "full_information_summary.json"
    if model_only:
        candidate_path = evidence_dir / "candidates.jsonl"
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing candidates for model-only run: {candidate_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
        summary.update({"run_id": run_id, "source_dir": str(source_dir), "output_root": str(output_root), "candidates": count_jsonl(candidate_path), "candidate_path": str(candidate_path)})
        summary["model_run"] = run_pilot(
            pilot_id="idsa_full_v1",
            run_id=run_id,
            output_root=output_root,
            client_name="openai-compatible",
            input_candidates_path=candidate_path,
            max_model_calls=max_model_calls,
            resume=resume,
            force=False,
        )
        summary["model_artifacts"] = model_artifact_counts(output_root, run_id)
        write_json(summary_path, summary)
        return summary
    raw_dir = evidence_dir / "markdown_raw"
    clean_dir = evidence_dir / "markdown_clean"
    sections_dir = evidence_dir / "sections"
    chunks_dir = evidence_dir / "chunks"
    if force and evidence_dir.exists():
        backup = evidence_dir.with_name(f"{evidence_dir.name}.backup")
        if backup.exists():
            shutil.rmtree(backup)
        evidence_dir.rename(backup)
    for path in [raw_dir, clean_dir, sections_dir, chunks_dir]:
        path.mkdir(parents=True, exist_ok=True)

    conversion = convert_all(
        source_dir,
        raw_dir,
        evidence_dir / "documents_raw.jsonl",
        ocr_mode="never",
        workers=workers,
        progress_every=5,
        document_kind="guideline",
    )
    cleaning = clean_all(raw_dir, clean_dir, evidence_dir / "documents.jsonl", progress_every=5)
    active_docs, excluded_docs = filter_documents(evidence_dir / "documents.jsonl", clean_dir)
    write_jsonl(evidence_dir / "documents_for_extraction.jsonl", active_docs)
    write_jsonl(evidence_dir / "documents_excluded_for_extraction.jsonl", excluded_docs)

    sections, chunks = build_sections_and_chunks(active_docs, sections_dir, chunks_dir)
    candidates = CandidateBuilder().build(sections, run_id)
    if max_items is not None:
        candidates = candidates[:max_items]
    candidate_path = evidence_dir / "candidates.jsonl"
    write_models(candidate_path, candidates)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "conversion": conversion,
        "cleaning": cleaning,
        "documents": {
            "raw": count_jsonl(evidence_dir / "documents_raw.jsonl"),
            "cleaned": count_jsonl(evidence_dir / "documents.jsonl"),
            "for_extraction": len(active_docs),
            "excluded_for_extraction": len(excluded_docs),
            "exclusion_reasons": dict(Counter(row["reason"] for row in excluded_docs)),
        },
        "sections": {"files": len(list(sections_dir.glob("*.jsonl"))), "records": len(sections)},
        "chunks": chunks,
        "candidates": len(candidates),
        "candidate_path": str(candidate_path),
        "model_run": None,
    }
    if not skip_model:
        summary["model_run"] = run_pilot(
            pilot_id="idsa_full_v1",
            run_id=run_id,
            output_root=output_root,
            client_name="openai-compatible",
            input_candidates_path=candidate_path,
            max_model_calls=max_model_calls,
            resume=resume,
            force=False,
        )
    summary["model_artifacts"] = model_artifact_counts(output_root, run_id)
    write_json(summary_path, summary)
    return summary


def filter_documents(documents_path: Path, clean_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in read_jsonl(documents_path):
        clean_path = Path(row.get("markdown_clean_path") or clean_dir / f"{row.get('doc_id')}.md")
        body = ""
        if clean_path.exists():
            _metadata, body = parse_front_matter(clean_path.read_text(encoding="utf-8", errors="replace"))
        reason = exclusion_reason(row, body)
        if reason:
            excluded.append({"doc_id": row.get("doc_id", ""), "title": row.get("title", ""), "source_file": row.get("source_file", ""), "reason": reason})
            continue
        active.append(row)
    return active, excluded


def exclusion_reason(row: dict[str, Any], body: str) -> str:
    text = "\n".join([str(row.get("title") or ""), str(row.get("abstract") or ""), body[:4000]])
    if contains_japanese_kana(text):
        return "JAPANESE_GUIDELINE_EXCLUDED"
    if row.get("cleaning_quality") == "poor":
        return "POOR_TEXT_QUALITY_EXCLUDED"
    if row.get("pdf_needs_ocr") or row.get("source_pdf_needs_ocr"):
        return "OCR_REQUIRED_EXCLUDED"
    return ""


def contains_japanese_kana(text: str) -> bool:
    return any("\u3040" <= char <= "\u30ff" for char in text or "")


def build_sections_and_chunks(active_docs: list[dict[str, Any]], sections_dir: Path, chunks_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for path in sections_dir.glob("*.jsonl"):
        path.unlink()
    for path in chunks_dir.glob("*.jsonl"):
        path.unlink()
    sections: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    for row in active_docs:
        clean_path = Path(str(row["markdown_clean_path"]))
        encoded = encode_file(clean_path, sections_dir)
        chunked = chunk_file(clean_path, chunks_dir)
        sections.extend(item.model_dump(mode="json") for item in encoded)
        all_chunks.extend(item.model_dump(mode="json") for item in chunked)
    write_jsonl(chunks_dir / "all_chunks.jsonl", all_chunks)
    return sections, {"files": len(list(chunks_dir.glob("*.jsonl"))) - 1, "records": len(all_chunks), "chunks_dir": str(chunks_dir)}


def model_artifact_counts(output_root: Path, run_id: str) -> dict[str, int]:
    run_dir = output_root / "runs" / run_id
    return {
        "extraction_results": count_jsonl(run_dir / "extraction_results.jsonl"),
        "verification_results": count_jsonl(run_dir / "verification_results.jsonl"),
        "validation_results": count_jsonl(run_dir / "validation_results.jsonl"),
        "route_results": count_jsonl(run_dir / "route_results.jsonl"),
        "model_responses": count_jsonl(run_dir / "model_responses.jsonl"),
        "extraction_failures": count_jsonl(run_dir / "extraction_failures.jsonl"),
        "verification_failures": count_jsonl(run_dir / "verification_failures.jsonl"),
    }


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full IDSA guideline information extraction into information/IDSA")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-model-calls", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(
        source_dir=args.source_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        workers=args.workers,
        skip_model=args.skip_model,
        model_only=args.model_only,
        max_items=args.max_items,
        max_model_calls=args.max_model_calls,
        resume=args.resume,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

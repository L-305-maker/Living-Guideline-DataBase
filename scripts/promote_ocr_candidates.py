from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import fasttext


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.cleaning.chunker import chunk_all
from src.pipeline.cleaning.cleaner import clean_all
from src.retrieval.document_repr import build_document_representations
from src.utils.front_matter import dump_front_matter, parse_front_matter


DEFAULT_INVENTORY = ROOT / "data/evidence_candidate/candidate_inventory.csv"
DEFAULT_DATA_DIR = ROOT / "data/evidence_candidate"
DEFAULT_MINERU = DEFAULT_DATA_DIR / "mineru_inventory_batch_001/output"
DEFAULT_DEEPSEEK = DEFAULT_DATA_DIR / "deepseek_batch_001/markdown_raw"
DEFAULT_LANGUAGE_MODEL = ROOT / "data/language_models/lid.176.ftz"
ALLOWED_LANGUAGES = {"en", "zh"}
TAG_RE = re.compile(r"<[^>]+>")
IMAGE_RE = re.compile(r"(?P<prefix>\(|src=[\"'])images/(?P<name>[^)\"']+)")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["candidate_type"] == "repair_document"
        and row["action"] == "pending_baidu_ocr"
        and row["source_resolved"].lower() == "true"
    ]


def locate_markdown(row: dict[str, str], mineru_dir: Path, deepseek_dir: Path) -> tuple[Path, str]:
    deepseek = deepseek_dir / f"{row['doc_id']}.md"
    if deepseek.is_file():
        return deepseek, "deepseek"
    mineru = mineru_dir / row["doc_id"] / "ocr" / f"{row['doc_id']}.md"
    if mineru.is_file():
        return mineru, "mineru"
    raise FileNotFoundError(f"OCR Markdown not found for {row['doc_id']}")


def language_text(markdown: str) -> str:
    _metadata, body = parse_front_matter(markdown)
    body = re.sub(r"!\[[^]]*]\([^)]*\)", " ", body)
    body = TAG_RE.sub(" ", body)
    body = re.sub(r"[`#*_>|~$\\]", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def sampled_chunks(text: str, size: int = 1600, limit: int = 48) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    chunk_count = min(limit, max(1, len(text) // size))
    starts = [round(index * (len(text) - size) / max(1, chunk_count - 1)) for index in range(chunk_count)]
    return [text[start : start + size] for start in starts]


def detect_language(model: Any, text: str) -> dict[str, Any]:
    weights: defaultdict[str, float] = defaultdict(float)
    total = 0
    for chunk in sampled_chunks(text):
        labels, probabilities = model.predict(chunk.replace("\n", " "), k=3)
        weight = len(chunk)
        total += weight
        for label, probability in zip(labels, probabilities):
            weights[label.removeprefix("__label__")] += float(probability) * weight
    shares = {language: value / max(1, total) for language, value in weights.items()}
    primary = max(shares, key=shares.get) if shares else "unknown"
    visible = sum(char.isalnum() for char in text)
    cjk_ratio = len(CJK_RE.findall(text)) / max(1, visible)
    if cjk_ratio >= 0.20:
        primary = "zh"
    allowed_share = sum(shares.get(language, 0.0) for language in ALLOWED_LANGUAGES)
    return {
        "primary_language": primary,
        "language_confidence": round(shares.get(primary, cjk_ratio if primary == "zh" else 0.0), 4),
        "allowed_language_share": round(allowed_share, 4),
        "cjk_ratio": round(cjk_ratio, 4),
        "language_top3": "|".join(f"{key}:{value:.4f}" for key, value in sorted(shares.items(), key=lambda item: item[1], reverse=True)[:3]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def classify(
    inventory: list[dict[str, str]], mineru_dir: Path, deepseek_dir: Path, model_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = fasttext.load_model(str(model_path))
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in inventory:
        markdown_path, provider = locate_markdown(row, mineru_dir, deepseek_dir)
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        language = detect_language(model, language_text(markdown))
        output = {**row, **language, "ocr_provider": provider, "ocr_markdown_path": str(markdown_path)}
        if language["primary_language"] in ALLOWED_LANGUAGES:
            output["processing_status"] = "include"
            included.append(output)
        else:
            output["processing_status"] = "exclude_non_zh_en"
            output["exclusion_reason"] = f"primary_language={language['primary_language']}"
            excluded.append(output)
    return included, excluded


def publication_date(doc_id: str) -> str:
    parts = doc_id.split("_")
    return parts[1] if len(parts) > 1 and re.fullmatch(r"(?:19|20)\d{2}", parts[1]) else "unknown"


def resolved_title(candidate: str, body: str) -> str:
    title = candidate.strip()
    if re.fullmatch(r"[0-9a-fA-F]{16,}", title):
        heading = re.search(r"(?m)^#\s+(.+?)\s*$", body)
        if heading:
            return heading.group(1).strip()
    return title


def link_assets(source_ocr_dir: Path, stage_raw: Path, doc_id: str, body: str) -> str:
    images = source_ocr_dir / "images"
    if not images.is_dir():
        return body
    target_dir = stage_raw / "assets" / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in images.iterdir():
        if not source.is_file():
            continue
        target = target_dir / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return IMAGE_RE.sub(lambda match: f"{match.group('prefix')}../markdown_raw/assets/{doc_id}/{match.group('name')}", body)


def prepare_raw(included: list[dict[str, Any]], stage_raw: Path) -> dict[str, int]:
    stage_raw.mkdir(parents=True, exist_ok=True)
    providers: Counter[str] = Counter()
    assets = 0
    for row in included:
        source_path = Path(row["ocr_markdown_path"])
        metadata, body = parse_front_matter(source_path.read_text(encoding="utf-8", errors="replace"))
        if row["ocr_provider"] == "mineru":
            before = sum(1 for _ in (source_path.parent / "images").glob("*") if _.is_file())
            body = link_assets(source_path.parent, stage_raw, row["doc_id"], body)
            assets += before
        metadata.update(
            {
                "id": row["doc_id"],
                "title": resolved_title(row["title"], body),
                "publication_date": metadata.get("publication_date") or publication_date(row["doc_id"]),
                "source_institution": row["source_institution"],
                "source_file": row["source_file"],
                "document_kind": "guideline",
                "pdf_page_count": row["pdf_page_count"],
                "ocr_engine": "mineru_pipeline" if row["ocr_provider"] == "mineru" else "infini_deepseek_ocr_2",
                "ocr_applied": "true",
                "ocr_status": "applied",
                "ocr_error": "",
                "detected_language": row["primary_language"],
                "language_confidence": row["language_confidence"],
                "allowed_language_share": row["allowed_language_share"],
                "candidate_reasons": row["reasons"],
            }
        )
        for key in ("cleaning_quality", "cleaning_flags", "clinical_department", "clinical_departments", "department_scope"):
            metadata.pop(key, None)
        (stage_raw / f"{row['doc_id']}.md").write_text(dump_front_matter(metadata, body), encoding="utf-8", newline="\n")
        providers[row["ocr_provider"]] += 1
    return {"markdown": len(included), "assets": assets, **{f"provider_{key}": value for key, value in providers.items()}}


def rewrite_document_paths(path: Path, stage: Path, data_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("markdown_raw_path", "markdown_clean_path"):
                row[key] = str(row.get(key, "")).replace(str(stage / "markdown_raw"), str(data_dir / "markdown_raw")).replace(
                    str(stage / "markdown_clean"), str(data_dir / "markdown_clean")
                )
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_stage(stage: Path, data_dir: Path) -> dict[str, Any]:
    clean = clean_all(stage / "markdown_raw", stage / "markdown_clean", stage / "documents.jsonl", progress_every=100)
    rewrite_document_paths(stage / "documents.jsonl", stage, data_dir)
    representations = build_document_representations(stage / "markdown_clean", stage)
    chunks = chunk_all(stage / "markdown_clean", stage / "chunks")
    return {"clean": clean, "document_representations": representations, "chunks": chunks}


def validate_stage(stage: Path) -> dict[str, int]:
    paths = {
        "documents": stage / "documents.jsonl",
        "document_cards": stage / "document_cards.jsonl",
        "document_views": stage / "document_views.jsonl",
        "all_chunks": stage / "chunks/all_chunks.jsonl",
    }
    counts = {name: sum(1 for line in path.open(encoding="utf-8") if line.strip()) for name, path in paths.items()}
    if not counts["documents"] or counts["documents"] != counts["document_cards"] or not counts["all_chunks"]:
        raise RuntimeError(f"Stage validation failed: {counts}")
    return counts


def promote(stage: Path, data_dir: Path, run_id: str) -> Path:
    data_root = data_dir.resolve()
    if data_root != (ROOT / "data/evidence_candidate").resolve():
        raise RuntimeError(f"Refusing to promote outside expected data directory: {data_root}")
    backup = data_dir / "backups" / f"ocr_promotion_{run_id}"
    backup.mkdir(parents=True, exist_ok=False)
    names = ["markdown_raw", "markdown_clean", "chunks", "documents.jsonl", "documents_excluded.jsonl", "document_cards.jsonl", "document_views.jsonl"]
    for name in names:
        current = data_dir / name
        if current.exists():
            shutil.move(str(current), str(backup / name))
    for name in names:
        prepared = stage / name
        if prepared.exists():
            shutil.move(str(prepared), str(data_dir / name))
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote reviewed OCR Markdown into evidence_candidate artifacts")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mineru-dir", type=Path, default=DEFAULT_MINERU)
    parser.add_argument("--deepseek-dir", type=Path, default=DEFAULT_DEEPSEEK)
    parser.add_argument("--language-model", type=Path, default=DEFAULT_LANGUAGE_MODEL)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    inventory = load_inventory(args.inventory.resolve())
    included, excluded = classify(inventory, args.mineru_dir.resolve(), args.deepseek_dir.resolve(), args.language_model.resolve())
    fields = list(included[0]) + ["exclusion_reason"]
    write_csv(args.data_dir / "candidate_processing_inventory.csv", included, fields)
    write_csv(args.data_dir / "candidate_language_exclusions.csv", excluded, fields)
    summary: dict[str, Any] = {
        "inventory_selected": len(inventory),
        "included": len(included),
        "excluded_non_zh_en": len(excluded),
        "included_languages": dict(Counter(row["primary_language"] for row in included)),
        "excluded_languages": dict(Counter(row["primary_language"] for row in excluded)),
        "apply": args.apply,
    }
    if args.apply:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage = args.data_dir / f".ocr_promotion_stage_{run_id}"
        stage.mkdir(parents=True, exist_ok=False)
        summary["prepared"] = prepare_raw(included, stage / "markdown_raw")
        summary["build"] = build_stage(stage, args.data_dir)
        summary["validated"] = validate_stage(stage)
        summary["backup_dir"] = str(promote(stage, args.data_dir, run_id))
    (args.data_dir / "candidate_processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

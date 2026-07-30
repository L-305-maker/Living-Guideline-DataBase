"""Prepare frozen real-model experiment batches."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.guideline_information.extraction.adapters import ADAPTER_VERSION
from src.guideline_information.extraction.candidate_builder import CandidateBuilder
from src.guideline_information.extraction.model_config import get_model_config_value, model_config_audit
from src.guideline_information.extraction.response_normalizer import NORMALIZER_VERSION
from src.guideline_information.extraction.schemas import EXTRACTION_SCHEMA, VERIFICATION_SCHEMA
from src.guideline_information.extraction.span_validator import ALLOWED_COORDINATES_BY_FIELD
from src.guideline_information.ids import ID_ALGORITHM_VERSION
from src.guideline_information.models import RecommendationCandidate, SCHEMA_VERSION
from src.guideline_information.paths import DEFAULT_INFORMATION_DIR, PilotPaths
from src.guideline_information.repository import write_models
from src.utils.ids import sha256_file
from src.utils.io import DATA_DIR, read_jsonl

DEFAULT_EXPERIMENT_ROOT = DATA_DIR.parent / "guideline_information" / "experiments"


def prepare_experiment_batch(
    *,
    pilot_id: str,
    pilot_root: str | Path = DEFAULT_INFORMATION_DIR,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    experiment_id: str = "idsa_real_model_experiment_v1",
    source_run_id: str = "idsa_real_model_candidate_batch",
    stage_a_size: int = 5,
    stage_b_size: int = 15,
    seed: int = 42,
    parent_experiment_id: str = "",
    restart_reason: str = "",
) -> dict[str, Any]:
    pilot_paths = PilotPaths.create(pilot_root, pilot_id)
    if not pilot_paths.selected_sections.exists():
        raise FileNotFoundError(f"Missing selected sections: {pilot_paths.selected_sections}")
    sections = list(read_jsonl(pilot_paths.selected_sections))
    candidates = CandidateBuilder().build(sections, source_run_id)
    stage_a, stage_b, reasons = _select_batches(candidates, stage_a_size=stage_a_size, stage_b_size=stage_b_size)
    exp_dir = Path(experiment_root) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    stage_a_path = exp_dir / "stage_a_candidates.jsonl"
    stage_b_path = exp_dir / "stage_b_candidates.jsonl"
    manifest_path = exp_dir / "experiment_manifest.json"
    report_path = exp_dir / "selection_report.md"
    write_models(stage_a_path, stage_a)
    write_models(stage_b_path, stage_b)
    manifest = {
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "restart_reason": restart_reason,
        "pilot_id": pilot_id,
        "source_run_id": source_run_id,
        "selected_candidate_ids": [item.candidate_id for item in stage_a + stage_b],
        "stage_a_candidate_ids": [item.candidate_id for item in stage_a],
        "stage_b_candidate_ids": [item.candidate_id for item in stage_b],
        "selected_doc_ids": sorted({item.doc_id for item in stage_a + stage_b}),
        "selection_strategy": "stratified recommendation and hard-negative batch from frozen pilot sections",
        "selection_seed": seed,
        "input_file_hashes": {
            str(pilot_paths.selected_documents): sha256_file(pilot_paths.selected_documents),
            str(pilot_paths.selected_sections): sha256_file(pilot_paths.selected_sections),
        },
        "code_commit": _git(["rev-parse", "HEAD"]),
        "working_tree_dirty": bool(_git(["status", "--short", "--untracked-files=all"])),
        "working_tree_diff_hash": _git_diff_hash(),
        "prompt_versions": {"extractor": "extraction_v1", "verifier": "verification_v1"},
        "frozen_hashes": frozen_configuration_hashes(),
        "schema_version": SCHEMA_VERSION,
        "id_algorithm_version": ID_ALGORITHM_VERSION,
        "model_configuration": model_configuration_status(),
        "model_name": get_model_config_value("GUIDELINE_LLM_MODEL"),
        "engineering_versions": {"adapter": ADAPTER_VERSION, "response_normalizer": NORMALIZER_VERSION},
        "engineering_hashes": {
            "adapter_sha256": sha256_file(Path(__file__).resolve().parent / "extraction" / "adapters.py"),
            "response_normalizer_sha256": sha256_file(Path(__file__).resolve().parent / "extraction" / "response_normalizer.py"),
        },
        "controlled_restart_flags": {
            "prompt_changed": False,
            "canonical_schema_changed": False,
            "candidate_builder_changed": False,
            "context_builder_changed": False,
            "span_policy_changed": False,
            "adapter_changed": True,
            "normalizer_changed": True,
        },
        "stage_a_count": len(stage_a),
        "stage_b_count": len(stage_b),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_selection_report(manifest, stage_a, stage_b, reasons), encoding="utf-8")
    return {
        "experiment_id": experiment_id,
        "stage_a_candidates": len(stage_a),
        "stage_b_candidates": len(stage_b),
        "artifacts": {
            "experiment_manifest": str(manifest_path),
            "stage_a_candidates": str(stage_a_path),
            "stage_b_candidates": str(stage_b_path),
            "selection_report": str(report_path),
        },
        "model_configuration": manifest["model_configuration"],
    }


def frozen_configuration_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    prompts = base / "prompts"
    return {
        "extraction_prompt_sha256": sha256_file(prompts / "extraction_v1.txt"),
        "verification_prompt_sha256": sha256_file(prompts / "verification_v1.txt"),
        "extraction_schema_sha256": _sha256_json(EXTRACTION_SCHEMA),
        "verification_schema_sha256": _sha256_json(VERIFICATION_SCHEMA),
        "span_validator_sha256": sha256_file(base / "extraction" / "span_validator.py"),
        "field_allowed_sources_sha256": _sha256_json({key: sorted(item.value for item in values) for key, values in ALLOWED_COORDINATES_BY_FIELD.items()}),
    }


def model_configuration_status() -> dict[str, bool]:
    audit = model_config_audit()
    return {
        "base_url_configured": bool(audit["base_url_configured"]),
        "api_key_configured": bool(audit["api_key_configured"]),
        "model_configured": bool(audit["model_configured"]),
        "timeout_configured": bool(audit["timeout_configured"]),
    }


def missing_model_configuration() -> list[str]:
    mapping = {
        "GUIDELINE_LLM_BASE_URL": "base_url_configured",
        "GUIDELINE_LLM_API_KEY": "api_key_configured",
        "GUIDELINE_LLM_MODEL": "model_configured",
        "GUIDELINE_LLM_TIMEOUT_SECONDS": "timeout_configured",
    }
    status = model_configuration_status()
    return [env for env, key in mapping.items() if not status[key]]


def _select_batches(candidates: list[RecommendationCandidate], *, stage_a_size: int, stage_b_size: int) -> tuple[list[RecommendationCandidate], list[RecommendationCandidate], dict[str, list[str]]]:
    ordered = sorted(candidates, key=lambda item: (-item.candidate_score, item.doc_id, item.candidate_id))
    used: set[str] = set()
    reasons: dict[str, list[str]] = {}

    def pick(label: str, predicate) -> RecommendationCandidate | None:
        for item in ordered:
            if item.candidate_id not in used and predicate(item):
                used.add(item.candidate_id)
                reasons.setdefault(item.candidate_id, []).append(label)
                return item
        return None

    stage_a: list[RecommendationCandidate] = []
    for label, predicate in [
        ("formal_recommendation", _is_recommendation),
        ("formal_recommendation_second_doc", lambda item: _is_recommendation(item) and item.doc_id not in {cand.doc_id for cand in stage_a}),
        ("strength_or_certainty", _has_strength_or_certainty),
        ("complex_condition_or_multi_intervention", _is_complex),
        ("hard_negative", _is_hard_negative),
    ]:
        selected = pick(label, predicate)
        if selected is not None:
            stage_a.append(selected)
    for item in ordered:
        if len(stage_a) >= stage_a_size:
            break
        if item.candidate_id not in used:
            used.add(item.candidate_id)
            reasons.setdefault(item.candidate_id, []).append("stage_a_fill")
            stage_a.append(item)

    stage_b: list[RecommendationCandidate] = []
    coverage = [
        ("numbered_recommendation", lambda item: any(line[:2].strip(". )").isdigit() for line in item.candidate_text.splitlines()[:4])),
        ("we_recommend", lambda item: "we recommend" in item.candidate_text.lower()),
        ("we_suggest", lambda item: "we suggest" in item.candidate_text.lower()),
        ("is_recommended", lambda item: "is recommended" in item.candidate_text.lower()),
        ("strong_recommendation", lambda item: "strong recommendation" in item.candidate_text.lower()),
        ("conditional_recommendation", lambda item: "conditional recommendation" in item.candidate_text.lower()),
        ("good_practice", lambda item: "good practice" in item.candidate_text.lower()),
        ("methods_recommendation_word", lambda item: "method" in _path_text(item) and "recommend" in item.candidate_text.lower()),
        ("evidence_summary", lambda item: "evidence" in _path_text(item) or "evidence" in item.candidate_text.lower()),
        ("reference_section", lambda item: "reference" in _path_text(item) or "reference" in item.candidate_profile.get("block_type", "")),
        ("multi_drug_or_dosage", lambda item: any(token in item.candidate_text.lower() for token in [" plus ", "mg", " q", "dose", "duration"])),
        ("population_context", lambda item: bool(item.context_before or item.context_after) and any(token in _path_text(item) for token in ["adult", "children", "patient", "population"])),
        ("negative_or_conditional", lambda item: any(token in item.candidate_text.lower() for token in ["unless", "except", "only if", "not recommend"])),
        ("hard_negative", _is_hard_negative),
    ]
    for label, predicate in coverage:
        if len(stage_b) >= stage_b_size:
            break
        selected = pick(label, predicate)
        if selected is not None:
            stage_b.append(selected)
    for item in ordered:
        if len(stage_b) >= stage_b_size:
            break
        if item.candidate_id not in used:
            used.add(item.candidate_id)
            reasons.setdefault(item.candidate_id, []).append("stage_b_fill")
            stage_b.append(item)
    return stage_a, stage_b, reasons


def _is_recommendation(candidate: RecommendationCandidate) -> bool:
    return candidate.candidate_profile.get("block_type") == "recommendation_candidate"


def _is_hard_negative(candidate: RecommendationCandidate) -> bool:
    return bool(candidate.candidate_profile.get("hard_negative"))


def _has_strength_or_certainty(candidate: RecommendationCandidate) -> bool:
    text = candidate.candidate_text.lower()
    return any(token in text for token in ["strong", "weak", "conditional", "moderate", "low quality", "high quality", "evidence"])


def _is_complex(candidate: RecommendationCandidate) -> bool:
    text = candidate.candidate_text.lower()
    return any(token in text for token in [" plus ", " or ", "unless", "except", "only if", "choose one", "alternative", "±"])


def _path_text(candidate: RecommendationCandidate) -> str:
    return " > ".join(candidate.section_path).lower()


def _render_selection_report(manifest: dict[str, Any], stage_a: list[RecommendationCandidate], stage_b: list[RecommendationCandidate], reasons: dict[str, list[str]]) -> str:
    lines = ["# IDSA Real Model Experiment Selection", ""]
    lines.append(f"- experiment_id: {manifest['experiment_id']}")
    lines.append(f"- pilot_id: {manifest['pilot_id']}")
    lines.append(f"- stage_a_count: {len(stage_a)}")
    lines.append(f"- stage_b_count: {len(stage_b)}")
    lines.append(f"- working_tree_dirty: {manifest['working_tree_dirty']}")
    lines.append(f"- working_tree_diff_hash: {manifest['working_tree_diff_hash']}")
    lines.append("")
    lines.append("## Document Distribution")
    for doc_id, count in Counter(item.doc_id for item in stage_a + stage_b).most_common():
        lines.append(f"- {doc_id}: {count}")
    lines.append("")
    lines.append("## Candidate Profile")
    for profile, count in Counter(str(item.candidate_profile) for item in stage_a + stage_b).most_common():
        lines.append(f"- {profile}: {count}")
    lines.append("")
    lines.append("## Stage A")
    for item in stage_a:
        lines.append(f"- {item.candidate_id} | {item.doc_id} | {','.join(reasons.get(item.candidate_id, []))}")
    lines.append("")
    lines.append("## Stage B Reserved")
    for item in stage_b:
        lines.append(f"- {item.candidate_id} | {item.doc_id} | {','.join(reasons.get(item.candidate_id, []))}")
    lines.append("")
    return "\n".join(lines)




def _sha256_json(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=Path.cwd())
        return hashlib.sha1(diff).hexdigest()
    except Exception:
        return ""








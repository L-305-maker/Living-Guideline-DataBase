from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.domain.common import stable_id
from src.domain.profile import GuidelineProfile
from src.common.process_jsonl import iter_jsonl, write_jsonl

JsonDict = Dict[str, Any]

# 为每一个指南生成“指南画像”。
# 指南画像负责回答: 指南来自哪一个机构、采用何种分级制度、
# 判断置信度是多少、后续质量控制应该采用哪些策略。

DETECTOR_VERSION = "guideline_profiler_v2"

ORG_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z&.,' -]{2,140}?\b(?:"
    r"Association|Society|College|Academy|Institute|Organization|Organisation|"
    r"Task Force|Ministry|Department|Council|Network|Committee|Commission|"
    r"Agency|Center|Centre|Foundation|Initiative"
    r")(?:\s+of\s+[A-Z][A-Za-z&.,' -]{2,100})?)\b"
)

GRADING_PATTERNS: List[Tuple[str, str, re.Pattern[str], float]] = [
    ("GRADE", "generic", re.compile(r"\bGRADE\b|Grading of Recommendations|certainty of (?:the )?evidence|strong recommendation|conditional recommendation", re.I), 0.34),
    ("COR_LOE", "generic", re.compile(r"\b(?:COR|class)\s+(?:I|IIa|IIb|III)\b|\bLOE\b|level of evidence|Class of Recommendation", re.I), 0.38),
    ("VERB_BASED", "generic", re.compile(r"\b(?:offer|consider|do not offer|recommend|suggest|should|should not)\b", re.I), 0.18),
    ("NUMERIC_LETTER_GRADE", "generic", re.compile(r"\bgrade\s+[12][A-D]\b|\b[12][A-D]\b", re.I), 0.28),
    ("LETTER_GRADE", "letter_grade", re.compile(r"\bGrade\s+[ABCD]\b|I statement", re.I), 0.28),
]

COLLECTION_PRIORS: Dict[str, Tuple[str, float]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def short_text(value: Any, limit: int = 24000) -> str:
    text = normalize_text(value)
    return text[:limit]


def metadata_haystack(record: JsonDict) -> str:
    raw_pdf_path = normalize_text(record.get("raw_pdf_path"))
    raw_pdf_name = Path(raw_pdf_path).name if raw_pdf_path else ""
    fields = [
        record.get("title"),
        record.get("url"),
        record.get("issuer"),
        raw_pdf_name,
        record.get("publication_date_text"),
        record.get("last_updated_text"),
    ]
    return "\n".join(normalize_text(field) for field in fields if field)


def strong_metadata_haystack(record: JsonDict) -> str:
    return "\n".join(
        normalize_text(field)
        for field in [record.get("title"), record.get("url")]
        if field
    )


def content_haystack(record: JsonDict) -> str:
    return short_text(record.get("content"))


def add_score(scores: Dict[str, float], key: str, value: float) -> None:
    scores[key] = min(1.0, scores.get(key, 0.0) + value)


def evidence(kind: str, field: str, value: Any, target: str, weight: float, note: str = "") -> JsonDict:
    """生成一条检测证据，用于解释 profile 判断为什么成立。"""

    return {
        "kind": kind,
        "field": field,
        "value": str(value)[:500],
        "target": target,
        "weight": round(weight, 4),
        "note": note,
    }


def best_score(scores: Dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "unknown", 0.0
    key, value = max(scores.items(), key=lambda item: item[1])
    return key, round(min(value, 1.0), 4)


def detect_issuer(record: JsonDict) -> tuple[str, float, List[JsonDict]]:
    scores: Dict[str, float] = {}
    evidence_items: List[JsonDict] = []
    meta = metadata_haystack(record)
    strong_meta = strong_metadata_haystack(record)
    content = content_haystack(record)
    collection_source = str(record.get("source") or "")

    if collection_source in COLLECTION_PRIORS:
        issuer, weight = COLLECTION_PRIORS[collection_source]
        add_score(scores, issuer, weight)
        evidence_items.append(evidence("weak_collection_prior", "source", collection_source, issuer, weight))

    issuer_field = normalize_text(record.get("issuer"))
    if issuer_field:
        add_score(scores, issuer_field, 0.7)
        evidence_items.append(evidence("metadata_issuer_field", "issuer", issuer_field, issuer_field, 0.7))

    for field, text, weight in [
        ("title_or_url", strong_meta, 0.4),
        ("metadata", meta, 0.25),
        ("content", content, 0.18),
    ]:
        for match in ORG_NAME_RE.finditer(text):
            issuer = normalize_text(match.group(1)).strip(" ,.;:")
            if len(issuer) < 8 or len(issuer) > 180:
                continue
            add_score(scores, issuer, weight)
            evidence_items.append(evidence("generic_issuer_phrase", field, issuer, issuer, weight))

    issuer, confidence = best_score(scores)
    if confidence < 0.25:
        return "unknown", confidence, evidence_items
    return issuer, confidence, evidence_items



def detect_grading(record: JsonDict) -> tuple[str, str, float, List[JsonDict]]:
    scores: Dict[str, float] = {}
    versions: Dict[str, str] = {}
    evidence_items: List[JsonDict] = []
    meta = metadata_haystack(record)
    content = content_haystack(record)

    for system, version, pattern, weight in GRADING_PATTERNS:
        if pattern.search(meta):
            add_score(scores, system, min(weight, 0.22))
            versions.setdefault(system, version)
            evidence_items.append(evidence("metadata_grading_match", "metadata", pattern.pattern, system, min(weight, 0.22)))
        if pattern.search(content):
            add_score(scores, system, weight)
            versions.setdefault(system, version)
            evidence_items.append(evidence("content_grading_match", "content", pattern.pattern, system, weight))

    system, confidence = best_score(scores)
    if confidence < 0.25:
        return "unknown", "unknown", confidence, evidence_items
    return system, versions.get(system, "generic"), confidence, evidence_items


def resolution_state(issuer_confidence: float, grading_confidence: float, override_state: str = "") -> str:
    if override_state == "human_confirmed":
        return "human_confirmed"
    if grading_confidence < 0.25:
        if issuer_confidence >= 0.75:
            return "auto_low"
        return "unresolved"
    confidence = min(issuer_confidence, grading_confidence) if grading_confidence else issuer_confidence
    if confidence >= 0.75:
        return "auto_high"
    if confidence >= 0.5:
        return "auto_med"
    if confidence >= 0.25:
        return "auto_low"
    return "unresolved"


def load_policies(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "policies" not in data or "default_policy" not in data:
        raise ValueError("profile policies must contain policies and default_policy")
    return data


def policy_for_system(policies: JsonDict, grading_system: str) -> JsonDict:
    policy = (policies.get("policies") or {}).get(grading_system)
    if isinstance(policy, dict):
        return policy
    return policies.get("default_policy") or {}


def load_overrides(path: str | Path) -> Dict[str, JsonDict]:
    override_path = Path(path)
    if not override_path.exists():
        return {}
    overrides: Dict[str, JsonDict] = {}
    for row in iter_jsonl(override_path):
        if row.get("__example__"):
            continue
        record_id = str(row.get("record_id") or "")
        if record_id:
            overrides[record_id] = row
    return overrides


def guideline_id_from_record(record: JsonDict) -> str:
    seed = record.get("guideline_seed")
    if isinstance(seed, dict):
        return str(seed.get("guideline_id") or "")
    return ""


def _detected_profile(record: JsonDict) -> JsonDict:
    issuer, issuer_confidence, issuer_evidence = detect_issuer(record)
    grading_system, grading_version, grading_confidence, grading_evidence = detect_grading(record)
    return {
        "issuer": issuer,
        "issuer_confidence": issuer_confidence,
        "grading_system": grading_system,
        "grading_version": grading_version,
        "grading_confidence": grading_confidence,
        "profile_evidence": issuer_evidence + grading_evidence,
    }


def _apply_profile_override(profile: JsonDict, override: JsonDict, record_id: str) -> None:
    if not override:
        return
    profile["issuer"] = str(override.get("issuer") or profile["issuer"])
    profile["issuer_confidence"] = float(override.get("issuer_confidence") or profile["issuer_confidence"])
    profile["grading_system"] = str(override.get("grading_system") or profile["grading_system"])
    profile["grading_version"] = str(override.get("grading_system_version") or profile["grading_version"])
    profile["grading_confidence"] = float(override.get("grading_confidence") or profile["grading_confidence"])
    override_evidence = override.get("evidence_for_profile")
    if isinstance(override_evidence, list):
        profile["profile_evidence"].extend(item for item in override_evidence if isinstance(item, dict))
    profile["profile_evidence"].append(evidence("curated_override_applied", "record_id", record_id, "profile", 1.0))


def _profile_raw_payload(record: JsonDict, policies: JsonDict, override: JsonDict) -> JsonDict:
    return {
        "url": record.get("url", ""),
        "raw_pdf_path": record.get("raw_pdf_path", ""),
        "url_provenance": record.get("url_provenance", ""),
        "policy_version": policies.get("policy_version", ""),
        "override_applied": bool(override),
    }


def build_profile(record: JsonDict, policies: JsonDict, overrides: Dict[str, JsonDict]) -> GuidelineProfile:
    record_id = str(record.get("record_id") or "")
    guideline_id = guideline_id_from_record(record)
    profile = _detected_profile(record)
    override = overrides.get(record_id, {})
    _apply_profile_override(profile, override, record_id)
    policy = policy_for_system(policies, profile["grading_system"])
    state = resolution_state(
        issuer_confidence=profile["issuer_confidence"],
        grading_confidence=profile["grading_confidence"],
        override_state=str(override.get("resolution_state") or ""),
    )
    profile_id = stable_id("guideline_profile", record_id, guideline_id, DETECTOR_VERSION)
    return GuidelineProfile(
        profile_id=profile_id,
        record_id=record_id,
        guideline_id=guideline_id or None,
        collection_source=str(record.get("source") or ""),
        profile_scope="document",
        issuer=profile["issuer"],
        issuer_confidence=round(profile["issuer_confidence"], 4),
        guideline_title=str(record.get("title") or ""),
        published_year=str(record.get("published_year") or "") or None,
        grading_system=profile["grading_system"],
        grading_system_version=profile["grading_version"],
        grading_confidence=round(profile["grading_confidence"], 4),
        dimensions=policy.get("dimensions", {}),
        accepted_strength_terms=list(policy.get("accepted_strength_terms") or []),
        accepted_certainty_terms=list(policy.get("accepted_certainty_terms") or []),
        accepted_direction_terms=list(policy.get("accepted_direction_terms") or []),
        canonical_mapping_note=policy.get("canonical_mapping_note"),
        evidence_for_profile=profile["profile_evidence"],
        resolution_state=state,
        detector_version=DETECTOR_VERSION,
        profiled_at=utc_now(),
        raw_payload=_profile_raw_payload(record, policies, override),
    )


def summarize_profiles(profiles: List[JsonDict], inputs: List[str]) -> JsonDict:
    issuer_counts: Counter[str] = Counter()
    grading_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    collection_counts: Counter[str] = Counter()
    low_confidence: List[JsonDict] = []
    for profile in profiles:
        issuer_counts[str(profile.get("issuer") or "unknown")] += 1
        grading_counts[str(profile.get("grading_system") or "unknown")] += 1
        state_counts[str(profile.get("resolution_state") or "unknown")] += 1
        collection_counts[str(profile.get("collection_source") or "unknown")] += 1
        if (
            float(profile.get("issuer_confidence") or 0.0) < 0.5
            or float(profile.get("grading_confidence") or 0.0) < 0.5
            or profile.get("resolution_state") in {"unresolved", "auto_low"}
        ):
            low_confidence.append(
                {
                    "record_id": profile.get("record_id"),
                    "title": profile.get("guideline_title"),
                    "collection_source": profile.get("collection_source"),
                    "issuer": profile.get("issuer"),
                    "issuer_confidence": profile.get("issuer_confidence"),
                    "grading_system": profile.get("grading_system"),
                    "grading_confidence": profile.get("grading_confidence"),
                    "resolution_state": profile.get("resolution_state"),
                }
            )
    return {
        "profile_items": len(profiles),
        "inputs": inputs,
        "issuer_counts": dict(issuer_counts),
        "grading_system_counts": dict(grading_counts),
        "resolution_state_counts": dict(state_counts),
        "collection_source_counts": dict(collection_counts),
        "low_confidence_count": len(low_confidence),
        "low_confidence_samples": low_confidence[:30],
        "detector_version": DETECTOR_VERSION,
        "created_at": utc_now(),
    }


def build_profiles_file(
    inputs: List[str | Path],
    policies_input: str | Path,
    overrides_input: str | Path,
    profiles_output: str | Path,
    report_output: str | Path,
) -> JsonDict:
    """从输入记录生成 guideline profile JSONL，并输出检测摘要报告。"""

    policies = load_policies(policies_input)
    overrides = load_overrides(overrides_input)
    profile_rows: List[JsonDict] = []
    input_strings = [str(path) for path in inputs]
    for input_path in inputs:
        for record in iter_jsonl(input_path):
            profile_rows.append(build_profile(record, policies, overrides).to_dict())
    report = summarize_profiles(profile_rows, input_strings)
    report["policies_input"] = str(policies_input)
    report["overrides_input"] = str(overrides_input)
    write_jsonl(profiles_output, profile_rows)
    write_jsonl(report_output, [report])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build document-level guideline profiles from cleaned source records.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Cleaned SourceRecord JSONL files.")
    parser.add_argument("--policies-input", default="data/reference/guideline_profile_policies.json")
    parser.add_argument("--overrides-input", default="data/reference/guideline_profile_overrides.jsonl")
    parser.add_argument("--profiles-output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_profiles_file(
        inputs=args.inputs,
        policies_input=args.policies_input,
        overrides_input=args.overrides_input,
        profiles_output=args.profiles_output,
        report_output=args.report_output,
    )
    print(
        "profiles={profile_items} issuers={issuer_counts} grading={grading_system_counts} states={resolution_state_counts}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()

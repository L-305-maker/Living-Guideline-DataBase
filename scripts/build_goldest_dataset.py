from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEARCH_SEEDS: list[dict[str, Any]] = [
    {
        "query_id": "search_001",
        "query": "KDIGO 2024 chronic kidney disease evaluation management guideline albuminuria eGFR",
        "source_institution": "KDIGO",
        "clinical_department": "肾内科",
        "time_range": "2024-2024",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "kdigo_2024_0b77a9e32ca6",
        "expected_canonical_title": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
        "rationale": "Document-level query with institution, year, disease, and core CKD evaluation terms.",
    },
    {
        "query_id": "search_002",
        "query": "KDIGO 2022 diabetes management chronic kidney disease guideline SGLT2 HbA1c",
        "source_institution": "KDIGO",
        "clinical_department": "肾内科",
        "time_range": "2022-2022",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "kdigo_2022_d2889a8b699e",
        "expected_canonical_title": "KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease",
        "rationale": "Targets the KDIGO diabetes in CKD guideline by year and treatment/monitoring anchors.",
    },
    {
        "query_id": "search_003",
        "query": "GINA 2024 Global Strategy for Asthma diagnosis management adults adolescents children",
        "source_institution": "GINA",
        "clinical_department": "呼吸内科",
        "time_range": "2024-2024",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "gina_2024_b228dd55c5c9",
        "expected_canonical_title": "GINA 2024 Global Strategy for Asthma Management and Prevention",
        "rationale": "Uses GINA, year, asthma strategy, and patient-group anchors.",
    },
    {
        "query_id": "search_004",
        "query": "GOLD 2025 COPD diagnosis management prevention report exacerbation bronchodilator",
        "source_institution": "GOLD",
        "clinical_department": "呼吸内科",
        "time_range": "2025-2025",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "gold_2025_a1a47993d068",
        "expected_canonical_title": "GOLD 2025 Global Strategy for the Diagnosis, Management, and Prevention of COPD",
        "rationale": "Targets the full GOLD 2025 COPD report with disease and exacerbation anchors.",
    },
    {
        "query_id": "search_005",
        "query": "AASM 2019 positive airway pressure obstructive sleep apnea CPAP APAP BPAP guideline",
        "source_institution": "AASM",
        "clinical_department": "呼吸内科",
        "time_range": "2019-2019",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "aasm_2019_321cbacd79ae",
        "expected_canonical_title": "AASM Clinical Practice Guideline for Positive Airway Pressure Treatment of Adult Obstructive Sleep Apnea",
        "rationale": "Document query with society, year, PAP modality, and OSA terms.",
    },
    {
        "query_id": "search_006",
        "query": "ATS IDSA 2019 adult community acquired pneumonia guideline outpatient amoxicillin oseltamivir",
        "source_institution": "IDSA",
        "clinical_department": "感染内科",
        "time_range": "2019-2019",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "idsa_2019_d79f88501db7",
        "expected_canonical_title": "ATS/IDSA 2019 Guideline for Adult Community-acquired Pneumonia",
        "rationale": "Combines institution, year, CAP, and treatment anchors.",
    },
    {
        "query_id": "search_007",
        "query": "WHO consolidated guidelines HIV testing services 2015 pre-test post-test PITC",
        "source_institution": "WHO",
        "clinical_department": "感染内科",
        "time_range": "2015-2015",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "who_2015_cfa6f2fb27b1",
        "expected_canonical_title": "WHO Consolidated Guidelines on HIV Testing Services",
        "rationale": "Targets the WHO HTS document with testing-service terminology.",
    },
    {
        "query_id": "search_008",
        "query": "NICE routine preoperative tests elective surgery ASA grade ECG kidney function 2016",
        "source_institution": "NICE",
        "clinical_department": "心血管内科",
        "time_range": "2016-2016",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "nice_2016_99c7b963228c",
        "expected_canonical_title": "NICE Routine Preoperative Tests for Elective Surgery",
        "rationale": "Document search anchored by NICE, preoperative testing, ASA grade, ECG, and kidney function.",
    },
    {
        "query_id": "search_009",
        "query": "NICE clopidogrel modified-release dipyridamole occlusive vascular event peripheral arterial disease 2024",
        "source_institution": "NICE",
        "clinical_department": "心血管内科",
        "time_range": "2024-2024",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "nice_2024_d1c95ed6ef62",
        "expected_canonical_title": "NICE Clopidogrel and Modified-release Dipyridamole for Occlusive Vascular Events",
        "rationale": "Targets a specific NICE technology appraisal by drug names and indication.",
    },
    {
        "query_id": "search_010",
        "query": "中国偏头痛诊治指南 2022 版 解读 CGRP 曲坦 吉泮 预防性治疗",
        "source_institution": "CMA",
        "clinical_department": "神经内科",
        "time_range": "2022-2022",
        "publication_date": None,
        "recency_boost": False,
        "expected_primary_doc_id": "cma_2022_e2e7858e7417",
        "expected_canonical_title": "中国偏头痛诊治指南(2022版)解读",
        "rationale": "Chinese-language document query anchored by the guideline title and updated migraine therapies.",
    },
]


RETRIEVE_SEEDS: list[dict[str, Any]] = [
    {
        "query_id": "retrieve_001",
        "query": "KDIGO 2024 成人 CKD eGFR 20 或尿白蛋白 ACR 200 时是否推荐使用 SGLT2i",
        "source_institution": "KDIGO",
        "clinical_department": "肾内科",
        "time_range": "2024-2024",
        "publication_date": None,
        "expected_primary_doc_id": "kdigo_2024_0b77a9e32ca6",
        "expected_primary_chunk_id": "kdigo_2024_0b77a9e32ca6#chunk_00203",
        "expected_canonical_title": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
        "rationale": "Chunk contains the KDIGO SGLT2i recommendation thresholds for adults with CKD.",
    },
    {
        "query_id": "retrieve_002",
        "query": "KDIGO 2022 糖尿病合并 CKD HbA1c 长期血糖控制每年监测几次",
        "source_institution": "KDIGO",
        "clinical_department": "肾内科",
        "time_range": "2022-2022",
        "publication_date": None,
        "expected_primary_doc_id": "kdigo_2022_d2889a8b699e",
        "expected_primary_chunk_id": "kdigo_2022_d2889a8b699e#chunk_00345",
        "expected_canonical_title": "KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease",
        "rationale": "Chunk states HbA1c monitoring frequency and when to increase it.",
    },
    {
        "query_id": "retrieve_003",
        "query": "GINA 2024 哮喘急性发作氧疗血氧饱和度目标 93 95 住院患者",
        "source_institution": "GINA",
        "clinical_department": "呼吸内科",
        "time_range": "2024-2024",
        "publication_date": None,
        "expected_primary_doc_id": "gina_2024_b228dd55c5c9",
        "expected_primary_chunk_id": "gina_2024_b228dd55c5c9#chunk_00650",
        "expected_canonical_title": "GINA 2024 Global Strategy for Asthma Management and Prevention",
        "rationale": "Chunk includes controlled oxygen therapy saturation targets for acute asthma care.",
    },
    {
        "query_id": "retrieve_004",
        "query": "GOLD 2025 COPD 急性加重初始支气管扩张剂 SABA SAMA 长效支气管扩张剂",
        "source_institution": "GOLD",
        "clinical_department": "呼吸内科",
        "time_range": "2025-2025",
        "publication_date": None,
        "expected_primary_doc_id": "gold_2025_a1a47993d068",
        "expected_primary_chunk_id": "gold_2025_a1a47993d068#chunk_00406",
        "expected_canonical_title": "GOLD 2025 Global Strategy for the Diagnosis, Management, and Prevention of COPD",
        "rationale": "Chunk lists initial bronchodilator and maintenance therapy recommendations for COPD exacerbations.",
    },
    {
        "query_id": "retrieve_005",
        "query": "AASM 2019 OSA PAP 治疗 CPAP APAP 是否优先于 BPAP 以及教育和远程监测干预",
        "source_institution": "AASM",
        "clinical_department": "呼吸内科",
        "time_range": "2019-2019",
        "publication_date": None,
        "expected_primary_doc_id": "aasm_2019_321cbacd79ae",
        "expected_primary_chunk_id": "aasm_2019_321cbacd79ae#chunk_00003",
        "expected_canonical_title": "AASM Clinical Practice Guideline for Positive Airway Pressure Treatment of Adult Obstructive Sleep Apnea",
        "rationale": "Chunk aggregates PAP recommendations including CPAP/APAP over BPAP and adherence interventions.",
    },
    {
        "query_id": "retrieve_006",
        "query": "ATS IDSA 2019 成人社区获得性肺炎门诊经验性治疗 无合并症 阿莫西林 多西环素 大环内酯",
        "source_institution": "IDSA",
        "clinical_department": "感染内科",
        "time_range": "2019-2019",
        "publication_date": None,
        "expected_primary_doc_id": "idsa_2019_d79f88501db7",
        "expected_primary_chunk_id": "idsa_2019_d79f88501db7#chunk_00047",
        "expected_canonical_title": "ATS/IDSA 2019 Guideline for Adult Community-acquired Pneumonia",
        "rationale": "Chunk contains outpatient CAP antibiotic regimens and risk-factor notes.",
    },
    {
        "query_id": "retrieve_007",
        "query": "WHO 2015 HIV testing services STI 结核 TB 服务中是否应常规提供 HIV 检测",
        "source_institution": "WHO",
        "clinical_department": "感染内科",
        "time_range": "2015-2015",
        "publication_date": None,
        "expected_primary_doc_id": "who_2015_cfa6f2fb27b1",
        "expected_primary_chunk_id": "who_2015_cfa6f2fb27b1#chunk_00210",
        "expected_canonical_title": "WHO Consolidated Guidelines on HIV Testing Services",
        "rationale": "Chunk discusses HIV testing integration with TB and STI services.",
    },
    {
        "query_id": "retrieve_008",
        "query": "NICE 术前常规检查 ASA 3 或 ASA 4 肾功能 ECG 肺功能 动脉血气 何时考虑",
        "source_institution": "NICE",
        "clinical_department": "心血管内科",
        "time_range": "2016-2016",
        "publication_date": None,
        "expected_primary_doc_id": "nice_2016_99c7b963228c",
        "expected_primary_chunk_id": "nice_2016_99c7b963228c#chunk_00028",
        "expected_canonical_title": "NICE Routine Preoperative Tests for Elective Surgery",
        "rationale": "Chunk contains ASA 3/4 routine preoperative testing recommendations.",
    },
    {
        "query_id": "retrieve_009",
        "query": "NICE 氯吡格雷 改良释放双嘧达莫 闭塞性血管事件 外周动脉疾病 推荐适用人群",
        "source_institution": "NICE",
        "clinical_department": "心血管内科",
        "time_range": "2024-2024",
        "publication_date": None,
        "expected_primary_doc_id": "nice_2024_d1c95ed6ef62",
        "expected_primary_chunk_id": "nice_2024_d1c95ed6ef62#chunk_00003",
        "expected_canonical_title": "NICE Clopidogrel and Modified-release Dipyridamole for Occlusive Vascular Events",
        "rationale": "Chunk defines the population covered by the NICE antiplatelet appraisal.",
    },
    {
        "query_id": "retrieve_010",
        "query": "中国偏头痛 2022 CGRP 单克隆抗体 预防性治疗 一线用药 疗效评估 12 到 18 个月",
        "source_institution": "CMA",
        "clinical_department": "神经内科",
        "time_range": "2022-2022",
        "publication_date": None,
        "expected_primary_doc_id": "cma_2022_e2e7858e7417",
        "expected_primary_chunk_id": "cma_2022_e2e7858e7417#chunk_00032",
        "expected_canonical_title": "中国偏头痛诊治指南(2022版)解读",
        "rationale": "Chunk summarizes CGRP monoclonal antibody positioning and follow-up timing in migraine prevention.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the latest document-backed goldest search/retrieve datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/evidence"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def compact_preview(text: str, max_chars: int = 500) -> str:
    preview = " ".join(text.split())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 1].rstrip() + "…"


def load_docs(data_dir: Path) -> dict[str, dict[str, Any]]:
    docs = read_jsonl(data_dir / "documents.jsonl")
    return {doc["doc_id"]: doc for doc in docs}


def load_chunk(data_dir: Path, doc_id: str, chunk_id: str) -> dict[str, Any]:
    chunk_path = data_dir / "chunks" / f"{doc_id}.jsonl"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Missing chunk file: {chunk_path}")
    for chunk in read_jsonl(chunk_path):
        if chunk.get("chunk_id") == chunk_id:
            return chunk
    raise KeyError(f"Missing chunk {chunk_id} in {chunk_path}")


def enrich_search_rows(docs_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEARCH_SEEDS:
        doc_id = seed["expected_primary_doc_id"]
        doc = docs_by_id.get(doc_id)
        if doc is None:
            raise KeyError(f"Search seed references missing doc_id: {doc_id}")
        rows.append(
            {
                "query_id": seed["query_id"],
                "task": "search",
                "query": seed["query"],
                "topk": 10,
                "source_institution": seed["source_institution"],
                "clinical_department": seed["clinical_department"],
                "time_range": seed["time_range"],
                "publication_date": seed["publication_date"],
                "recency_boost": seed["recency_boost"],
                "expected_doc_ids": [doc_id],
                "expected_primary_doc_id": doc_id,
                "expected_title": doc.get("title", ""),
                "expected_canonical_title": seed["expected_canonical_title"],
                "expected_source_institution": doc.get("source_institution", ""),
                "expected_clinical_department": doc.get("clinical_department", ""),
                "expected_publication_date": doc.get("publication_date", ""),
                "source_file": doc.get("source_file", ""),
                "markdown_clean_path": doc.get("markdown_clean_path", ""),
                "cleaning_quality": doc.get("cleaning_quality", ""),
                "pdf_text_quality": doc.get("pdf_text_quality", ""),
                "abstract_preview": compact_preview(doc.get("abstract", ""), 360),
                "rationale": seed["rationale"],
            }
        )
    return rows


def enrich_retrieve_rows(data_dir: Path, docs_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in RETRIEVE_SEEDS:
        doc_id = seed["expected_primary_doc_id"]
        chunk_id = seed["expected_primary_chunk_id"]
        doc = docs_by_id.get(doc_id)
        if doc is None:
            raise KeyError(f"Retrieve seed references missing doc_id: {doc_id}")
        chunk = load_chunk(data_dir, doc_id, chunk_id)
        if chunk.get("doc_id") != doc_id:
            raise ValueError(f"Chunk {chunk_id} belongs to {chunk.get('doc_id')}, expected {doc_id}")
        rows.append(
            {
                "query_id": seed["query_id"],
                "task": "retrieve",
                "query": seed["query"],
                "topk": 10,
                "source_institution": seed["source_institution"],
                "clinical_department": seed["clinical_department"],
                "time_range": seed["time_range"],
                "publication_date": seed["publication_date"],
                "relevant_chunk_ids": [chunk_id],
                "relevant_doc_ids": [doc_id],
                "expected_primary_chunk_id": chunk_id,
                "expected_primary_doc_id": doc_id,
                "expected_title": doc.get("title", ""),
                "expected_canonical_title": seed["expected_canonical_title"],
                "expected_source_institution": doc.get("source_institution", ""),
                "expected_clinical_department": doc.get("clinical_department", ""),
                "expected_publication_date": doc.get("publication_date", ""),
                "expected_chunk_type": chunk.get("chunk_type", ""),
                "expected_section_path": chunk.get("section_path") or [],
                "expected_token_count": chunk.get("token_count", 0),
                "is_reference_section": bool(chunk.get("is_reference_section")),
                "source_file": chunk.get("source_file", doc.get("source_file", "")),
                "markdown_clean_path": chunk.get("markdown_clean_path", doc.get("markdown_clean_path", "")),
                "content_preview": compact_preview(chunk.get("content", ""), 500),
                "rationale": seed["rationale"],
            }
        )
    return rows


def validate_rows(search_rows: list[dict[str, Any]], retrieve_rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(search_rows) != 10:
        errors.append(f"Expected 10 search rows, got {len(search_rows)}")
    if len(retrieve_rows) != 10:
        errors.append(f"Expected 10 retrieve rows, got {len(retrieve_rows)}")

    search_ids = [row["query_id"] for row in search_rows]
    retrieve_ids = [row["query_id"] for row in retrieve_rows]
    if len(search_ids) != len(set(search_ids)):
        errors.append("Duplicate search query_id values")
    if len(retrieve_ids) != len(set(retrieve_ids)):
        errors.append("Duplicate retrieve query_id values")

    for row in search_rows:
        if row["cleaning_quality"] != "ok":
            errors.append(f"{row['query_id']} target doc cleaning_quality is {row['cleaning_quality']!r}")
        if row["pdf_text_quality"] != "ok":
            errors.append(f"{row['query_id']} target doc pdf_text_quality is {row['pdf_text_quality']!r}")

    for row in retrieve_rows:
        if row["expected_token_count"] <= 0:
            errors.append(f"{row['query_id']} target chunk has non-positive token count")
        if row["is_reference_section"]:
            errors.append(f"{row['query_id']} target chunk is in a reference section")

    return {
        "ok": not errors,
        "errors": errors,
        "search_count": len(search_rows),
        "retrieve_count": len(retrieve_rows),
        "search_doc_ids": [row["expected_primary_doc_id"] for row in search_rows],
        "retrieve_chunk_ids": [row["expected_primary_chunk_id"] for row in retrieve_rows],
        "retrieve_doc_ids": [row["expected_primary_doc_id"] for row in retrieve_rows],
    }


def write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    readme = f"""# Goldest Evaluation Dataset

This directory contains a small document-backed goldest dataset built from the latest `data/evidence/documents.jsonl` and `data/evidence/chunks/*.jsonl`.

- `search_goldest.jsonl`: 10 document-search cases. Judge by `expected_doc_ids` / `expected_primary_doc_id`.
- `retrieve_goldest.jsonl`: 10 chunk-retrieval cases. Judge by `relevant_chunk_ids` and `relevant_doc_ids`.
- `goldest_dataset_report.json`: build metadata and validation summary.

The cases are intentionally high-signal: each query includes a stable guideline anchor such as institution, year, disease, treatment, or section concept. The retrieve cases avoid reference sections and require positive token counts.

Build time: {report["built_at"]}
Validation status: {"ok" if report["validation"]["ok"] else "failed"}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "goldest"
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_by_id = load_docs(data_dir)
    search_rows = enrich_search_rows(docs_by_id)
    retrieve_rows = enrich_retrieve_rows(data_dir, docs_by_id)
    validation = validate_rows(search_rows, retrieve_rows)

    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "documents_path": str(data_dir / "documents.jsonl"),
        "chunks_dir": str(data_dir / "chunks"),
        "output_dir": str(output_dir),
        "validation": validation,
    }
    if not validation["ok"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))

    write_jsonl(output_dir / "search_goldest.jsonl", search_rows)
    write_jsonl(output_dir / "retrieve_goldest.jsonl", retrieve_rows)
    (output_dir / "goldest_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    write_readme(output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""List eligible guideline PDFs for the eight reviewed sources by department."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz

from crawler.wjes_guidelines import classify_title


SOURCES = {
    "cns_pdf": ("CNS", "神经外科"),
    "sccm_guidelines": ("SCCM", "重症医学科"),
    "esicm_guidelines": ("ESICM", "重症医学科"),
    "aaaai_pdf": ("AAAAI JTF", "过敏/变态反应科"),
    "aapmr_pdf": ("AAPM&R", "康复医学科"),
    "bts_pdf": ("British Transplantation Society", "器官移植科"),
    "wjes_pdf": ("World Journal of Emergency Surgery", "急诊外科"),
    "asps_pdf": ("ASPS", "整形外科"),
}

WJES_TITLE_OVERRIDES = {
    "11b25b9c1cfeb12ea4e1.pdf": "Correction: Surgical stabilization of rib fractures (SSRF): the WSES and CWIS position paper",
    "11c7584afd2b5161847b.pdf": "Intra-abdominal infections survival guide: a position statement by the Global Alliance For Infections In Surgery",
    "162da1c509e1575d1650.pdf": "Thoracic trauma WSES-AAST guidelines",
    "a6018aa904e03a8bda84.pdf": "ECLAPTE: Effective Closure of LAParoTomy in Emergency—2023 WSES guidelines",
    "daf3752ee30232974562.pdf": "Correction: ECLAPTE: Effective Closure of LAParoTomy in Emergency—2023 WSES guidelines",
}

RELATED = [
    ("小儿科", r"\b(pediatric|paediatric|children|childhood|infant|neonatal)\b"),
    ("感染科", r"\b(infection|infectious|sepsis|septic|antimicrobial|antibiotic|covid|virus|viral)\b"),
    ("心血管科", r"\b(cardiac|cardiovascular|heart|arrhythm|vascular)\b"),
    ("呼吸科", r"\b(respiratory|lung|pulmonary|airway|ventilation|thoracic)\b"),
    ("消化科", r"\b(gastro|intestinal|bowel|colorectal|colon|stomach|esoph|pancrea)\b"),
    ("肝胆外科", r"\b(liver|hepatic|biliary|gallbladder|cholecyst)\b"),
    ("肾脏科", r"\b(kidney|renal)\b"),
    ("骨科/脊柱外科", r"\b(spine|spinal|vertebr|orthop|fracture|joint|knee|hip|shoulder)\b"),
    ("神经科", r"\b(neuro|brain|cranial|stroke|seizure|epilep|headache)\b"),
    ("肿瘤科", r"\b(cancer|carcin|tumou?r|neoplasm|oncolog)\b"),
    ("皮肤科", r"\b(dermat|eczema|urticaria|skin|soft tissue)\b"),
    ("麻醉/疼痛科", r"\b(anesth|analgesi|pain|sedation)\b"),
    ("营养科", r"\b(nutrition|enteral|parenteral)\b"),
    ("妇产科", r"\b(pregnan|maternal|obstetric|gynec|breastfeeding)\b"),
    ("创伤外科", r"\b(trauma|injur|wound|burn)\b"),
]


def clean_title(value: str) -> str:
    return " ".join(value.replace("\x00", " ").split()).strip(" -_")


def useful_metadata_title(value: str) -> bool:
    lowered = value.casefold()
    return len(value) >= 12 and not any(
        marker in lowered for marker in (
            "microsoft word", "untitled", "neu-d-", "acrobat", "report title",
            "project mandate template", "insert guideline owner", "introduction",
        )
    )


def filename_title(path: Path) -> str:
    stem = re.sub(r"_[0-9a-f]{10,20}$", "", path.stem)
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return clean_title(stem.replace("_", " ").replace("-", " "))


def pdf_title(path: Path) -> tuple[str, str]:
    with fitz.open(path) as document:
        metadata = clean_title(document.metadata.get("title") or "")
        text = clean_title(" ".join(page.get_text() for page in document[:2]))[:6000]
    if useful_metadata_title(metadata):
        return metadata, text
    cited = re.search(
        r"Please cite this guideline as:\s*(.+?)(?:\s+www\.|\s+Published\s|\s+View\s)",
        text,
        re.IGNORECASE,
    )
    if cited:
        return clean_title(cited.group(1)), text
    fallback = filename_title(path)
    if len(fallback) >= 12 and not re.fullmatch(r"[0-9a-f]+", fallback):
        return fallback, text
    return clean_title(text[:240]) or path.stem, text


def eligibility(source: str, title: str, text: str, path: Path) -> tuple[bool, str]:
    lowered = title.casefold()
    if source == "aaaai_pdf" and (
        "the new normal" in lowered or lowered == "allergy and immunology practice parameters and guidelines"
    ):
        return False, "commentary_not_parameter"
    if source == "bts_pdf" and "guideline development policy" in lowered:
        return False, "guideline_methodology"
    if source == "bts_pdf":
        if "leaflet" in lowered or "for people who want to donate a kidney" in lowered:
            return False, "patient_leaflet"
        if not re.search(
            r"guideline|guidance|recommendation|position|consensus|standard|framework|addendum",
            f"{title} {text[:2500]}",
            re.IGNORECASE,
        ):
            return False, "companion_without_guidance_signal"
    if source == "aapmr_pdf":
        if "dear " in text[:1500].casefold() and "endorsement" in text[:2500].casefold():
            return False, "endorsement_letter"
        if "aapm&r guideline review" in text[:600].casefold():
            return False, "internal_guideline_review"
    if source == "wjes_pdf":
        kind, reason = classify_title(title)
        return kind is not None, reason or kind or "wjes_guidance"
    if source == "asps_pdf":
        administrative = re.search(
            r"performance|methodolog|measurement|ethics|legislative|coding|faq|wellness|contribution|reform|course|leadership",
            f"{title} {path.name}",
            re.IGNORECASE,
        )
        guidance = re.search(r"clinical practice guideline|practice parameter", title, re.IGNORECASE)
        return bool(guidance and not administrative), "non_guideline_asps_material" if not guidance or administrative else "guideline"
    return True, "official_guideline_pdf_scope"


def related_departments(title: str, text: str, primary: str) -> list[str]:
    value = f"{title} {text[:1800]}"
    departments = [department for department, pattern in RELATED if re.search(pattern, value, re.IGNORECASE)]
    return [department for department in departments if department != primary]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(raw_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    included, excluded = [], []
    hash_counts = Counter()
    for source, (source_name, primary) in SOURCES.items():
        for path in sorted((raw_root / source).rglob("*.pdf")):
            title, text = pdf_title(path)
            if source == "wjes_pdf" and path.name in WJES_TITLE_OVERRIDES:
                title = WJES_TITLE_OVERRIDES[path.name]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            keep, reason = eligibility(source, title, text, path)
            row = {
                "primary_department": primary,
                "related_departments": "、".join(related_departments(title, text, primary)),
                "source": source_name,
                "source_directory": source,
                "title": title,
                "file": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "review_reason": reason,
            }
            if keep:
                included.append(row)
                hash_counts[digest] += 1
            else:
                excluded.append(row)

    for row in included:
        row["duplicate_content"] = hash_counts[row["sha256"]] > 1
    fields = [
        "primary_department", "related_departments", "source", "source_directory", "title",
        "file", "sha256", "bytes", "review_reason", "duplicate_content",
    ]
    write_csv(output / "guideline_pdf_details.csv", included, fields)
    write_csv(output / "excluded_pdf_details.csv", excluded, fields[:-1])

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in included:
        grouped[(row["primary_department"], row["source"])].append(row)
    summary_rows = []
    source_rows = {(department, source): rows for (department, source), rows in grouped.items()}
    for _, (source, department) in SOURCES.items():
        rows = source_rows.get((department, source), [])
        summary_rows.append({
            "primary_department": department,
            "source": source,
            "pdf_files": len(rows),
            "unique_pdf_content": len({row["sha256"] for row in rows}),
        })
    write_csv(output / "department_source_summary.csv", summary_rows, list(summary_rows[0]))

    lines = [
        "# PDF 指南按科室和来源明细", "",
        f"合格 PDF 文件：{len(included)}；唯一内容：{len(hash_counts)}；排除文件：{len(excluded)}。", "",
        "主科室由来源专业领域确定；相关科室由标题和前两页关键词补充。", "",
        "## 汇总", "", "| 主科室 | 来源 | PDF 文件 | 唯一内容 |", "|---|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['primary_department']} | {row['source']} | {row['pdf_files']} | {row['unique_pdf_content']} |"
        )
    for (department, source), rows in sorted(grouped.items()):
        lines.extend(["", f"## {department} — {source}", ""])
        for row in sorted(rows, key=lambda item: item["title"].casefold()):
            related = f"；相关：{row['related_departments']}" if row["related_departments"] else ""
            lines.append(f"- {row['title']}{related} — `{row['file']}`")
    lines.extend(["", "## 排除文件", ""])
    for row in sorted(excluded, key=lambda item: (item["source"], item["title"].casefold())):
        lines.append(f"- {row['source']}：{row['title']}（{row['review_reason']}）— `{row['file']}`")
    (output / "GUIDELINE_PDFS_BY_DEPARTMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "included_pdf_files": len(included),
        "unique_pdf_content": len(hash_counts),
        "excluded_pdf_files": len(excluded),
        "duplicate_content_groups": sum(count > 1 for count in hash_counts.values()),
        "by_source": dict(Counter(row["source"] for row in included)),
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_pdf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.raw_root, args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

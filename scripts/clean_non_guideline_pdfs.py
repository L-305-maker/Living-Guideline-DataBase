"""Audit raw PDFs and remove files that are clearly not medical guidelines."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import fitz


POSITIVE = re.compile(
    r"guideline|guidance|recommendation|consensus|position[\s_-]*statement|"
    r"practice[\s_-]*parameter|clinical[\s_-]*standard|standard[\s_-]*of[\s_-]*care|"
    r"best[\s_-]*practice|care[\s_-]*pathway|指南|共识|诊疗规范|临床规范|诊疗方案",
    re.I,
)

CATEGORIES = {
    "conference_event": re.compile(
        r"conference|congress|annual[\s_-]*meeting|meeting[\s_-]*(?:program|programme)|"
        r"final[\s_-]*(?:program|programme)|abstract[\s_-]*(?:book|submission)|poster|"
        r"exhibitor|sponsor(?:ship)?|symposium|webinar|workshop|会议|大会|年会|投稿通知|征文通知",
        re.I,
    ),
    "organization_admin": re.compile(
        r"annual[\s_-]*report|financial[\s_-]*statement|constitution|bylaws?|governance|"
        r"strategic[\s_-]*plan|membership|subscription[\s_-]*fee|registration[\s_-]*(?:form|terms)|"
        r"application[\s_-]*form|booking[\s_-]*form|nomination[\s_-]*form|award[\s_-]*application|"
        r"grant[\s_-]*application|scholarship|job[\s_-]*description|organogram|"
        r"conflict[\s_-]*of[\s_-]*interest|privacy[\s_-]*policy|gdpr|cookie[\s_-]*policy|"
        r"terms[\s_-]*(?:and|&)[\s_-]*conditions|章程|会员|注册表|申请表|报名表|年度报告|隐私政策",
        re.I,
    ),
    "publication_material": re.compile(
        r"author[\s_-]*instructions?|instructions?[\s_-]*for[\s_-]*authors?|manuscript[\s_-]*submission|"
        r"editorial[\s_-]*policy|newsletter|magazine|brochure|press[\s_-]*release|podcast|"
        r"presentation[\s_-]*slides?|patient[\s_-]*(?:information|leaflet|handout|booklet)|"
        r"parent[\s_-]*(?:information|leaflet|handout|booklet)|宣传册|新闻稿|患者手册|家长手册|投稿须知",
        re.I,
    ),
    "research_only": re.compile(
        r"systematic[\s_-]*review|meta[\s_-]*analysis|randomi[sz]ed[\s_-]*(?:controlled[\s_-]*)?trial|"
        r"cohort[\s_-]*study|case[\s_-]*(?:report|series)|study[\s_-]*protocol|scoping[\s_-]*review|"
        r"literature[\s_-]*review|cross[\s_-]*sectional[\s_-]*study|survey[\s_-]*study|clinical[\s_-]*trial|"
        r"系统评价|荟萃分析|病例报告|队列研究|随机对照试验|研究方案",
        re.I,
    ),
    "quality_measure": re.compile(
        r"measurement[\s_-]*(?:set|manual)|quality[\s_-]*(?:measure|indicator)|"
        r"audit[\s_-]*tool|scorecard|implementation[\s_-]*toolkit|质量指标|评价指标|审核工具",
        re.I,
    ),
}

STRONG_EXCLUSION = re.compile(
    r"annual[\s_-]*report|financial[\s_-]*statement|constitution|bylaws?|strategic[\s_-]*plan|"
    r"membership|subscription[\s_-]*fee|registration[\s_-]*(?:form|terms)|application[\s_-]*form|"
    r"booking[\s_-]*form|nomination[\s_-]*form|job[\s_-]*description|organogram|privacy[\s_-]*policy|"
    r"gdpr|cookie[\s_-]*policy|terms[\s_-]*(?:and|&)[\s_-]*conditions|"
    r"meeting[\s_-]*(?:program|programme)|final[\s_-]*(?:program|programme)|"
    r"abstract[\s_-]*(?:book|submission)|poster|exhibitor|webinar|"
    r"author[\s_-]*instructions?|instructions?[\s_-]*for[\s_-]*authors?|manuscript[\s_-]*submission|"
    r"newsletter|magazine|brochure|press[\s_-]*release|presentation[\s_-]*slides?|"
    r"patient[\s_-]*(?:leaflet|handout|booklet)|parent[\s_-]*(?:leaflet|handout|booklet)|"
    r"章程|会员|注册表|申请表|报名表|年度报告|隐私政策|投稿通知|征文通知|宣传册|患者手册|家长手册|投稿须知",
    re.I,
)


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value, flags=re.UNICODE).strip()


def inspect_pdf(path: Path) -> dict[str, str | int]:
    error = ""
    metadata_title = ""
    first_text = ""
    pages = 0
    try:
        with fitz.open(path) as doc:
            pages = doc.page_count
            metadata_title = (doc.metadata or {}).get("title", "") or ""
            chunks = [doc.load_page(i).get_text("text") for i in range(min(2, pages))]
            first_text = "\n".join(chunks)[:6000]
    except Exception as exc:  # malformed or non-PDF content is itself review evidence
        error = f"{type(exc).__name__}: {exc}"

    lines = [normalize(line) for line in first_text.splitlines() if normalize(line)]
    visible_title = " ".join(lines[:8])[:1200]
    name_scope = " ".join((normalize(path.stem), normalize(metadata_title)))
    positive_scope = " ".join((name_scope, " ".join(lines[:8])))
    category = ""
    evidence = ""
    for name, pattern in CATEGORIES.items():
        match = pattern.search(name_scope)
        if match:
            category = name
            evidence = match.group(0)
            break

    # Strongly administrative formats are never guideline files. Otherwise a clinical
    # guideline/consensus signal wins over broad words such as "conference" or "governance".
    if error:
        decision, category, evidence = "review", "pdf_error", error[:300]
    elif STRONG_EXCLUSION.search(name_scope):
        decision = "remove"
    elif POSITIVE.search(positive_scope):
        decision, category = "keep", category or "guideline_signal"
    elif category:
        decision = "remove"
    else:
        decision, category = "review", "no_clear_signal"

    stat = path.stat()
    return {
        "decision": decision,
        "category": category,
        "evidence": evidence,
        "source": path.parent.name,
        "file": path.name,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "pages": pages,
        "metadata_title": metadata_title[:1000],
        "visible_title": visible_title,
        "error": error,
    }


def manifest_ref(row: dict, manifest: Path) -> Path | None:
    raw = row.get("local_path") or row.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    for base in (manifest.parent, Path.cwd()):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def sync_manifests(root: Path, removed: set[Path]) -> tuple[int, int]:
    changed = removed_rows = 0
    for manifest in root.rglob("*.jsonl"):
        lines = manifest.read_text(encoding="utf-8-sig").splitlines()
        kept: list[str] = []
        removed_here = 0
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            ref = manifest_ref(row, manifest)
            if ref in removed:
                removed_here += 1
            else:
                kept.append(line)
        if removed_here:
            manifest.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            changed += 1
            removed_rows += removed_here
    return changed, removed_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw_pdf"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("*.pdf"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect_pdf, paths))

    removed: set[Path] = set()
    if args.apply:
        for row in rows:
            if row["decision"] != "remove":
                continue
            path = Path(str(row["path"])).resolve()
            if root not in path.parents:
                raise RuntimeError(f"refusing to delete outside root: {path}")
            path.unlink()
            removed.add(path)
        manifests_changed, manifest_rows_removed = sync_manifests(root, removed)
    else:
        manifests_changed = manifest_rows_removed = 0

    columns = list(rows[0]) if rows else []
    with (report_dir / "audit.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "root": str(root),
        "scanned": len(rows),
        "applied": args.apply,
        "decisions": dict(Counter(str(row["decision"]) for row in rows)),
        "categories": dict(Counter(str(row["category"]) for row in rows)),
        "removed_files": len(removed),
        "removed_bytes": sum(int(row["size_bytes"]) for row in rows if row["decision"] == "remove"),
        "manifests_changed": manifests_changed,
        "manifest_rows_removed": manifest_rows_removed,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()





"""Classify raw guideline PDFs and (optionally) move non-guideline files to data/excluded.

Usage (PowerShell, from repo root):
    python -B scripts/classify_guideline_pdfs.py --root data/raw_pdf          # dry-run report only
    python -B scripts/classify_guideline_pdfs.py --root data/raw_pdf --apply  # move non-guideline files

Output: --report (default reports/guideline_classify.jsonl), one JSON line per PDF:
    path, source_dir, filename, guide_score, non_guide_score, verdict, reasons

verdict: guideline | non_guideline | unknown   (only non_guideline is moved with --apply)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pymupdf is required: python -m pip install pymupdf") from exc

GUIDE_TOKENS = [
    "clinical practice guideline", "practice guideline", "practice parameter",
    "guideline update", "key action statement", "strong recommendation",
    "evidence-based", "evidence based", "systematic review", "meta-analysis",
    "consensus", "recommendation", "the diagnosis and management",
    "the management of", "treatment of", "diagnosis and treatment",
    "指南", "共识", "推荐意见", "临床实践", "诊治",
    "专家共识", "专家建议", "推荐", "建议", "适应证", "禁忌证",
    "中国", "中华医学会", "诊疗规范", "管理指南",
]
NON_GUIDE_TOKENS = [
    "annual report", "financial report", "donors", "job description",
    "vice president", "director of", "manager of", "coordinator",
    "apply now", "careers", "workforce", "membership dues", "donation",
    "fundraising", "code for interactions", "newsletter", "annual meeting",
    "gala", "job title", "salary range", "position summary",
    "年度报告", "财务", "招聘", "捐赠",
]
GUIDE_NAME_TOKENS = ["cpg", "guideline", "practice_parameter", "guidance",
                    "专家共识", "共识", "指南", "诊治指南", "规范", "专家建议"]
NON_GUIDE_NAME_TOKENS = ["annual_report", "financial", "donor", "job_description", "workforce", "newsletter",
                         "年度报告", "招聘", "捐赠", "会议通知", "征稿"]


def extract_head(path: Path, limit: int = 20000, max_pages: int | None = None) -> str:
    """全量扫描 PDF：默认读取全部页，limit 仅作为返回字符上限（分类只需关键词，无需全文）。"""
    try:
        doc = fitz.open(str(path))
        page_count = doc.page_count
        if max_pages is None:
            max_pages = page_count
        pages = [doc[i].get_text("text") for i in range(min(max_pages, page_count))]
        doc.close()
        return " ".join(chr(10).join(pages).split())[:limit].lower()
    except Exception as exc:  # noqa: BLE001
        return "__error__:" + str(exc)


def classify(path: Path) -> dict[str, Any]:
    head = extract_head(path)
    name = path.name.lower()
    guide_score = sum(1 for tok in GUIDE_TOKENS if tok in head) + sum(1 for tok in GUIDE_NAME_TOKENS if tok in name)
    non_guide_score = sum(1 for tok in NON_GUIDE_TOKENS if tok in head) + sum(1 for tok in NON_GUIDE_NAME_TOKENS if tok in name)
    reasons = [t for t in GUIDE_TOKENS if t in head][:4] + [t for t in GUIDE_NAME_TOKENS if t in name][:2]
    non_reasons = [t for t in NON_GUIDE_TOKENS if t in head][:4] + [t for t in NON_GUIDE_NAME_TOKENS if t in name][:2]
    if non_guide_score and non_guide_score > guide_score:
        verdict = "non_guideline"
    elif guide_score >= 1:
        verdict = "guideline"
    else:
        verdict = "unknown"
    return {
        "path": str(path),
        "source_dir": path.parent.name,
        "filename": path.name,
        "guide_score": guide_score,
        "non_guide_score": non_guide_score,
        "verdict": verdict,
        "reasons": reasons,
        "non_reasons": non_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw_pdf")
    parser.add_argument("--report", default="reports/guideline_classify.jsonl")
    parser.add_argument("--excluded-dir", default="data/excluded")
    parser.add_argument("--apply", action="store_true", help="actually move non_guideline files (default: dry-run)")
    args = parser.parse_args()

    root = Path(args.root)
    report = Path(args.report)
    excluded = Path(args.excluded_dir)
    report.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.pdf")):
        if excluded in path.parents:
            continue
        rows.append(classify(path))

    with report.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + chr(10))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"scanned {len(rows)} pdfs -> {report}  {counts}")

    if args.apply:
        moved, errors = [], []
        for row in rows:
            if row["verdict"] != "non_guideline":
                continue
            src = Path(row["path"])
            dest = excluded / row["source_dir"] / row["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dest))
                moved.append(row["filename"])
            except Exception as exc:  # noqa: BLE001
                errors.append((row["filename"], str(exc)))
        print(f"moved {len(moved)} files to {excluded}")
        if errors:
            print("errors:", errors)
    else:
        non_guide = [r["path"] for r in rows if r["verdict"] == "non_guideline"]
        if non_guide:
            print("dry-run: add --apply to move these non_guideline files:")
            for p in non_guide:
                print("  ", p)


if __name__ == "__main__":
    main()

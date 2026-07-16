"""Merge department classification and guideline-likeness audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SOURCE_NAMES = {
    "aaaai_pdf": "AAAAI JTF", "aan_pdf": "AAN", "aao_hns_pdf": "AAO-HNS",
    "aaos_pdf": "AAOS", "aapd_pdf": "AAPD", "aapmr_pdf": "AAPM&R",
    "aasld_pdf": "AASLD", "aasm_pdf": "AASM", "aats_pdf": "AATS",
    "acog_pdf": "ACOG", "acpgbi_pdf": "ACPGBI", "ada_pdf": "ADA",
    "aha_pdf": "AHA", "ameriburn_pdf": "American Burn Association", "apsa_pdf": "APSA",
    "asps_pdf": "ASPS", "asrm_pdf": "ASRM", "ats_pdf": "ATS",
    "bapras_pdf": "BAPRAS", "boa_pdf": "BOA", "bsp_pdf": "BSP",
    "bspd_pdf": "BSPD", "btf_pdf": "Brain Trauma Foundation",
    "bts_pdf": "British Transplantation Society", "cdc_pdf": "CDC", "cma_pdf": "CMA",
    "cns_pdf": "CNS", "consensus": "Consensus collection", "eacts_pdf": "EACTS",
    "eaes_pdf": "EAES", "east_pdf": "EAST", "eras_pdf": "ERAS",
    "ernica_pdf": "ERNICA", "esc_pdf": "ESC", "escp_pdf": "ESCP",
    "eshre_pdf": "ESHRE", "esicm_guidelines": "ESICM", "espghan_pdf": "ESPGHAN",
    "esvs_pdf": "ESVS", "gina_pdf": "GINA", "gold_pdf": "GOLD",
    "idsa_pdf": "IDSA", "isbi_pdf": "ISBI", "iwgdf_pdf": "IWGDF",
    "jacc_pdf": "JACC", "kdigo_pdf": "KDIGO", "naspghan_pdf": "NASPGHAN",
    "nice_pdf": "NICE", "pmc_pdf": "PMC", "posna_pdf": "POSNA", "rch_pdf": "RCH",
    "rcog_pdf": "RCOG", "rcpch_pdf": "RCPCH", "sages_pdf": "SAGES",
    "sccm_guidelines": "SCCM", "sdcep_pdf": "SDCEP", "sign_pdf": "SIGN",
    "sts_pdf": "STS", "svs_pdf": "SVS", "trip_pdf": "TRIP",
    "uspstf_pdf": "USPSTF", "vadod_pdf": "VA/DoD", "who_pdf": "WHO",
    "wjes_pdf": "World Journal of Emergency Surgery", "wounds_pdf": "Wounds",
    "wses_pdf": "WSES",
}


def norm(value: str) -> str:
    return str(Path(value).resolve()).casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--curated-included", type=Path)
    parser.add_argument("--curated-excluded", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    classified = pd.read_csv(args.classification, encoding="utf-8-sig")
    audit = pd.read_csv(args.audit, encoding="utf-8-sig")
    classified["join_path"] = classified.file.map(norm)
    audit["join_path"] = audit.path.map(lambda value: str(Path(value).resolve()).casefold())
    audit = audit.rename(columns={
        "decision": "guideline_status", "category": "guideline_category",
        "evidence": "guideline_evidence", "metadata_title": "pdf_metadata_title",
        "visible_title": "pdf_visible_title", "error": "pdf_error",
    })
    audit_columns = [
        "join_path", "guideline_status", "guideline_category", "guideline_evidence",
        "pages", "size_bytes", "pdf_metadata_title", "pdf_visible_title", "pdf_error",
    ]
    merged = classified.merge(audit[audit_columns], on="join_path", how="left", validate="one_to_one")
    if merged.guideline_status.isna().any():
        raise RuntimeError("classification and audit paths do not fully match")

    if args.curated_included:
        curated = pd.read_csv(args.curated_included, encoding="utf-8-sig")
        included_paths = set(curated.file.map(norm))
        mask = merged.join_path.isin(included_paths)
        merged.loc[mask, ["guideline_status", "guideline_category", "guideline_evidence"]] = [
            "keep", "curated_guideline", "eight_source_content_review",
        ]
    if args.curated_excluded:
        curated = pd.read_csv(args.curated_excluded, encoding="utf-8-sig")
        reasons = dict(zip(curated.file.map(norm), curated.review_reason))
        mask = merged.join_path.isin(reasons)
        merged.loc[mask, "guideline_status"] = "remove"
        merged.loc[mask, "guideline_category"] = "curated_non_guideline"
        merged.loc[mask, "guideline_evidence"] = merged.loc[mask, "join_path"].map(reasons)

    merged["source_name"] = merged.source.map(SOURCE_NAMES).fillna(merged.source)
    merged["department_review_status"] = merged.secondary_status.map(
        lambda value: "manual_review" if value == "manual_review" else "accepted"
    )
    merged["guideline_status"] = merged.guideline_status.map({
        "keep": "confirmed_signal", "review": "needs_content_review", "remove": "clear_non_guideline",
    })
    merged["needs_any_review"] = (
        (merged.department_review_status == "manual_review")
        | (merged.guideline_status == "needs_content_review")
    )
    columns = [
        "file", "source", "source_name", "department", "department_review_status",
        "method", "confidence", "second_department", "second_confidence",
        "guideline_status", "guideline_category", "guideline_evidence", "needs_any_review",
        "evidence_title", "pdf_metadata_title", "pdf_visible_title", "pages", "size_bytes", "pdf_error",
    ]
    merged = merged[columns].sort_values(["source_name", "department", "file"])
    merged.to_csv(output / "all_pdf_source_department.csv", index=False, encoding="utf-8-sig")
    confirmed = merged[merged.guideline_status == "confirmed_signal"]
    confirmed.to_csv(output / "confirmed_guidelines.csv", index=False, encoding="utf-8-sig")
    merged[merged.needs_any_review].to_csv(output / "needs_review.csv", index=False, encoding="utf-8-sig")
    merged[merged.guideline_status == "clear_non_guideline"].to_csv(
        output / "clear_non_guideline.csv", index=False, encoding="utf-8-sig"
    )

    cross = (
        merged.groupby(["source_name", "department", "guideline_status"])
        .size().reset_index(name="count")
        .sort_values(["source_name", "count"], ascending=[True, False])
    )
    cross.to_csv(output / "source_department_counts_all.csv", index=False, encoding="utf-8-sig")
    confirmed_cross = (
        confirmed.groupby(["source_name", "department"]).size().reset_index(name="count")
        .sort_values(["source_name", "count"], ascending=[True, False])
    )
    confirmed_cross.to_csv(
        output / "source_department_counts_confirmed.csv", index=False, encoding="utf-8-sig"
    )
    department_counts = (
        confirmed.groupby("department").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    department_counts.to_csv(output / "department_counts_confirmed.csv", index=False, encoding="utf-8-sig")

    source_summary = merged.groupby("source_name").agg(
        total=("file", "size"),
        confirmed=("guideline_status", lambda values: (values == "confirmed_signal").sum()),
        guideline_review=("guideline_status", lambda values: (values == "needs_content_review").sum()),
        clear_non_guideline=("guideline_status", lambda values: (values == "clear_non_guideline").sum()),
        department_review=("department_review_status", lambda values: (values == "manual_review").sum()),
    ).reset_index().sort_values("total", ascending=False)
    source_summary.to_csv(output / "source_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "pdf_files": int(len(merged)),
        "sources": int(merged.source_name.nunique()),
        "departments": int(merged.department.nunique()),
        "guideline_status": {key: int(value) for key, value in merged.guideline_status.value_counts().items()},
        "department_manual_review": int((merged.department_review_status == "manual_review").sum()),
        "needs_any_review": int(merged.needs_any_review.sum()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

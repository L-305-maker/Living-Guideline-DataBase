"""Best-effort metadata extraction from Markdown and file paths."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


PATH_HINTS = {
    "aan_pdf": "AAN",
    "aasld_pdf": "AASLD",
    "aasm_pdf": "AASM",
    "aha_pdf": "AHA",
    "ats_pdf": "ATS",
    "cdc_pdf": "CDC",
    "china_pdf": "China",
    "cma_pdf": "CMA",
    "esc_pdf": "ESC",
    "gina_pdf": "GINA",
    "gold_pdf": "GOLD",
    "idsa_pdf": "IDSA",
    "jacc_pdf": "JACC",
    "kdigo_pdf": "KDIGO",
    "nice_pdf": "NICE",
    "pmc_pdf": "PMC",
    "sign_pdf": "SIGN",
    "trip_pdf": "TRIP",
    "uspstf_pdf": "USPSTF",
    "vadod_pdf": "VA/DOD",
    "who_pdf": "WHO",
}
YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
FULLWIDTH_RE = re.compile(r"[\uff01-\uff5e]")
BAD_TITLE_CHARS_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_title_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"^\d{5,}[_\-\s]+", "", stem)
    stem = re.sub(r"^\d{5,}", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return "Untitled"
    return stem if re.search(r"[\u4e00-\u9fff]", stem) else stem.title()


def extract_source_institution(pdf_path: str | Path, text: str = "") -> str:
    path = Path(pdf_path)
    for part in path.parts:
        hint = PATH_HINTS.get(part.lower())
        if hint:
            return hint
    lower = f"{path.name}\n{text[:4000]}".lower()
    if "world health organization" in lower or re.search(r"\bwho\b", lower):
        return "WHO"
    if "nice" in lower or "national institute for health and care excellence" in lower:
        return "NICE"
    if "centers for disease control" in lower or re.search(r"\bcdc\b", lower):
        return "CDC"
    if "va/dod" in lower or "department of veterans affairs" in lower:
        return "VA/DOD"
    if "american academy of neurology" in lower or re.search(r"\baan\b", lower):
        return "AAN"
    if "american heart association" in lower or re.search(r"\baha\b", lower):
        return "AHA"
    if "american thoracic society" in lower or re.search(r"\bats\b", lower):
        return "ATS"
    if "global initiative for asthma" in lower or re.search(r"\bgina\b", lower):
        return "GINA"
    if "u.s. preventive services task force" in lower or "us preventive services task force" in lower:
        return "USPSTF"
    if "american association for the study of liver diseases" in lower or re.search(r"\baasld\b", lower):
        return "AASLD"
    return "Unknown"


def helper_normalize_digits(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def helper_candidate_years(text: str) -> list[int]:
    normalized = helper_normalize_digits(text)
    years = [int(match.group(1)) for match in YEAR_RE.finditer(normalized)]
    return [year for year in years if 2012 <= year <= 2026]


def extract_publication_date(pdf_path: str | Path, text: str = "") -> str:
    # Prefer file names because journal headers often contain unrelated historical years.
    file_years = helper_candidate_years(str(pdf_path))
    if file_years:
        return f"{file_years[0]}-01-01"
    haystack = f"{Path(pdf_path).name}\n{text[:12000]}"
    years = helper_candidate_years(haystack)
    if years:
        return f"{years[0]}-01-01"
    return "unknown"


def helper_title_score(title: str) -> float:
    normalized = helper_normalize_digits(title or "").strip()
    if not normalized:
        return -100.0
    visible = [ch for ch in normalized if not ch.isspace()]
    if not visible:
        return -100.0
    letters_digits = sum(1 for ch in visible if ch.isalpha() or ch.isdigit() or "\u4e00" <= ch <= "\u9fff")
    cjk = sum(1 for ch in visible if "\u4e00" <= ch <= "\u9fff")
    controls = sum(1 for ch in visible if unicodedata.category(ch)[0] == "C")
    ascii_symbols = sum(1 for ch in visible if ord(ch) < 128 and not ch.isalnum())
    score = letters_digits / len(visible)
    score += min(cjk, 12) * 0.03
    score -= controls * 0.5
    score -= ascii_symbols / max(1, len(visible)) * 0.5
    if len(normalized) < 6:
        score -= 0.4
    if len(normalized) > 160:
        score -= 0.2
    if re.search(r"\b(vol|no|journal|杂志|第\d+卷|第\d+期)\b", normalized, re.I):
        score -= 0.15
    return score


def is_suspicious_title(title: str) -> bool:
    normalized = helper_normalize_digits(title or "").strip()
    if len(normalized) < 4 or BAD_TITLE_CHARS_RE.search(normalized):
        return True
    journal_header_patterns = [
        r"中华.{0,12}杂志.{0,80}第\s*\d+\s*卷.{0,40}第\s*\d+\s*期",
        r"Chin\s+J\s+.{0,80}\bVol\.?\s*\d+.{0,40}\bNo\.?\s*\d+",
        r"\bVol\.?\s*\d+.{0,40}\bNo\.?\s*\d+",
        r"^\d+\s+年\s*\d+\s*月\s*第\s*\d+\s*卷\s*第\s*\d+\s*期",
    ]
    if any(re.search(pattern, normalized, re.I) for pattern in journal_header_patterns):
        return True
    visible = [ch for ch in normalized if not ch.isspace()]
    if not visible:
        return True
    letters_digits = sum(1 for ch in visible if ch.isalpha() or ch.isdigit() or "\u4e00" <= ch <= "\u9fff")
    ascii_symbols = sum(1 for ch in visible if ord(ch) < 128 and not ch.isalnum())
    if letters_digits / len(visible) < 0.25:
        return True
    if len(visible) > 30 and ascii_symbols / len(visible) > 0.45:
        return True
    return False


def clean_title(title: str) -> str:
    title = helper_normalize_digits(title or "")
    title = BAD_TITLE_CHARS_RE.sub("", title)
    title = re.sub(r"<!--.*?-->", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^[\W_]+|[\W_]+$", "", title, flags=re.UNICODE).strip()
    return title[:180].strip()


def extract_title(markdown: str, pdf_path: str | Path) -> str:
    candidates: list[str] = []
    match = HEADING_RE.search(markdown)
    if match and match.group(1).strip():
        candidates.append(match.group(1).strip())
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("<!--"):
            candidates.append(stripped[:220])
            if len(candidates) >= 12:
                break
    candidates.append(clean_title_from_filename(pdf_path))
    cleaned = [clean_title(candidate) for candidate in candidates]
    usable = [candidate for candidate in cleaned if candidate and not is_suspicious_title(candidate)]
    if usable:
        return max(usable, key=helper_title_score)
    return max((candidate for candidate in cleaned if candidate), key=helper_title_score, default=clean_title_from_filename(pdf_path))


def extract_abstract(markdown: str, max_chars: int = 1200) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        lines.append(stripped)
        if sum(len(item) for item in lines) >= max_chars:
            break
    return " ".join(lines)[:max_chars]

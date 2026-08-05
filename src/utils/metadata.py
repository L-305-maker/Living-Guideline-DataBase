# 从 Markdown 与文件路径中"尽力"提取元数据（title / publication_date / source_institution / abstract）。
#
# 关键约束：
# - PATH_HINTS：data/raw_pdf/ 下目录与发布机构短名白名单（路径优先）；
# - TEXT_SOURCE_PATTERNS：从正文文本二次识别发布机构；
# - CMA_TEXT_PATTERNS + CMA_GENERIC_ORG_RE：中华医学会等中文机构识别；
# - extract_publication_date 优先取文件名年份（期刊页眉可能含与发布日无关的历史年份）。
"""Best-effort metadata extraction from Markdown and file paths."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


# 完整来源白名单：与 data/raw_pdf/ 下目录一一对应（含 gin / sccm_guidelines 两个非 *_pdf 目录）。
# 新增 PDF 来源时同步在本表追加映射；helper_source_from_path 按此优先匹配。
PATH_HINTS = {
    "aaaai_pdf": "AAAAI",
    "aan_pdf": "AAN",
    "aao_hns_pdf": "AAO-HNS",
    "aaos_pdf": "AAOS",
    "aapmr_pdf": "AAPM&R",
    "aasld_pdf": "AASLD",
    "aasm_pdf": "AASM",
    "aats_pdf": "AATS",
    "acog_pdf": "ACOG",
    "acpgbi_pdf": "ACPGBI",
    "ada_pdf": "ADA",
    "aha_pdf": "AHA",
    "ameriburn_pdf": "ABA",
    "apsa_pdf": "APSA",
    "ascrs_pdf": "ASCRS",
    "asps_pdf": "ASPS",
    "asrm_pdf": "ASRM",
    "ats_pdf": "ATS",
    "bapras_pdf": "BAPRAS",
    "bc_pdf": "BC",
    "boa_pdf": "BOA",
    "bsp_pdf": "BSP",
    "bspd_pdf": "BSPD",
    "btf_pdf": "BTF",
    "bts_pdf": "BTS",
    "ccs_pdf": "CCS",
    "cdc_pdf": "CDC",
    "china_pdf": "China",
    "cma_pdf": "CMA",
    "cns_pdf": "CNS",
    "cua_pdf": "CUA",
    "eacts_pdf": "EACTS",
    "eaes_pdf": "EAES",
    "east_pdf": "EAST",
    "eras_pdf": "ERAS",
    "ernica_pdf": "ERNICA",
    "ers_pdf": "ERS",
    "esc_pdf": "ESC",
    "escp_pdf": "ESCP",
    "eshre_pdf": "ESHRE",
    "esicm_pdf": "ESICM",
    "espghan_pdf": "ESPGHAN",
    "esvs_pdf": "ESVS",
    "gin": "GIN",
    "gina_pdf": "GINA",
    "gold_pdf": "GOLD",
    "idsa_pdf": "IDSA",
    "inesss_pdf": "INESSS",
    "isbi_pdf": "ISBI",
    "iwgdf_pdf": "IWGDF",
    "jacc_pdf": "JACC",
    "kdigo_pdf": "KDIGO",
    "naspghan_pdf": "NASPGHAN",
    "nice_pdf": "NICE",
    "pmc_pdf": "PMC",
    "posna_pdf": "POSNA",
    "rch_pdf": "RCH",
    "rcog_pdf": "RCOG",
    "rcpch_pdf": "RCPCH",
    "sages_pdf": "SAGES",
    "sccm_guidelines": "SCCM",
    "sdcep_pdf": "SDCEP",
    "sign_pdf": "SIGN",
    "sts_pdf": "STS",
    "svs_pdf": "SVS",
    "trip_pdf": "TRIP",
    "uspstf_pdf": "USPSTF",
    "vadod_pdf": "VA/DOD",
    "vsgbi_pdf": "VSGBI",
    "who_pdf": "WHO",
    "wjes_pdf": "WSES",
    "wounds_pdf": "Wounds",
    "wses_pdf": "WSES",
}

# 文本二次识别的发布机构（按优先级高的先匹配）；未命中时回退 "Unknown"。
TEXT_SOURCE_PATTERNS = [
    ("WHO", ("world health organization", "世界卫生组织")),
    ("NICE", ("national institute for health and care excellence", "nice guideline", "nice guidelines")),
    ("CDC", ("centers for disease control", "centers for disease control and prevention")),
    ("VA/DOD", ("va/dod", "department of veterans affairs")),
    ("AAN", ("american academy of neurology",)),
    ("AHA", ("american heart association",)),
    ("ATS", ("american thoracic society",)),
    ("GINA", ("global initiative for asthma",)),
    ("USPSTF", ("u.s. preventive services task force", "us preventive services task force")),
    ("AASLD", ("american association for the study of liver diseases",)),
]

# CMA（中华医学会）相关的中英文模式：包含 医学会 / 协会 / 学会 / 联盟 等关键词。
CMA_TEXT_PATTERNS = (
    "中华医学会",
    "中国医师协会",
    "中华预防医学会",
    "中国医药教育协会",
    "中国抗癌协会",
    "中国医院协会",
    "中国研究型医院学会",
    "中国老年医学学会",
    "中国康复医学会",
    "中国中西医结合学会",
    "中国营养学会",
    "中国卒中学会",
    "中国药学会",
    "中国医疗保健国际交流促进会",
    "国家感染性疾病临床医学研究中心",
    "国家传染病医学中心",
    "国家卫生健康委",
    "疾病预防控制中心",
    "预防医学会",
    "cma.j.",
    "Chinese Medical Association",
    "Chinese Medical Doctor Association",
    "Chinese Preventive Medicine Association",
)

# CMA 兜底正则：中/英文『中华/中国 + ... + 医学会/协会/学会/联盟/委员会/专家组』
CMA_GENERIC_ORG_RE = re.compile(r"(?:中华|中国)[\u4e00-\u9fff]{0,30}(?:医学会|医师协会|协会|学会|联盟|委员会|专家组)")

# 摘要边界：识别【摘要】/【提要】/Abstract: 等标记，便于只取标题块之前的内容。
ABSTRACT_BOUNDARY_RE = re.compile(r"(\u3010\s*(?:摘要|提要)\s*\u3011|(?:摘要|提要|Abstract)\s*[:\uff1a])", re.I)
YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# 全角字符 → 半角：用于年份提取前的归一化
FULLWIDTH_RE = re.compile(r"[\uff01-\uff5e]")

# 坏标题字符：\ufffd（替换符）、C0 控制字符（除 \t \n 等常见空白）、DEL 与 C1 控制区。
BAD_TITLE_CHARS_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_title_from_filename(path: str | Path) -> str:
    """从文件名派生标题：去数字前缀 + ID-style 字符，把 _/- 转空格。

    中文文件名保留原样；纯英文文件名走 title case 提升可读性。
    空 stem 兜底返回 'Untitled'。
    """
    stem = Path(path).stem
    stem = re.sub(r"^\d{5,}[_\-\s]+", "", stem)
    stem = re.sub(r"^\d{5,}", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return "Untitled"
    return stem if re.search(r"[\u4e00-\u9fff]", stem) else stem.title()


# 视为"通用"的目录名：跳过这些路径段，不参与发布机构识别。
GENERIC_SOURCE_DIRS = {"raw_pdf", "consensus", "pdfs", "documents", "pages"}


def helper_source_from_path(path: Path) -> str | None:
    """从 PDF 路径中按目录段推断发布机构。

    行为：
    - 跳过 GENERIC_SOURCE_DIRS 段（这些不含发布机构信息）；
    - 先按 PATH_HINTS 白名单精确匹配；
    - 否则若以 `_pdf` 结尾（且长度 > 4），用去掉 `_pdf` 后缀作为短名。
    """
    for part in path.parts:
        lowered = part.lower()
        if lowered in GENERIC_SOURCE_DIRS:
            continue
        hint = PATH_HINTS.get(lowered)
        if hint:
            return hint
        if lowered.endswith("_pdf") and len(lowered) > 4:
            return lowered[:-4]
    return None


def helper_source_header(text: str, max_chars: int = 3000) -> str:
    """取正文头部到摘要边界为止的内容，用于发布机构二次识别。

    注意：摘要之后会引用大量其它机构，截在边界前避免误判。
    """
    head = (text or "")[:max_chars]
    match = ABSTRACT_BOUNDARY_RE.search(head)
    return head[: match.start()] if match else head


def helper_is_consensus_path(path: Path) -> bool:
    """判定路径是否位于 consensus/ 目录下（影响 source_institution 默认值）。"""
    return any(part.lower() == "consensus" for part in path.parts)


def helper_has_cma_signal(text: str) -> bool:
    """判定文本是否含有 CMA（中华医学会及其相关学会）的特征短语。"""
    if any(pattern in text for pattern in CMA_TEXT_PATTERNS):
        return True
    return bool(CMA_GENERIC_ORG_RE.search(text))


def extract_source_institution(pdf_path: str | Path, text: str = "", title: str = "") -> str:
    """推断 PDF 的发布机构：路径白名单 → CMA 模式 → TEXT_SOURCE_PATTERNS → Unknown。

    仅使用标题/文件名/摘要前 3000 字，避免正文中引用的机构被误判为发布方。
    consensus/ 目录下的 PDF 显式返回 'Unknown'（待人工标注）。
    """
    path = Path(pdf_path)
    path_hint = helper_source_from_path(path)
    if path_hint:
        return path_hint

    header = helper_source_header(text)
    source_haystack = f"{path.name}\n{title}\n{header}"
    if helper_has_cma_signal(source_haystack):
        return "CMA"
    if helper_is_consensus_path(path):
        return "Unknown"

    lower = source_haystack.lower()
    for source, patterns in TEXT_SOURCE_PATTERNS:
        if any(pattern.lower() in lower for pattern in patterns):
            return source
    return "Unknown"


def helper_normalize_digits(text: str) -> str:
    """NFKC 规范化：把全角数字 ０-９ 转半角 0-9 等。"""
    return unicodedata.normalize("NFKC", text)


def helper_candidate_years(text: str) -> list[int]:
    """从字符串中提取 [2012, 2026] 区间内的 4 位年份，去重保序。"""
    normalized = helper_normalize_digits(text)
    years = [int(match.group(1)) for match in YEAR_RE.finditer(normalized)]
    return [year for year in years if 2012 <= year <= 2026]


def extract_publication_date(pdf_path: str | Path, text: str = "") -> str:
    """推断 PDF 发布日期：文件名年份优先 → 正文头部年份 → 'unknown'。

    关键设计：期刊页眉可能含历史年份（成立时间等），文件名年份更准确；
    日期默认补 -01-01（月日未知，按 1 月 1 日兜底）。
    """
    file_years = helper_candidate_years(str(pdf_path))
    if file_years:
        return f"{file_years[0]}-01-01"
    haystack = f"{Path(pdf_path).name}\n{text[:12000]}"
    years = helper_candidate_years(haystack)
    if years:
        return f"{years[0]}-01-01"
    return "unknown"


def helper_title_score(title: str) -> float:
    """标题质量评分（越高越像真标题）。

    加分项：字母/数字/CJK 占比；
    减分项：控制字符、ASCII 符号、短标题、卷期号等期刊噪声。
    """
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
    """判定标题是否噪声（期刊卷期、过短、过多元字符等）。"""
    normalized = helper_normalize_digits(title or "").strip()
    if len(normalized) < 4 or BAD_TITLE_CHARS_RE.search(normalized):
        return True
    # 期刊卷期模式：中华医学杂志 第 30 卷 第 5 期、Chin J ... Vol 30 No 5 等
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
    """清洗标题：NFKC 规范化 + 去坏字符 + 去注释 + 去首尾非词字符 + 截断 180。"""
    title = helper_normalize_digits(title or "")
    title = BAD_TITLE_CHARS_RE.sub("", title)
    title = re.sub(r"<!--.*?-->", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^[\W_]+|[\W_]+$", "", title, flags=re.UNICODE).strip()
    return title[:180].strip()


def extract_title(markdown: str, pdf_path: str | Path) -> str:
    """从 Markdown 与 PDF 文件名中提取最可能的标题。

    候选来源：
    1. 第一个 # 标题；
    2. 前 12 个非注释非空行；
    3. PDF 文件名派生（clean_title_from_filename）。
    取 helper_title_score 最高的候选；都不可用时回退到文件名版本。
    """
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
    """提取摘要：跳过标题行与注释行，拼接正文直到总字符数达 max_chars。"""
    lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        lines.append(stripped)
        if sum(len(item) for item in lines) >= max_chars:
            break
    return " ".join(lines)[:max_chars]
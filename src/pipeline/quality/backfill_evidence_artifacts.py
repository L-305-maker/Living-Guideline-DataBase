"""回填 evidence 产物的弱字段：title / publication_date / abstract / clinical_department / card fields。

主要逻辑（按文档级 vs 分块级分类）：
1. 文档级（backfill_evidence_artifacts 主循环）：
   - title：generic / 噪声 → helper_find_better_title 重新抽取；
   - source_institution：Unknown → helper_infer_source 从文件名/正文推断；
   - publication_date：缺失或越界 → extract_publication_date 重解析；
   - abstract：过短 → helper_find_best_abstract（锚定 / 兜底 / 标题三段）；
   - clinical_department：未分类 → classify_clinical_department 重跑；
   - cleaning_quality：helper_apply_audit_quality 按正文信号重新评估。
2. 卡片字段（helper_backfill_card_fields）：从全文信号补 key_recommendations / scope / target_population / clinical_questions_pico。
3. 派生同步（sync_derived=True 时）：写完 documents/cards/views 后再同步 sections 与 chunks。

dry_run=True 时计算完整变更但不写文件，便于预览。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.models.schemas import DocumentRecord, dump_model
from src.pipeline.cleaning.cleaner import assess_cleaned_body
from src.pipeline.cleaning.semantic_chunker import retrieval_text
from src.pipeline.quality.repair_quality import (
    helper_reference_like,
    helper_sync_sections_from_documents,
)
from src.retrieval.document_repr.builder import (
    POPULATION_LINE_RE,
    SCOPE_LINE_RE,
    build_document_card,
    build_document_views,
    helper_candidate_lines,
    helper_clip,
    helper_dedupe_lines,
    helper_is_useful_line,
    helper_normalize_space,
)
from src.utils.clinical_department import UNKNOWN_DEPARTMENT, classify_clinical_department
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.ids import sha256_text
from src.utils.io import DATA_DIR, ensure_dir, iter_markdown_files, read_jsonl, write_jsonl
from src.utils.metadata import (
    clean_title,
    clean_title_from_filename,
    extract_abstract,
    extract_publication_date,
    extract_source_institution,
    is_suspicious_title,
)


VALID_DATE_RE = re.compile(r"^(20[1-2]\d)-\d{2}-\d{2}$")  # 合法 publication_date 格式：YYYY-MM-DD 且年份 2010-2029。
CJK_RE = re.compile(r"[\u4e00-\u9fff]")  # 中文字符判定，用于 expected_chinese 与文本编码失败检测。
REFERENCE_START_RE = re.compile(r"(?im)^\s*#{0,6}\s*(references|bibliography|\u53c2\u8003\u6587\u732e)\b")  # 参考文献标题（references/bibliography/参考文献）起始，便于截取正文信号。
TITLE_KEYWORD_RE = re.compile(  # 标题关键词（中英文），评分时加权：guideline/recommendation/指南/推荐 等。
    r"(guideline|guidelines|recommendation|recommendations|consensus|statement|practice parameter|"
    r"\u6307\u5357|\u5171\u8bc6|\u63a8\u8350|\u89c4\u8303|\u8bca\u7597|\u8bca\u6cbb|\u4e13\u5bb6\u5efa\u8bae)",
    re.I,
)
STRONG_TITLE_KEYWORD_RE = re.compile(  # 强标题关键词：guideline/consensus/指南/共识/规范 等，更高分加成。
    r"(guideline|guidelines|consensus|statement|practice parameters?|"
    r"\u6307\u5357|\u5171\u8bc6|\u89c4\u8303|\u8bca\u7597|\u8bca\u6cbb|\u4e13\u5bb6\u5efa\u8bae)",
    re.I,
)
GENERIC_TITLE_RE = re.compile(  # 通用噪声标题（untitled/contents/纯 hash 等），命中即扣分。
    r"^(?:"
    r"untitled|contents?|special articles?|guideline update|recommendations?|summary|executive summary|"
    r"practice guidelines?|clinical practice guidelines?|an official website.*|official websites use.*|"
    r"[a-f0-9]{8,20}|[a-z]?[a-f0-9]{8,20}"
    r")$",
    re.I,
)
SHORT_PUBLISHER_SUBTITLE_RE = re.compile(  # 短发布机构副标题（如 American Academy ... Guideline），易被误识别为标题。
    r"^(?:an?\s+)?(?:american academy|american college|american association|european society|"
    r"national institute|world health organization).{0,90}(?:guideline|statement|report|consensus)$",
    re.I,
)
TITLE_NOISE_RE = re.compile(  # 标题里常见的 DOI / 卷期号 / URL 噪声模式，命中即降分。
    r"(?:doi\s*:|keywords?\s*:|copyright|accepted for publication|correspondence|"
    r"\bvol\.?\s*\d+|\bno\.?\s*\d+|https?://|www\.)",
    re.I,
)
BODY_SENTENCE_PREFIX_RE = re.compile(  # 正文段起始词（introduction/methods/recommendations 等），用于阻断标题块搜索。
    r"^(?:introduction|methods?|recommendations?|purpose|findings|conclusions?|background|summary|abstract)\s*[:\uff1a]|"
    r"^(?:this document|the present document)\b",
    re.I,
)
ABSTRACT_PROSE_START_RE = re.compile(  # 摘要/正文起首模式，标题块搜索遇此中断。
    r"^(?:a systematic|the purpose|the present document|this guideline|this clinical practice guideline|"
    r"this document|available data|methods?\s*:|introduction\s*:|recommendations?\s*:)",
    re.I,
)
AUTHOR_OR_AFFILIATION_RE = re.compile(  # 作者/单位特征（MD/PhD/university 等），用于判定标题区结束。
    r"(?:\b(?:MD|PhD|MS|MSc|MA|MPH|FRCP|RN|DO)\b|"
    r"\b(?:university|hospital|department|center|centre|school of medicine)\b|;)",
    re.I,
)
ABSTRACT_START_RE = re.compile(  # 摘要章节起始关键词（中英文），helper_find_anchored_abstract 用。
    r"^\s*(?:#{1,6}\s*)?(abstract|summary|executive summary|overview|background|objective|purpose|"
    r"recommendations?|conclusions?|\u6458\u8981|\u63d0\u8981)\s*[:\uff1a]?",
    re.I,
)
ABSTRACT_STOP_RE = re.compile(  # 摘要章节结束关键词（keywords/introduction 等），遇到停止收集。
    r"^\s*(?:#{1,6}\s*)?(keywords?|citation|contents?|introduction|references|bibliography|"
    r"\u5173\u952e\u8bcd|\u76ee\u5f55|\u5f15\u8a00|\u53c2\u8003\u6587\u732e)\b",
    re.I,
)
LOW_VALUE_ABSTRACT_RE = re.compile(r"^\s*(?:table|figure|\||doi\s*:|keywords?\s*:|https?://|www\.)", re.I)  # 低价值摘要行（表格/figure/DOI 等），跳过。
MOJIBAKE_MARKER_RE = re.compile(r"[锛绗鍗鏈鏉傚織圽穦碶€]")  # mojibake 标记字符集合，用于检测中文文档文本编码失败。
STRICT_QUESTION_LINE_RE = re.compile(  # 严格 PICO 问题行匹配（clinical questions / 临床问题 等）。
    r"(\bPICO\b|\b(?:clinical|key|research)\s+questions?\b|^\s*(?:Q|Question)\s*\d+[:.)-]|"
    r"\u4e34\u5e8a\u95ee\u9898|\u5173\u952e\u95ee\u9898)",
    re.I,
)
STRICT_RECOMMENDATION_LINE_RE = re.compile(  # 严格推荐行匹配（recommend / recommend not / 推荐 / 应当 等）。
    r"(recommendation\s*(?:statement|statements|\d+)?|we recommend|we suggest|strong recommendation|"
    r"conditional recommendation|is recommended|are recommended|not recommended|should(?:\s+not)?|"
    r"\u63a8\u8350|\u5efa\u8bae|\u4e0d\u63a8\u8350|\u4e0d\u5b9c|\u5e94\u8be5|\u5e94\u5f53|\u5e94\u4e88|\u5e94\u8003\u8651|\u4e0d\u5e94)",
    re.I,
)
CITATION_LIKE_RE = re.compile(  # 引用/期刊引用模式，避免把文献列表误判为推荐。
    r"(\[[Jj]\]|\b(?:doi|pmid)\b|(?:19|20)\d{2}\s*[,;]\s*\d+|"
    r"(?:19|20)\d{2}[^.\n]{0,80}:\s*\d+|"
    r"\b(?:Surg|Med|Clin|Journal|Neurosurgery|Pancreatology|Transpl|Hepatol|Lancet|JAMA|BMJ)\b.{0,80}(?:19|20)\d{2})",
    re.I,
)


def helper_split_flags(value: str | None) -> list[str]:
    """把 cleaning_flags 字符串按 ',' 拆分为 list，自动去空白。"""
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def helper_valid_publication_date(value: str | None) -> bool:
    """判定 publication_date 是否为合法 YYYY-MM-DD 格式。"""
    if not value or value == "unknown":
        return False
    return bool(VALID_DATE_RE.match(value))


def helper_metadata_bool(metadata: dict[str, str], key: str) -> bool:
    """读 front-matter 布尔字段（1/true/yes/y 视为 True）。"""
    return str(metadata.get(key, "")).strip().lower() in {"1", "true", "yes", "y"}


def helper_weak_text(value: str | None, min_chars: int = 80) -> bool:
    """判定文本是否过短（去空白后 < min_chars，默认 80）。"""
    return len(helper_normalize_space(value or "")) < min_chars


def helper_is_generic_title(title: str) -> bool:
    """通用标题判定：空白 / generic 模式 / 字母比 < 35% / is_suspicious_title。"""
    normalized = clean_title(title)
    if not normalized:
        return True
    lower = normalized.lower()
    if GENERIC_TITLE_RE.match(lower):
        return True
    if is_suspicious_title(normalized):
        return True
    visible = [char for char in normalized if not char.isspace()]
    if len(visible) < 5:
        return True
    letters = sum(1 for char in visible if char.isalpha() or "\u4e00" <= char <= "\u9fff")
    return letters / max(1, len(visible)) < 0.35


def helper_clean_title_candidate(line: str) -> str:
    """清洗候选标题：去 Markdown 标题符 / 「Title:」前缀 / 截断 180 字符。"""
    line = re.sub(r"^\s*#{1,6}\s*", "", line or "").strip()
    line = re.sub(r"^\s*(?:title|题名)\s*[:\uff1a]\s*", "", line, flags=re.I)
    line = clean_title(line)
    if len(line) >= 180:
        line = line[:180].rsplit(" ", 1)[0].strip()
    return line


def helper_first_alpha_is_lower(text: str) -> bool:
    """判定首个字母是否小写（句子开头时不应作为标题候选）。"""
    for char in text:
        if char.isalpha():
            return char.isascii() and char.islower()
    return False


def helper_join_title_lines(lines: list[str]) -> str:
    """把多行标题片段按行拼接，处理连字符断行。"""
    output = ""
    for line in lines:
        stripped = helper_clean_title_candidate(line)
        if not stripped:
            continue
        if output.endswith("-"):
            output = output[:-1] + stripped
        elif output:
            output += " " + stripped
        else:
            output = stripped
    return clean_title(output)


def helper_candidate_title_blocks(body: str) -> list[tuple[str, int]]:
    """从文首 ≤600 行扫描候选标题块，按 STRONG_TITLE_KEYWORD 早停。"""
    # 候选标题只从文首有限区域提取，并过滤页眉、目录和正文句子等高风险噪声。
    raw_lines = body.splitlines()[:600]
    candidates: list[tuple[str, int]] = []
    seen_early_title = False
    for index, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("|") or stripped.startswith("!"):
            continue
        if seen_early_title and (AUTHOR_OR_AFFILIATION_RE.search(stripped) or ABSTRACT_PROSE_START_RE.search(stripped)):
            break
        single = helper_clean_title_candidate(stripped)
        if single:
            candidates.append((single, index))
            if index < 40 and STRONG_TITLE_KEYWORD_RE.search(single):
                seen_early_title = True
        if BODY_SENTENCE_PREFIX_RE.search(stripped) or AUTHOR_OR_AFFILIATION_RE.search(stripped):
            continue
        previous = raw_lines[index - 1].strip() if index > 0 else ""
        previous_title = helper_clean_title_candidate(previous) if previous else ""
        starts_title_block = index == 0 or not previous or helper_is_generic_title(previous_title)
        if not starts_title_block:
            continue
        block: list[str] = []
        for offset in range(0, 8):
            if index + offset >= len(raw_lines):
                break
            candidate_line = raw_lines[index + offset].strip()
            if not candidate_line:
                break
            cleaned_line = helper_clean_title_candidate(candidate_line)
            generic_line = helper_is_generic_title(cleaned_line)
            generic_suffix = offset > 0 and bool(STRONG_TITLE_KEYWORD_RE.search(cleaned_line))
            if not cleaned_line or (generic_line and not generic_suffix) or TITLE_NOISE_RE.search(cleaned_line):
                if offset == 0:
                    block = []
                break
            if offset > 0 and (BODY_SENTENCE_PREFIX_RE.search(candidate_line) or AUTHOR_OR_AFFILIATION_RE.search(candidate_line)):
                break
            if offset > 0 and len(cleaned_line) > 140 and not TITLE_KEYWORD_RE.search(cleaned_line):
                break
            block.append(candidate_line)
        if len(block) >= 2:
            joined = helper_join_title_lines(block)
            if joined:
                candidates.append((joined, index))
                if index < 40 and STRONG_TITLE_KEYWORD_RE.search(joined):
                    seen_early_title = True
    return candidates


def helper_title_candidate_score(title: str, index: int) -> float:
    """标题评分：位置 + 长度 + 关键词命中 + CJK 数 + 强关键词位置加权。"""
    # 标题评分组合位置、长度和结构信号，任何单一弱特征都不能独立决定结果。
    if not title or helper_is_generic_title(title) or TITLE_NOISE_RE.search(title):
        return -1000.0
    if SHORT_PUBLISHER_SUBTITLE_RE.search(title) and len(title) < 110:
        return -1000.0
    if BODY_SENTENCE_PREFIX_RE.search(title) or helper_first_alpha_is_lower(title):
        return -1000.0
    visible = [char for char in title if not char.isspace()]
    if not visible:
        return -1000.0
    comma_like = len(re.findall(r"[,;\uff0c\uff1b]", title))
    if comma_like >= 8 and not TITLE_KEYWORD_RE.search(title):
        return -1000.0
    score = 0.0
    strong_match = STRONG_TITLE_KEYWORD_RE.search(title)
    score += 90.0 if strong_match else 0.0
    score += 25.0 if TITLE_KEYWORD_RE.search(title) and not strong_match else 0.0
    score += 35.0 if len(title) >= 90 and strong_match else 0.0
    if strong_match and strong_match.start() <= 60:
        score += 35.0
    elif strong_match and strong_match.start() > 100:
        score -= 45.0
    score += 18.0 if 10 <= len(title) <= 170 else -10.0
    score += min(len(title), 120) / 12.0
    score += min(len(CJK_RE.findall(title)), 20) * 1.2
    score += max(0.0, 34.0 - index * 0.8)
    if len(title) > 180:
        score -= 20.0
    return score


def helper_find_better_title(body: str, source_file: str) -> str:
    """综合候选标题与文件名标题，选最高分（>=55）的结果。"""
    candidates = helper_candidate_title_blocks(body)
    filename_title = clean_title_from_filename(source_file)
    if filename_title:
        candidates.append((filename_title, 250))
    if not candidates:
        return ""
    best_title, _index = max(candidates, key=lambda item: helper_title_candidate_score(item[0], item[1]))
    return best_title if helper_title_candidate_score(best_title, _index) >= 55 else ""


def helper_readable_ratio(text: str) -> float:
    """可读字符占比（去空白后），用于摘要与正文质量判定。"""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    readable = sum(1 for char in compact if char.isalnum() or "\u4e00" <= char <= "\u9fff" or char in "，。；：、“”‘’（）《》.,;:!?()/%-")
    return readable / len(compact)


def helper_usable_abstract(text: str, max_chars: int = 1200) -> str:
    """规范化摘要：截断 1200 + 最短长度 40 + 可读比 ≥0.55。"""
    abstract = helper_normalize_space(text or "")
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rstrip()
    if len(abstract) < 40:
        return ""
    if helper_readable_ratio(abstract) < 0.55:
        return ""
    return abstract


def helper_find_anchored_abstract(body: str, max_chars: int = 1200) -> str:
    """从摘要关键词起首的位置收集摘要，遇到 keywords/introduction 停止。"""
    lines: list[str] = []
    collecting = False
    for raw_line in body.splitlines():
        stripped = raw_line.strip(" #*\t")
        if not stripped:
            if collecting and lines:
                break
            continue
        if ABSTRACT_STOP_RE.search(stripped) and lines:
            break
        if ABSTRACT_START_RE.search(stripped):
            collecting = True
            stripped = ABSTRACT_START_RE.sub("", stripped).strip(" :\uff1a")
        if not collecting:
            continue
        if LOW_VALUE_ABSTRACT_RE.search(stripped):
            continue
        if helper_is_useful_line(stripped, min_chars=20):
            lines.append(stripped)
        if sum(len(item) for item in lines) >= max_chars:
            break
    return helper_usable_abstract(" ".join(lines), max_chars)


def helper_find_fallback_abstract(body: str, title: str, max_chars: int = 1200) -> str:
    """从正文首部拿摘要（避开标题重复行），作为锚定方法的兜底。"""
    title_norm = helper_normalize_space(title).lower()
    lines: list[str] = []
    for raw_line in extract_abstract(body, max_chars=max_chars * 3).splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|") or LOW_VALUE_ABSTRACT_RE.search(stripped):
            continue
        if helper_normalize_space(stripped).lower() == title_norm:
            continue
        if helper_is_useful_line(stripped, min_chars=25):
            lines.append(stripped)
        if sum(len(item) for item in lines) >= max_chars:
            break
    return helper_usable_abstract(" ".join(lines), max_chars)


def helper_find_best_abstract(body: str, title: str) -> str:
    """三段式取最佳摘要：锚定 → fallback → 标题兜底。"""
    return helper_find_anchored_abstract(body) or helper_find_fallback_abstract(body, title) or helper_usable_abstract(title, 400)


def helper_infer_source(source_file: str, body: str) -> str:
    """从文件名与正文推断来源机构（Unknown 时回退）。"""
    return extract_source_institution(source_file, body)


def helper_looks_like_text_encoding_failure(metadata: dict[str, str], body: str) -> bool:
    """中文文档正文却几乎无 CJK 字符时判为文本编码失败（OCR/PDF 抽取出错）。"""
    source_file = (metadata.get("source_file") or "").lower()
    title = metadata.get("title") or ""
    source = metadata.get("source_institution") or ""
    expected_chinese = source == "CMA" or "cma_pdf" in source_file or bool(CJK_RE.search(title))
    if not expected_chinese:
        return False
    compact = re.sub(r"\s+", "", body or "")
    if len(compact) < 600:
        return False
    cjk = len(CJK_RE.findall(compact))
    marker_count = len(MOJIBAKE_MARKER_RE.findall(compact[:12000]))
    symbol_count = sum(1 for char in compact if not char.isalnum() and not ("\u4e00" <= char <= "\u9fff"))
    cjk_ratio = cjk / max(1, len(compact))
    marker_ratio = marker_count / max(1, cjk)
    symbol_ratio = symbol_count / max(1, len(compact))
    return marker_count >= 80 or marker_ratio >= 0.08 or (cjk_ratio < 0.12 and symbol_ratio > 0.22)


def helper_apply_audit_quality(metadata: dict[str, str], body: str) -> bool:
    """重新评估 cleaning_quality 与 flags，合并编码失败标记。"""
    original_quality = metadata.get("cleaning_quality", "")
    original_flags = metadata.get("cleaning_flags", "")
    report = assess_cleaned_body(body)
    flags = helper_dedupe_lines([*helper_split_flags(original_flags), *report["flags"]], 40)
    if helper_looks_like_text_encoding_failure(metadata, body):
        flags = helper_dedupe_lines([*flags, "likely_text_encoding_failure"], 40)
    severe = {"low_text_signal", "likely_ocr_failure", "noisy_ocr_lines", "noisy_ocr_title", "likely_text_encoding_failure"}
    if severe & set(flags):
        quality = "poor"
    elif flags:
        quality = "warning"
    else:
        quality = "ok"
    metadata["cleaning_quality"] = quality
    metadata["cleaning_flags"] = ",".join(flags)
    return metadata.get("cleaning_quality", "") != original_quality or metadata.get("cleaning_flags", "") != original_flags


def helper_content_before_references(body: str) -> str:
    """截取参考文献之前的正文，避免文献列表污染全文信号。"""
    match = REFERENCE_START_RE.search(body or "")
    return body[: match.start()] if match else body


def helper_collect_full_text_signals(body: str) -> dict[str, list[str]]:
    """收集 recommendation / question / scope / population 四类信号，超额截断。"""
    # 全文信号用于补充文档级元数据，但不得覆盖来源文件和显式 front matter。
    signals: dict[str, list[str]] = {"recommendation": [], "question": [], "scope": [], "population": []}
    limits = {"recommendation": 28, "question": 18, "scope": 16, "population": 14}
    text = helper_content_before_references(body)
    for line in helper_candidate_lines(text):
        if not helper_is_useful_line(line):
            continue
        if CITATION_LIKE_RE.search(line) and not re.search(r"(\u63a8\u8350|\u5efa\u8bae|\u5e94\u8be5|\u5e94\u5f53|\u5e94\u4e88)", line):
            continue
        if len(signals["recommendation"]) < limits["recommendation"] and STRICT_RECOMMENDATION_LINE_RE.search(line):
            signals["recommendation"].append(helper_clip(line, 800))
        if len(signals["question"]) < limits["question"] and STRICT_QUESTION_LINE_RE.search(line):
            signals["question"].append(helper_clip(line, 700))
        if len(signals["scope"]) < limits["scope"] and SCOPE_LINE_RE.search(line):
            signals["scope"].append(helper_clip(line, 700))
        if len(signals["population"]) < limits["population"] and POPULATION_LINE_RE.search(line):
            signals["population"].append(helper_clip(line, 650))
        if all(len(signals[name]) >= limits[name] for name in signals):
            break
    return {name: helper_dedupe_lines(values, limits[name]) for name, values in signals.items()}


def helper_rebuild_card_text(card: dict[str, Any]) -> None:
    fields = card.get("fields") or {}
    card_sections = [
        ("Title", card.get("title", "")),
        ("Publisher", card.get("source_institution", "")),
        ("Document Type", "guideline / consensus"),
        ("Abstract", fields.get("title_abstract", "")),
        ("Scope", fields.get("scope", "")),
        ("Target Population", fields.get("target_population", "")),
        ("Key Recommendations", fields.get("key_recommendations", "")),
        ("Clinical Questions / PICO", fields.get("clinical_questions_pico", "")),
        ("Evidence Review", fields.get("evidence_review", "")),
        ("Heading Tree", fields.get("heading_tree", "")),
        ("Important Tables", fields.get("important_tables", "")),
        ("Conclusion", fields.get("conclusion", "")),
    ]
    card_text = "\n\n".join(f"[{name}]\n{value}" for name, value in card_sections if value)
    card["card_text"] = helper_clip(card_text, 12000)


def helper_backfill_card_fields(card: dict[str, Any], body: str) -> set[str]:
    """对弱 card 字段从全文信号补齐；保持已有高可信字段不变。"""
    # 只回填缺失或确定错误的卡片字段，并保持已有高可信信息不变。
    flags = set(helper_split_flags(card.get("cleaning_flags", "")))
    if card.get("cleaning_quality") == "poor" or "likely_text_encoding_failure" in flags:
        return set()
    fields = card.setdefault("fields", {})
    changed: set[str] = set()
    needs_recommendation = helper_weak_text(fields.get("key_recommendations"), 80)
    needs_scope = helper_weak_text("\n".join([fields.get("scope", ""), fields.get("target_population", "")]), 80)
    needs_pico = helper_weak_text(fields.get("clinical_questions_pico"), 60)
    if not (needs_recommendation or needs_scope or needs_pico):
        return changed
    signals = helper_collect_full_text_signals(body)
    if needs_recommendation and signals["recommendation"]:
        fields["key_recommendations"] = "\n".join(signals["recommendation"])
        changed.add("recommendation_summary")
    if needs_scope:
        if signals["scope"]:
            fields["scope"] = helper_clip("\n".join(signals["scope"]), 2200)
        if signals["population"]:
            fields["target_population"] = helper_clip("\n".join(signals["population"]), 1400)
        if signals["scope"] or signals["population"]:
            changed.add("scope_population")
    if needs_pico and signals["question"]:
        fields["clinical_questions_pico"] = "\n".join(signals["question"])
        changed.add("pico_questions")
    if changed:
        helper_rebuild_card_text(card)
    return changed


def helper_card_view_text(card: dict[str, Any], view_type: str) -> str:
    """按 view_type 从 card.fields 取对应文本。"""
    fields = card.get("fields") or {}
    if view_type == "recommendation_summary":
        return fields.get("key_recommendations", "")
    if view_type == "scope_population":
        return "\n".join([fields.get("scope", ""), fields.get("target_population", "")]).strip()
    if view_type == "pico_questions":
        return fields.get("clinical_questions_pico", "")
    return ""


def helper_document_record_from_metadata(
    metadata: dict[str, str],
    abstract: str,
    markdown: str,
    clean_path: Path,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    existing = existing or {}
    doc_id = metadata.get("id") or clean_path.stem
    raw_path = existing.get("markdown_raw_path") or str(DATA_DIR / "markdown_raw" / f"{doc_id}.md")
    record = DocumentRecord(
        doc_id=doc_id,
        title=metadata.get("title", "") or clean_path.stem,
        publication_date=metadata.get("publication_date") or "unknown",
        source_institution=metadata.get("source_institution") or "Unknown",
        clinical_department=metadata.get("clinical_department") or UNKNOWN_DEPARTMENT,
        source_file=metadata.get("source_file", ""),
        markdown_raw_path=raw_path,
        markdown_clean_path=str(clean_path),
        abstract=abstract,
        content_sha256=sha256_text(markdown),
        cleaning_quality=metadata.get("cleaning_quality", ""),
        cleaning_flags=metadata.get("cleaning_flags", ""),
        source_pdf_text_quality=metadata.get("source_pdf_text_quality", ""),
        source_pdf_needs_ocr=helper_metadata_bool(metadata, "source_pdf_needs_ocr"),
        source_pdf_is_scanned=helper_metadata_bool(metadata, "source_pdf_is_scanned"),
        pdf_text_quality=metadata.get("pdf_text_quality", ""),
        pdf_needs_ocr=helper_metadata_bool(metadata, "pdf_needs_ocr"),
        pdf_is_scanned=helper_metadata_bool(metadata, "pdf_is_scanned"),
        ocr_engine=metadata.get("ocr_engine", ""),
        ocr_applied=helper_metadata_bool(metadata, "ocr_applied"),
        ocr_status=metadata.get("ocr_status", ""),
        ocr_error=metadata.get("ocr_error", ""),
    )
    return dump_model(record)


def helper_note_example(examples: dict[str, list[dict[str, str]]], key: str, doc_id: str, before: str, after: str) -> None:
    if len(examples[key]) >= 8:
        return
    examples[key].append({"doc_id": doc_id, "before": before, "after": after})


def helper_sync_sections(data_dir: Path, documents_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    return helper_sync_sections_from_documents(data_dir, documents_by_id, clear_missing_fields=True)


def helper_iter_jsonl_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from read_jsonl(path)


def helper_sync_chunks(data_dir: Path, documents_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    changed_files = 0
    changed_rows = 0
    chunk_dir = data_dir / "chunks"
    chunk_files = sorted(path for path in chunk_dir.glob("*.jsonl") if not path.name.startswith("all_chunks"))
    for path in chunk_files:
        rows = list(read_jsonl(path))
        output: list[dict[str, Any]] = []
        file_changed = False
        for row in rows:
            new_row = dict(row)
            doc = documents_by_id.get(new_row.get("doc_id", ""))
            if not doc:
                file_changed = True
                continue
            for key in ("title", "publication_date", "source_institution", "clinical_department"):
                if new_row.get(key) != doc.get(key):
                    new_row[key] = doc.get(key, "")
                    file_changed = True
            # 分块的参考文献标记继承自完整章节；仅用孤立分块重新判断会漏掉大量参考文献内容。
            section_path = new_row.get("section_path") or []
            expected_retrieval = retrieval_text(
                new_row.get("title", ""),
                section_path if isinstance(section_path, list) else [],
                new_row.get("chunk_type", "other"),
                new_row.get("content", ""),
            )
            if new_row.get("retrieval_text") != expected_retrieval:
                new_row["retrieval_text"] = expected_retrieval
                file_changed = True
            if new_row != row:
                changed_rows += 1
            output.append(new_row)
        if file_changed:
            write_jsonl(path, output)
            changed_files += 1
    write_jsonl(chunk_dir / "all_chunks.jsonl", helper_iter_jsonl_rows(chunk_files))
    return {"chunk_files_changed": changed_files, "chunk_rows_changed": changed_rows, "chunks_total": sum(1 for _ in read_jsonl(chunk_dir / "all_chunks.jsonl"))}


def helper_sync_derived_metadata(data_dir: Path, documents_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """sections 与 chunks 同步的合并入口。"""
    stats: dict[str, Any] = {}
    stats.update(helper_sync_sections(data_dir, documents_by_id))
    stats.update(helper_sync_chunks(data_dir, documents_by_id))
    return stats


def backfill_evidence_artifacts(
    data_dir: str | Path = DATA_DIR,
    dry_run: bool = False,
    sync_derived: bool = True,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    clean_dir = data_path / "markdown_clean"
    documents_path = data_path / "documents.jsonl"
    # 旧 documents 仅作为缺失字段回填来源；最终输出仍以 markdown_clean 全量重建为准。
    existing_documents = {record["doc_id"]: record for record in read_jsonl(documents_path)} if documents_path.exists() else {}

    counters: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    documents: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []

    for clean_path in iter_markdown_files(clean_dir):
        original_markdown = clean_path.read_text(encoding="utf-8", errors="replace")
        metadata, body = parse_front_matter(original_markdown)
        if not metadata:
            counters["markdown_missing_front_matter"] += 1
            continue
        doc_id = metadata.get("id") or clean_path.stem
        existing = existing_documents.get(doc_id, {})
        metadata.setdefault("id", doc_id)
        metadata.setdefault("source_file", existing.get("source_file", ""))

        before_title = metadata.get("title", "")
        current_title = clean_title(before_title)
        if helper_is_generic_title(current_title):
            candidate_title = helper_find_better_title(body, metadata.get("source_file", ""))
            if candidate_title and candidate_title != current_title:
                metadata["title"] = candidate_title
                counters["title_backfilled"] += 1
                helper_note_example(examples, "title_backfilled", doc_id, before_title, candidate_title)
        else:
            metadata["title"] = current_title

        before_source = metadata.get("source_institution", "") or "Unknown"
        if before_source == "Unknown":
            inferred_source = helper_infer_source(metadata.get("source_file", ""), body)
            if inferred_source and inferred_source != "Unknown":
                metadata["source_institution"] = inferred_source
                counters["source_backfilled"] += 1
                helper_note_example(examples, "source_backfilled", doc_id, before_source, inferred_source)

        before_date = metadata.get("publication_date", "") or "unknown"
        if not helper_valid_publication_date(before_date):
            inferred_date = extract_publication_date(metadata.get("source_file", ""), body)
            if helper_valid_publication_date(inferred_date):
                metadata["publication_date"] = inferred_date
                counters["publication_date_backfilled"] += 1
                helper_note_example(examples, "publication_date_backfilled", doc_id, before_date, inferred_date)

        abstract = helper_find_best_abstract(body, metadata.get("title", ""))
        existing_abstract = existing.get("abstract", "")
        if helper_weak_text(existing_abstract, 80) and abstract and abstract != existing_abstract:
            counters["abstract_backfilled"] += 1
            helper_note_example(examples, "abstract_backfilled", doc_id, existing_abstract, abstract)

        before_department = metadata.get("clinical_department", "") or UNKNOWN_DEPARTMENT
        if before_department in {"", UNKNOWN_DEPARTMENT, "Unknown", "未分类"}:
            department = classify_clinical_department(metadata.get("title", ""), abstract, body)
            if department and department != before_department:
                metadata["clinical_department"] = department
                counters["clinical_department_backfilled"] += 1
                helper_note_example(examples, "clinical_department_backfilled", doc_id, before_department, department)

        if helper_apply_audit_quality(metadata, body):
            counters["quality_reassessed"] += 1

        updated_markdown = dump_front_matter(metadata, body)
        # dry_run 仍计算完整变更和派生对象，但不触碰 Markdown 与 JSONL 文件。
        if updated_markdown != original_markdown:
            counters["markdown_front_matter_changed"] += 1
            if not dry_run:
                clean_path.write_text(updated_markdown, encoding="utf-8", newline="\n")

        record = helper_document_record_from_metadata(metadata, abstract, updated_markdown, clean_path, existing)
        documents.append(record)

        card = build_document_card(updated_markdown, clean_path)
        changed_views = helper_backfill_card_fields(card, body)
        for view_type in changed_views:
            counters[f"{view_type}_backfilled"] += 1
            helper_note_example(examples, f"{view_type}_backfilled", doc_id, "", helper_card_view_text(card, view_type)[:500])
        cards.append(card)
        views.extend(build_document_views(card))

    if not dry_run:
        # 三类主产物来自同一轮内存结果，保证文档、卡片和视图使用一致的元数据快照。
        write_jsonl(documents_path, documents)
        write_jsonl(data_path / "document_cards.jsonl", cards)
        write_jsonl(data_path / "document_views.jsonl", views)

    source_counts = Counter(record.get("source_institution", "Unknown") for record in documents)
    quality_counts = Counter(record.get("cleaning_quality", "") for record in documents)
    view_counts = Counter(view.get("view_type", "") for view in views)
    missing_required_views = {
        "scope_population": len(documents) - view_counts.get("scope_population", 0),
        "recommendation_summary": len(documents) - view_counts.get("recommendation_summary", 0),
        "pico_questions": len(documents) - view_counts.get("pico_questions", 0),
    }
    derived_stats: dict[str, Any] = {}
    if sync_derived and not dry_run:
        # 主产物落盘后再同步 section/chunk，避免派生记录引用尚未提交的文档元数据。
        derived_stats = helper_sync_derived_metadata(data_path, {record["doc_id"]: record for record in documents})

    report = {
        "dry_run": dry_run,
        "documents": len(documents),
        "cards": len(cards),
        "views": len(views),
        "changes": dict(counters),
        "source_counts": dict(source_counts.most_common()),
        "quality_counts": dict(quality_counts.most_common()),
        "view_counts": dict(view_counts.most_common()),
        "missing_required_views": missing_required_views,
        "examples": examples,
        "derived_sync": derived_stats,
    }
    if not dry_run:
        report_dir = ensure_dir(data_path / "reports")
        report_path = report_dir / "backfill_evidence_artifacts_report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync-derived", action="store_true")
    args = parser.parse_args()
    report = backfill_evidence_artifacts(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        sync_derived=not args.no_sync_derived,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

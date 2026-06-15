from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional


DEFAULT_INPUT = "outputs/data.jsonl"
DEFAULT_OUTPUT = "outputs/recommendations.jsonl"
DEFAULT_MIN_SCORE = 0.75

MAX_RECOMMENDATION_CHARS = 1200
MIN_RECOMMENDATION_CHARS = 25


STRONG_RECOMMENDATION_RE = re.compile(
    r"\b("
    r"we\s+recommend|is\s+recommended|are\s+recommended|recommend(?:ed|s|ing)?|"
    r"should|should\s+not|must|must\s+not|do\s+not|do\s+not\s+use|"
    r"offer|provide|administer|initiate|start|use|treat|screen|monitor|refer"
    r")\b|推荐|建议|应当|应|不应|必须|禁用|给予|使用|筛查|监测|转诊",
    re.IGNORECASE,
)

WEAK_RECOMMENDATION_RE = re.compile(
    r"\b("
    r"suggest(?:ed|s|ing)?|may|might|can|could|consider|reasonable|"
    r"may\s+be\s+considered|it\s+is\s+reasonable"
    r")\b|可考虑|可以|酌情|弱推荐|有条件推荐",
    re.IGNORECASE,
)

CLINICAL_SIGNAL_RE = re.compile(
    r"\b("
    r"patient|patients|adult|adults|child|children|infant|neonate|pregnan|"
    r"disease|syndrome|infection|cancer|diabetes|hypertension|stroke|asthma|"
    r"diagnos|treat|therapy|management|screening|prevention|dose|dosage|mg|ml|kg|"
    r"risk|symptom|clinical|surgery|vaccine|drug|antibiotic"
    r")\b|患者|成人|儿童|婴儿|孕妇|疾病|感染|癌|糖尿病|高血压|诊断|治疗|管理|筛查|预防|剂量|风险|症状|临床",
    re.IGNORECASE,
)

NOISE_RE = re.compile(
    r"\b("
    r"references?|bibliography|acknowledg|copyright|permission|appendix|"
    r"supplementary|table\s+of\s+contents|figure|doi|pmid|isbn"
    r")\b|参考文献|致谢|版权|附录|目录",
    re.IGNORECASE,
)

URL_RE = re.compile(r"https?://|www\.|doi\.org|\.gov/|\.org/|\.com/", re.IGNORECASE)

EVIDENCE_QUALITY_RE = re.compile(
    r"\b("
    r"high|moderate|low|very\s+low"
    r")\s+(?:quality|certainty)\s+(?:of\s+)?evidence\b|"
    r"\b(?:quality|certainty)\s+(?:of\s+)?evidence\s*[:：]?\s*(high|moderate|low|very\s+low)\b|"
    r"证据质量\s*[:：]?\s*(高|中|低|极低)",
    re.IGNORECASE,
)

RECOMMENDATION_STRENGTH_RE = re.compile(
    r"\b(strong|conditional|weak)\s+recommendation\b|"
    r"\brecommendation\s+strength\s*[:：]?\s*(strong|conditional|weak)\b|"
    r"(强推荐|弱推荐|有条件推荐)",
    re.IGNORECASE,
)


@dataclass
class Recommendation:
    id: str
    recommendation_text: str
    score: float
    label: str
    pico: Dict[str, str] = field(default_factory=dict)
    grade_quality: str = ""
    recommendation_strength: str = ""
    title: str = ""
    url: str = ""
    published_date: str = ""
    medical_topic: Any = None
    chunk_index: Optional[int] = None
    source: str = ""
    issuer: str = ""
    evidence_span: str = ""


def load_chunks(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    """流式读取 chunk jsonl，坏行跳过。"""
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过 JSON 解析失败行 {line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                yield item


def split_sentences(text: str) -> List[str]:
    """兼容英文和中文指南的句子切分。"""
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?。！？；;])\s+|(?<=[。！？；;])", text)
    sentences: List[str] = []
    for part in parts:
        part = part.strip(" \t\r\n-•*")
        if not part:
            continue
        # 对过长段落再按编号/项目符号弱切分。
        if len(part) > MAX_RECOMMENDATION_CHARS:
            subparts = re.split(r"\s+(?=(?:\d+\.|[A-Z]\.|[-•*])\s+)", part)
            sentences.extend(p.strip(" \t\r\n-•*") for p in subparts if p.strip())
        else:
            sentences.append(part)
    return sentences


def is_noise_sentence(sentence: str) -> bool:
    """过滤明显不是推荐意见的参考文献、URL、目录等内容。"""
    text = sentence.strip()
    if len(text) < MIN_RECOMMENDATION_CHARS:
        return True
    if URL_RE.search(text) and not CLINICAL_SIGNAL_RE.search(text):
        return True
    if NOISE_RE.search(text) and not STRONG_RECOMMENDATION_RE.search(text):
        return True
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.35 and not CLINICAL_SIGNAL_RE.search(text):
        return True
    return False


def score_recommendation(sentence: str) -> tuple[float, str]:
    """对候选句打分，返回分数和标签。"""
    if is_noise_sentence(sentence):
        return 0.0, "noise"

    score = 0.0
    label = "candidate"
    if STRONG_RECOMMENDATION_RE.search(sentence):
        score += 0.55
        label = "strong_rule"
    if WEAK_RECOMMENDATION_RE.search(sentence):
        score += 0.35
        if label == "candidate":
            label = "weak_rule"
    if CLINICAL_SIGNAL_RE.search(sentence):
        score += 0.25
    if EVIDENCE_QUALITY_RE.search(sentence) or RECOMMENDATION_STRENGTH_RE.search(sentence):
        score += 0.15
    if len(sentence) > MAX_RECOMMENDATION_CHARS:
        score -= 0.15
    if URL_RE.search(sentence):
        score -= 0.10

    return max(0.0, min(score, 1.0)), label


def _first_match(patterns: Iterable[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return " ".join(group for group in match.groups() if group).strip()
    return ""


def extract_pico(sentence: str) -> Dict[str, str]:
    """轻量 PICO 线索提取；无法稳定识别时保留空字符串。"""
    population = _first_match(
        [
            r"\b(?:in|for|among)\s+((?:adult|pediatric|pregnant|elderly|high-risk|hospitalized)?\s*patients?\s+with\s+[^,.;:]+)",
            r"\b((?:adults?|children|infants?|neonates|pregnant women)\s+with\s+[^,.;:]+)",
            r"对于([^，。；;]+?患者)",
        ],
        sentence,
    )
    intervention = _first_match(
        [
            r"\b(?:recommend|suggest|offer|provide|administer|initiate|start|use|treat\s+with)\s+([^,.;:]+)",
            r"\bshould\s+(?:receive|use|be\s+treated\s+with|be\s+offered|undergo)\s+([^,.;:]+)",
            r"(?:推荐|建议|应|给予|使用)([^，。；;]+)",
        ],
        sentence,
    )
    comparator = _first_match(
        [
            r"\b(?:compared\s+with|versus|vs\.?|rather\s+than|instead\s+of)\s+([^,.;:]+)",
            r"(?:相比于|而不是|代替)([^，。；;]+)",
        ],
        sentence,
    )
    outcome = _first_match(
        [
            r"\b(?:to|in\s+order\s+to)\s+((?:reduce|prevent|improve|increase|decrease|avoid|detect|control)\s+[^,.;:]+)",
            r"\bfor\s+((?:prevention|treatment|management|diagnosis|screening|control)\s+of\s+[^,.;:]+)",
            r"(?:以|用于)(降低|预防|改善|控制|诊断|筛查[^，。；;]*)",
        ],
        sentence,
    )
    return {
        "population": population,
        "intervention": intervention,
        "comparator": comparator,
        "outcome": outcome,
    }


def extract_grade_quality(sentence: str) -> str:
    match = EVIDENCE_QUALITY_RE.search(sentence)
    if not match:
        return ""
    value = next((group for group in match.groups() if group), "")
    mapping = {"高": "high", "中": "moderate", "低": "low", "极低": "very_low"}
    return mapping.get(value, value.lower().replace(" ", "_"))


def extract_recommendation_strength(sentence: str) -> str:
    match = RECOMMENDATION_STRENGTH_RE.search(sentence)
    if not match:
        return ""
    value = next((group for group in match.groups() if group), "")
    mapping = {"强推荐": "strong", "弱推荐": "weak", "有条件推荐": "conditional"}
    return mapping.get(value, value.lower().replace(" ", "_"))


def make_recommendation_id(metadata: Dict[str, Any], sentence: str) -> str:
    raw = "|".join(
        [
            str(metadata.get("url") or ""),
            str(metadata.get("title") or ""),
            str(metadata.get("chunk_index") or ""),
            sentence,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_from_chunk(item: Dict[str, Any], min_score: float = DEFAULT_MIN_SCORE) -> List[Recommendation]:
    text = str(item.get("page_content") or "")
    metadata = item.get("metadata") or {}
    results: List[Recommendation] = []
    sentences = split_sentences(text)

    for index, sentence in enumerate(sentences):
        score, label = score_recommendation(sentence)
        if score < min_score:
            continue
        clipped = sentence[:MAX_RECOMMENDATION_CHARS].strip()
        evidence_span = clipped
        if index + 1 < len(sentences):
            next_sentence = sentences[index + 1]
            if EVIDENCE_QUALITY_RE.search(next_sentence) or RECOMMENDATION_STRENGTH_RE.search(next_sentence):
                evidence_span = f"{clipped} {next_sentence}".strip()
        results.append(
            Recommendation(
                id=make_recommendation_id(metadata, clipped),
                recommendation_text=clipped,
                score=round(score, 3),
                label=label,
                pico=extract_pico(clipped),
                grade_quality=extract_grade_quality(evidence_span),
                recommendation_strength=extract_recommendation_strength(evidence_span),
                title=str(metadata.get("title") or ""),
                url=str(metadata.get("url") or ""),
                published_date=str(metadata.get("published_date") or ""),
                medical_topic=metadata.get("medical_topic"),
                chunk_index=metadata.get("chunk_index"),
                source=str(metadata.get("source") or ""),
                issuer=str(metadata.get("issuer") or ""),
                evidence_span=evidence_span,
            )
        )
    return results


def extract_recommendations(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> int:
    """从 chunk 文件提取推荐意见并保存 jsonl。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_recommendations = 0
    seen_ids: set[str] = set()

    with output.open("w", encoding="utf-8") as f:
        for item in load_chunks(input_path):
            total_chunks += 1
            recommendations = extract_from_chunk(item, min_score=min_score)
            for rec in recommendations:
                if rec.id in seen_ids:
                    continue
                seen_ids.add(rec.id)
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                total_recommendations += 1

            if total_chunks % 1000 == 0:
                print(f"已处理 {total_chunks} chunks，提取 {total_recommendations} 条推荐意见")

    print(f"完成：处理 {total_chunks} chunks，提取 {total_recommendations} 条推荐意见")
    print(f"输出文件：{output}")
    return total_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="从指南 chunks 中提取推荐意见")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="提取推荐意见")
    extract_parser.add_argument("--input", default=DEFAULT_INPUT, help="输入 chunk jsonl")
    extract_parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出推荐意见 jsonl")
    extract_parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="最低规则置信分")

    args = parser.parse_args()

    if args.command == "extract":
        extract_recommendations(args.input, args.output, min_score=args.min_score)


if __name__ == "__main__":
    main()

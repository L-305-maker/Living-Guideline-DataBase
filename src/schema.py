from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple


Strength = Literal["strong", "moderate", "weak", "against", "none"]
EvidenceLevel = Literal["high", "moderate", "low", "very_low", "none"]
BlockType = Literal["recommendation", "narrative", "table"]
ChunkType = Literal["section_parent", "recommendation_block", "narrative_chunk"]


@dataclass
# 记录文档或 chunk 的来源信息。
class Provenance:
    source: str = ""
    issuer: str = ""
    title: str = ""
    url: str = ""
    published_date: str = ""
    doc_id: str = ""
    page: Optional[int] = None
    char_start: int = 0
    char_end: int = 0


@dataclass
# 表示结构解析后的一个 section 内 block。
class StructuredBlock:
    type: BlockType
    text: str
    char_start: int
    char_end: int
    section_path: str
    section_title: str
    section_index: int
    page: Optional[int] = None
    rec_number: Optional[str] = None


@dataclass
# 表示 section 树中的一个节点。
class SectionNode:
    section_id: str
    title: str
    path: str
    level: int
    index: int
    text: str = ""
    summary: str = ""


@dataclass
# 表示一篇文档的结构解析结果。
class ParsedDocument:
    doc_id: str
    provenance: Provenance
    sections: list[SectionNode]
    blocks: list[StructuredBlock]


@dataclass
# 表示从推荐 block 中抽取出的临床建议。
class RecExtraction:
    statement: str                                     # 推荐意见句子
    strength: Strength = "none"                        #推荐强度
    evidence_level: EvidenceLevel = "none"             #证据等级
    is_rec_score: float = 0.0                          #自写规则判断分数
    confidence: float = 0.0                            #置信度
    source_grade_raw: str = ""                         #原始的GRADE文本，各个来源的数据会有所不同
    grade_source: str = "none"                         #分级来源
    population: Optional[str] = None                   #指南的针对人群
    contraindications: Optional[str] = None            #禁忌事项
    adverse_effects: Optional[str] = None              #副作用
    rationale: Optional[str] = None                    #推荐原因


@dataclass
# 表示最终写入 parent/child collection 的 chunk。
class Chunk:
    chunk_id: str
    doc_id: str
    parent_id: Optional[str]
    level: int
    chunk_type: ChunkType
    text: str
    context: str
    clinical: Dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    embedding_model: str = "ncbi/MedCPT-Article-Encoder"

    # 将 Chunk 转成可直接写入 JSONL 的字典。
    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# 将各个来源的文章不同的评价体系融合到一起
GRADE_MAP: Dict[str, EvidenceLevel] = {
    "high": "high",
    "a": "high",
    "level a": "high",
    "loe a": "high",
    "moderate": "moderate",
    "b": "moderate",
    "b-r": "moderate",
    "b-nr": "moderate",
    "b1": "moderate",
    "level b": "moderate",
    "loe b": "moderate",
    "low": "low",
    "c": "low",
    "c-ld": "low",
    "b2": "low",
    "level c": "low",
    "loe c": "low",
    "very low": "very_low",
    "very_low": "very_low",
    "d": "very_low",
    "e": "very_low",
    "c-eo": "very_low",
}


STRENGTH_MAP: Dict[str, Strength] = {
    "strong": "strong",
    "level 1": "strong",
    "class i": "strong",
    "cor i": "strong",
    "category a": "strong",
    "moderate": "moderate",
    "class iia": "moderate",
    "cor iia": "moderate",
    "weak": "weak",
    "conditional": "weak",
    "level 2": "weak",
    "class iib": "weak",
    "cor iib": "weak",
    "category b": "weak",
    "against": "against",
    "class iii": "against",
    "cor iii": "against",
    "iii-harm": "against",
    "iii-nb": "against",
    "harm": "against",
}
###############################################


# 基于稳定字段生成短哈希，保证 chunk_id 可复现。
def stable_hash(*parts: Any, length: int = 16) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


# 统一文本中的连续空白，方便比较、拼接和入库。
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# 从日期或任意字符串中提取年份。
def keep_year(value: Any) -> str:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return match.group(0) if match else ""


# 将分级文本标准化成便于匹配的 key。
def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", " ", value.lower()).strip()


# 合并来源原始分级、规则预测和模型预测，得到统一枚举值。
def normalize_grade(source: str,raw: str,strength_pred: Strength = "none",evidence_pred: EvidenceLevel = "none",) -> Tuple[Strength, EvidenceLevel, str]:
    source_key = normalize_key(source)
    raw_key = normalize_key(raw)

    strength: Strength = strength_pred
    evidence: EvidenceLevel = evidence_pred
    grade_source = "model" if strength_pred != "none" or evidence_pred != "none" else "none"

    # 优先从原始分级文本中识别推荐强度。
    for key, value in STRENGTH_MAP.items():
        if key in raw_key:
            strength = value
            grade_source = "regex"
            break

    # 再从原始分级文本中识别证据等级。
    for key, value in GRADE_MAP.items():
        if key in raw_key:
            evidence = value
            grade_source = "regex"
            break

    # 不同组织的简写含义不同，这里做少量来源特异修正。
    if source_key in {"ada"} and raw_key in {"a", "b", "c", "e"}:
        evidence = GRADE_MAP.get(raw_key, evidence)
        grade_source = "regex"

    if source_key in {"kdigo"}:
        if "level 1" in raw_key:
            strength = "strong"
        elif "level 2" in raw_key:
            strength = "weak"
            
    if source_key in {"esc", "acc", "aha", "acc aha"}:
        if "class iii" in raw_key or "cor iii" in raw_key:
            strength = "against"

    return strength, evidence, grade_source

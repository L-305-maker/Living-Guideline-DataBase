from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from .schema import EvidenceLevel, RecExtraction, Strength, StructuredBlock, normalize_grade, normalize_text


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

REC_SIGNAL_RE = re.compile(
    r"\b(we\s+recommend|we\s+suggest|recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|"
    r"should|should\s+not|must|must\s+not|avoid|contraindicat|not\s+recommended|"
    r"not\s+indicated|may\s+be\s+considered|offer|provide|administer|initiate|"
    r"start|stop|continue|screen|monitor|refer|treat)\b",
    re.I,
)

GRADE_RE = re.compile(
    r"\b("
    r"strong recommendation|conditional recommendation|weak recommendation|"
    r"high quality|moderate quality|low quality|very low quality|"
    r"high certainty|moderate certainty|low certainty|very low certainty|"
    r"quality of evidence\s*[:=]?\s*(?:high|moderate|low|very low)|"
    r"certainty of evidence\s*[:=]?\s*(?:high|moderate|low|very low)|"
    r"level\s+of\s+evidence\s*[:=]?\s*[A-D]|LOE\s*[:=]?\s*[A-D]|"
    r"Class\s+(?:I|IIa|IIb|III)|COR\s+(?:I|IIa|IIb|III)|"
    r"Level\s+[12]|Grade\s+[A-D]|Category\s+[AB]"
    r")\b",
    re.I,
)

NON_CLINICAL_RE = re.compile(
    r"\b(future studies|further studies|additional research|references?|methods?|"
    r"study should|studies should|authors should|model should|trial should|"
    r"should adhere methodologically|should be interpreted)\b",
    re.I,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|(?<=:)\s+(?=[A-Z])")


# 将 block 文本拆成候选句。
def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [part.strip(" -\t\r\n") for part in SENTENCE_SPLIT_RE.split(text) if part.strip(" -\t\r\n")]


# 从文本中提取原始 GRADE/证据等级片段。
def extract_grade_raw(text: str) -> str:
    matches = [m.group(0) for m in GRADE_RE.finditer(text)]
    return "; ".join(dict.fromkeys(matches))


# 用关键词规则给候选句打推荐意见分数。
def score_recommendation(text: str) -> float:
    score = 0.0
    if REC_SIGNAL_RE.search(text):
        score += 0.65
    if re.search(r"\b(patient|patients|adult|children|disease|diagnos|treat|therapy|dose|risk|screen)\b", text, re.I):
        # 如果有医疗场景的词汇就加上0.65分，更加细致的打分规则会在后续进一步优化
        score += 0.2
    if GRADE_RE.search(text):
        score += 0.15
    if NON_CLINICAL_RE.search(text):
        score -= 0.45
    if len(text) < 25:
        score -= 0.25
    return max(0.0, min(score, 1.0))


# 根据规则预测推荐强度。
def predict_strength(text: str) -> Strength:
    # 依据推荐相关关键词进行评分
    lower = text.lower()
    if re.search(r"\b(should not|must not|do not|avoid|contraindicat|not recommended|not indicated|class iii|cor iii|harm)\b", lower):
        return "against"
    if re.search(r"\b(strong recommendation|we recommend|is recommended|are recommended|must|should|class i|cor i|level 1|category a)\b", lower):
        return "strong"
    if re.search(r"\b(class iia|cor iia|moderate recommendation)\b", lower):
        return "moderate"
    if re.search(r"\b(conditional recommendation|weak recommendation|we suggest|suggest|may be considered|class iib|cor iib|level 2|category b)\b", lower):
        return "weak"
    return "none"


# 根据规则预测证据等级。
def predict_evidence(text: str) -> EvidenceLevel:
    lower = text.lower().replace("-", " ")
    if re.search(r"\b(very low quality|very low certainty|c eo|grade d|level d|evidence d)\b", lower):
        return "very_low"
    if re.search(r"\b(low quality|low certainty|c ld|grade c|level c|loe c|evidence c)\b", lower):
        return "low"
    if re.search(r"\b(moderate quality|moderate certainty|grade b|level b|loe b|b r|b nr|evidence b)\b", lower):
        return "moderate"
    if re.search(r"\b(high quality|high certainty|grade a|level a|loe a|evidence a)\b", lower):
        return "high"
    return "none"


# 按多个正则模式返回第一个命中的槽位文本。
def first_match(patterns: Iterable[str], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return normalize_text(" ".join(group for group in match.groups() if group))
    return None


# 轻量抽取人群、禁忌、副作用和理由等临床槽位。
def extract_slots(text: str) -> Dict[str, Optional[str]]:
    return {
        "population": first_match(
            [
                r"\b(?:in|for|among)\s+((?:adult|pediatric|pregnant|elderly|high-risk|hospitalized)?\s*patients?\s+with\s+[^,.;:]+)",
                r"\b((?:adults?|children|infants?|neonates|pregnant women)\s+with\s+[^,.;:]+)",
            ],
            text,
        ),
        "contraindications": first_match(
            [
                r"\b(?:contraindicated|should not be used|do not use|avoid)\s+(?:in|for)?\s*([^.;]+)",
                r"\bnot\s+(?:recommended|indicated)\s+(?:in|for)?\s*([^.;]+)",
            ],
            text,
        ),
        "adverse_effects": first_match(
            [r"\b(?:adverse effects?|harms?|toxicity|safety concern)\s*(?:include|:)?\s*([^.;]+)"],
            text,
        ),
        "rationale": first_match(
            [r"\b(?:because|as|given that)\s+([^.;]+)", r"\b(?:rationale|reason)\s*[:=]\s*([^.;]+)"],
            text,
        ),
    }


# 封装可选的 HuggingFace 分类器，避免主流程强依赖模型。
class OptionalSequenceClassifier:

    # 初始化可选分类器；默认不开启模型以保证离线可运行。
    def __init__(self, model_name_or_path: Optional[str] = None, enabled: bool = False) -> None:
        import torch
        self.enabled = enabled and bool(model_name_or_path)
        self.model_name_or_path = model_name_or_path or DEFAULT_MODEL_NAME
        self.tokenizer: Any = None
        self.model: Any = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.enabled:
            self._load()

    # 加载本地或远程 HF 分类模型，失败时回退到规则。
    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            logger.warning("transformers/torch not available; using regex recommendation extraction")
            self.enabled = False
            return
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).to(self.device).eval()
        except Exception as exc:
            logger.warning("Could not load classifier %s: %s; using regex fallback", self.model_name_or_path, exc)
            self.enabled = False

    # 对单条文本返回模型置信度；不可用时返回 None。
    def score(self, text: str) -> Optional[float]:
        if not self.enabled or self.model is None or self.tokenizer is None:
            return None
        try:
            import torch
            enc = self.tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
            return float(max(probs))
        except Exception as exc:
            logger.warning("Classifier inference failed: %s", exc)
            return None


# 推荐意见抽取器：规则兜底，可选模型增强。
class RecommendationExtractor:
    # 组合规则抽取和可选模型分类。
    def __init__(self,source: str = "",model_name_or_path: Optional[str] = None,use_model: bool = False,threshold: float = 0.65,) -> None:
        self.source = source
        self.threshold = threshold
        self.classifier = OptionalSequenceClassifier(model_name_or_path, enabled=use_model)

    # 同时计算规则分数和最终分数。
    def _score_candidate(self, text: str) -> tuple[float, float]:
        rule_score = score_recommendation(text)
        model_score = self.classifier.score(text)
        final_score = model_score if model_score is not None else rule_score
        return rule_score, final_score

    # 将一句推荐意见组装成统一 RecExtraction。
    def _build_extraction(self, statement: str, block_text: str, rule_score: float, confidence: float) -> RecExtraction:
        grade_raw = extract_grade_raw(block_text)
        strength_pred = predict_strength(statement + " " + grade_raw)
        evidence_pred = predict_evidence(statement + " " + grade_raw)
        strength, evidence, grade_source = normalize_grade(self.source, grade_raw, strength_pred, evidence_pred)
        if grade_source == "model" and not self.classifier.enabled:
            grade_source = "regex"
        slots = extract_slots(block_text)
        return RecExtraction(
            statement=statement,
            strength=strength,
            evidence_level=evidence,
            is_rec_score=round(rule_score, 4),
            confidence=round(confidence, 4),
            source_grade_raw=grade_raw,
            grade_source=grade_source,
            population=slots["population"],
            contraindications=slots["contraindications"],
            adverse_effects=slots["adverse_effects"],
            rationale=slots["rationale"],
        )

    # 从一个结构 block 中抽取推荐意见和 GRADE 信息。
    def extract(self, block: StructuredBlock) -> List[RecExtraction]:
        candidates = split_sentences(block.text)
        if block.type == "recommendation" and len(candidates) <= 1:
            candidates = [block.text]

        results: List[RecExtraction] = []
        seen: set[str] = set()
        for candidate in candidates:
            candidate = normalize_text(candidate)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)

            # 先高召回评分，再用阈值过滤非推荐句。
            rule_score, final_score = self._score_candidate(candidate)
            if final_score < self.threshold:
                continue

            results.append(self._build_extraction(candidate, block.text, rule_score, final_score))

        # 如果句级拆分没命中，但结构块整体很像推荐，则保留原子块。
        if not results and block.type == "recommendation":
            block_score = score_recommendation(block.text)
            if block_score >= self.threshold:
                results.append(self._build_extraction(block.text, block.text, block_score, block_score))
        return results

from __future__ import annotations

"""把全局 semantic chunker 降级为长 narrative 的语义细分子程序。

现状调用链：旧 `process_data/segment.py` 直接读取整篇 content, 按 MedCPT 相邻句相似度全局切块，
再输出旧格式 `outputs/data.jsonl`, recommendation 与 narrative 没有结构边界保护。

目标调用链：`structure_parser.py` 先给出 section/block 结构，`chunker.py` 只在长 narrative block
内部调用本模块: recommendation block 由 `recommendation_extractor.py` 原子组装，最终输出
`*.parents.jsonl` 与 `*.children.jsonl` 供 `storage.py` 入库.
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from src.schema import normalize_text


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
DEFAULT_MAX_TOKENS = 350
DEFAULT_MIN_TOKENS = 20
DEFAULT_SIM_THRESHOLD = 20.0
DEFAULT_BATCH_SIZE = 32
MODEL_MAX_LENGTH = 512

_DEFAULT_ENCODER: Optional["MedCPTEncoder"] = None

# 保存 MedCPT tokenizer/model/device，便于外部复用同一个实例。
@dataclass
class MedCPTEncoder:
    tokenizer: Any
    model: Any
    device: str
    model_name: str = DEFAULT_MODEL_NAME


# 用轻量规则估算 token 数，避免切分逻辑强依赖 tokenizer。
def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", normalize_text(text), flags=re.UNICODE))


#加载 MedCPT Article Encoder，供 narrative 内部语义细分使用。
def load_encoder(model_name: str = DEFAULT_MODEL_NAME) -> MedCPTEncoder:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading MedCPT encoder on %s: %s", device, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return MedCPTEncoder(tokenizer=tokenizer, model=model, device=device, model_name=model_name)


# 惰性加载默认 encoder，供未显式注入 encoder 的调用使用.
def get_default_encoder(model_name: str = DEFAULT_MODEL_NAME) -> MedCPTEncoder:

    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None or _DEFAULT_ENCODER.model_name != model_name:
        _DEFAULT_ENCODER = load_encoder(model_name)
    return _DEFAULT_ENCODER


# 将 narrative 文本拆成句子级候选单元。
def split_sentences(text: str) -> List[str]:

    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(
        r"(?<=[.!?;])\s+|(?<=:)\s+(?=[A-Z])|"
        r"(?=\b(?:Children|Adults|Neonates|First-line|Second-line|Alternative|"
        r"Treatment|Clinical features|Diagnosis|Management|Prevention)\b)",
        text,
    )
    return [part.strip(" -\t\r\n") for part in parts if part.strip(" -\t\r\n")]


# 在不用语义模型时按长度合并句子，保证 chunk 不过碎。
def pack_by_tokens(units: Sequence[str], max_tokens: int, min_tokens: int = DEFAULT_MIN_TOKENS) -> List[str]:

    chunks: List[str] = []
    current: List[str] = []
    for unit in units:
        unit = normalize_text(unit)
        if not unit:
            continue
        candidate = " ".join([*current, unit]).strip()
        if current and estimate_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current).strip())
            current = [unit]
        else:
            current.append(unit)
    if current:
        chunks.append(" ".join(current).strip())

    merged: List[str] = []
    for chunk in chunks:
        if merged and estimate_tokens(chunk) < min_tokens:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)
    return merged


# 批量计算句向量并做 L2 归一化.
def embed_sentences(sentences: Sequence[str], encoder: MedCPTEncoder, batch_size: int = DEFAULT_BATCH_SIZE) -> Any:

    import numpy as np
    import torch

    vectors = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = list(sentences[start : start + batch_size])
            encoded = encoder.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MODEL_MAX_LENGTH,
                return_tensors="pt",
            ).to(encoder.device)
            output = encoder.model(**encoded).last_hidden_state[:, 0, :]
            output = torch.nn.functional.normalize(output, p=2, dim=1)
            vectors.append(output.cpu().numpy())
    return np.vstack(vectors)


# 把阈值统一成相似度 cutoff；大于 1 时兼容旧代码的 percentile 语义。
def _similarity_cutoff(similarities: Sequence[float], sim_threshold: float) -> float:

    import numpy as np

    if not similarities:
        return 1.0
    if 0.0 < sim_threshold <= 1.0:
        return sim_threshold
    return float(np.percentile(list(similarities), sim_threshold))


# 递归选择相邻句相似度最低处切开，保留原算法的核心行为。
def _split_by_low_valley(units: Sequence[str],similarities: Sequence[float],start: int,end: int,max_tokens: int,min_tokens: int,) -> List[str]:
    
    current = " ".join(units[start:end]).strip()
    if estimate_tokens(current) <= max_tokens or end - start <= 1:
        return [current] if current else []

    candidates = []
    for split_at in range(start + 1, end):
        left = " ".join(units[start:split_at]).strip()
        right = " ".join(units[split_at:end]).strip()
        if estimate_tokens(left) >= min_tokens and estimate_tokens(right) >= min_tokens:
            candidates.append(split_at)

    if not candidates:
        candidates = list(range(start + 1, end))

    split_at = min(candidates, key=lambda idx: similarities[idx - 1])
    return _split_by_low_valley(units, similarities, start, split_at, max_tokens, min_tokens) + _split_by_low_valley(
        units, similarities, split_at, end, max_tokens, min_tokens
    )


# 只切长 narrative；短文本原样返回，长文本按 MedCPT 相邻句相似度细分。
def split_long_narrative(text: str,*,max_tokens:int = DEFAULT_MAX_TOKENS,sim_threshold: float = DEFAULT_SIM_THRESHOLD,encoder: Optional[MedCPTEncoder] = None,) -> List[str]:

    text = normalize_text(text)
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    units = split_sentences(text)
    if len(units) <= 1:
        return [text]

    active_encoder = encoder or get_default_encoder()
    embeddings = embed_sentences(units, active_encoder)

    import numpy as np

    similarities = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
    cutoff = _similarity_cutoff(similarities, sim_threshold)

    groups: List[str] = []
    current = [units[0]]
    for idx, sim in enumerate(similarities):
        candidate = " ".join([*current, units[idx + 1]]).strip()
        should_split = sim < cutoff and estimate_tokens(" ".join(current)) >= DEFAULT_MIN_TOKENS
        too_long = estimate_tokens(candidate) > max_tokens
        if should_split or too_long:
            groups.append(" ".join(current).strip())
            current = [units[idx + 1]]
        else:
            current.append(units[idx + 1])
    if current:
        groups.append(" ".join(current).strip())

    chunks: List[str] = []
    for group in groups:
        if estimate_tokens(group) > max_tokens:
            group_units = split_sentences(group)
            if len(group_units) <= 1:
                chunks.append(group)
                continue
            group_embeddings = embed_sentences(group_units, active_encoder)
            group_similarities = [float(np.dot(group_embeddings[i], group_embeddings[i + 1])) for i in range(len(group_embeddings) - 1)]
            chunks.extend(_split_by_low_valley(group_units, group_similarities, 0, len(group_units), max_tokens, DEFAULT_MIN_TOKENS))
        else:
            chunks.append(group)

    return [chunk for chunk in pack_by_tokens(chunks, max_tokens=max_tokens, min_tokens=DEFAULT_MIN_TOKENS) if chunk]


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="DEPRECATED: split one long narrative text with MedCPT")
    parser.add_argument("--text", default="", help="Narrative text to split")
    parser.add_argument("--input", default="", help="Plain text input file")
    parser.add_argument("--output", default="", help="JSON output file; defaults to stdout")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """保留 thin CLI 兼容手工测试，不再承担全局 chunker 职责。"""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")
    logger.warning("src.segment CLI is deprecated; use src.chunker for the parent-child pipeline")

    if args.text:
        text = args.text
    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    encoder = load_encoder(args.model)
    chunks = split_long_narrative(text, max_tokens=args.max_tokens, sim_threshold=args.sim_threshold, encoder=encoder)
    payload = json.dumps(chunks, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""结构优先的 parent-child chunk 调度器。

本模块负责把原始 record 清洗去重后交给结构解析器，再把 section/block 组装成
parents.jsonl 和 children.jsonl。recommendation block 保持原子，长 narrative block
才会调用 MedCPT 语义细分。
"""

import argparse
import glob
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.process_data.cleaner import clean_records
from src.process_data.deduplicator import deduplicate_records
from src.recommendation_extractor import RecommendationExtractor
from src.schema import Chunk, ParsedDocument, Provenance, RecExtraction, SectionNode, StructuredBlock, normalize_text, stable_hash
from src.segment import (
    DEFAULT_MAX_TOKENS,
    MedCPTEncoder,
    estimate_tokens,
    load_encoder,
    pack_by_tokens,
    split_long_narrative,
    split_sentences,
)
from src.structure_parser import parse_record
from src.utils.process_jsonl import iter_jsonl, write_jsonl_obj


logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS = 50


@dataclass
class ProcessingStats:
    """记录单次或批量处理的统计信息。"""

    records: int = 0
    parents: int = 0
    children: int = 0
    recommendation: int = 0
    narrative: int = 0
    failed: int = 0

    def add(self, other: "ProcessingStats") -> None:
        # 合并另一个统计对象。

        self.records += other.records
        self.parents += other.parents
        self.children += other.children
        self.recommendation += other.recommendation
        self.narrative += other.narrative
        self.failed += other.failed

    def count_child(self, chunk: Chunk) -> None:
        # 根据 child 类型累加统计。

        self.children += 1
        if chunk.chunk_type == "recommendation_block":
            self.recommendation += 1
        elif chunk.chunk_type == "narrative_chunk":
            self.narrative += 1

    def to_dict(self) -> Dict[str, int]:
        # 转成日志友好的字典。
        return asdict(self)


def write_chunk(handle: Any, chunk: Chunk) -> None:
    # 将一个 Chunk 写入 JSONL。
    write_jsonl_obj(handle, chunk.to_json())


# 基于 block 字符范围生成更精确的来源信息。
def provenance_for_block(doc: ParsedDocument, block: StructuredBlock) -> Provenance:

    return replace(
        doc.provenance,
        page=block.page,
        char_start=block.char_start,
        char_end=block.char_end,
    )

   
    # 生成 child 携带的 parent 上下文。
def section_context(doc: ParsedDocument, section: SectionNode) -> str:

    parts = [
        doc.provenance.title,
        section.path,
        f"Source: {doc.provenance.source}" if doc.provenance.source else "",
        f"Issuer: {doc.provenance.issuer}" if doc.provenance.issuer else "",
        f"Year: {doc.provenance.published_date}" if doc.provenance.published_date else "",
        section.summary,
    ]
    return "\n".join(part for part in parts if part)


def make_parent_chunk(doc: ParsedDocument, section: SectionNode) -> Chunk:
    """将 section 转成 parent collection 的 chunk。"""

    chunk_id = f"{doc.doc_id}_sec{section.index}"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        parent_id=None,
        level=section.level,
        chunk_type="section_parent",
        text=normalize_text(section.text),
        context=section_context(doc, section),
        clinical={},
        provenance=doc.provenance,
    )


def recommendation_child_text(block: StructuredBlock, recs: List[RecExtraction]) -> str:
    """把推荐抽取结果拼回 recommendation block 文本。"""

    if not recs:
        return block.text

    details: List[str] = [block.text]
    for rec in recs:
        extras: List[str] = []
        if rec.population:
            extras.append(f"Population: {rec.population}")
        if rec.strength != "none":
            extras.append(f"Strength: {rec.strength}")
        if rec.evidence_level != "none":
            extras.append(f"Evidence: {rec.evidence_level}")
        if rec.contraindications:
            extras.append(f"Contraindications: {rec.contraindications}")
        if rec.adverse_effects:
            extras.append(f"Adverse effects: {rec.adverse_effects}")
        if rec.rationale:
            extras.append(f"Rationale: {rec.rationale}")
        if extras:
            details.append(" | ".join(extras))
    return normalize_text(" ".join(details))


def make_recommendation_child(
    doc: ParsedDocument,
    block: StructuredBlock,
    parent: Chunk,
    recs: List[RecExtraction],
    rec_index: int,
) -> Chunk:
    """将 recommendation block 转成 child collection 的原子 chunk。"""

    strength = next((rec.strength for rec in recs if rec.strength != "none"), "none")
    evidence_level = next((rec.evidence_level for rec in recs if rec.evidence_level != "none"), "none")
    chunk_id = f"{doc.doc_id}_sec{block.section_index}_rec{block.rec_number or rec_index}"
    clinical = {
        "strength": strength,
        "evidence_level": evidence_level,
        "recommendations": [asdict(rec) for rec in recs],
        "source_grade_raw": "; ".join(rec.source_grade_raw for rec in recs if rec.source_grade_raw),
        "grade_source": next((rec.grade_source for rec in recs if rec.grade_source != "none"), "none"),
    }
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        parent_id=parent.chunk_id,
        level=parent.level + 1,
        chunk_type="recommendation_block",
        text=recommendation_child_text(block, recs),
        context=parent.context,
        clinical=clinical,
        provenance=provenance_for_block(doc, block),
    )


def make_narrative_child(doc: ParsedDocument, block: StructuredBlock, parent: Chunk, narr_index: int) -> Chunk:
    """将 narrative block 转成 child collection 的 chunk。"""

    chunk_id = f"{doc.doc_id}_sec{block.section_index}_narr{narr_index}"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        parent_id=parent.chunk_id,
        level=parent.level + 1,
        chunk_type="narrative_chunk",
        text=block.text,
        context=parent.context,
        clinical={},
        provenance=provenance_for_block(doc, block),
    )


def split_text_to_blocks(block: StructuredBlock, texts: List[str]) -> List[StructuredBlock]:
    """把 segment 返回的纯文本片段恢复成带 provenance 的 StructuredBlock。"""

    out: List[StructuredBlock] = []
    cursor = 0
    for text in texts:
        text = normalize_text(text)
        if not text:
            continue
        relative_start = block.text.find(text, cursor)
        if relative_start < 0:
            relative_start = block.text.find(text)
        if relative_start < 0:
            relative_start = cursor
        relative_end = relative_start + len(text)
        cursor = relative_end
        out.append(
            StructuredBlock(
                type="narrative",
                text=text,
                char_start=block.char_start + relative_start,
                char_end=block.char_start + relative_end,
                section_path=block.section_path,
                section_title=block.section_title,
                section_index=block.section_index,
                page=block.page,
                rec_number=block.rec_number,
            )
        )
    return out


def split_narrative_block(
    block: StructuredBlock,
    max_tokens: int,
    min_chars: int,
    sim_threshold: float,
    encoder: Optional[MedCPTEncoder],
    use_semantic_split: bool,
) -> List[StructuredBlock]:
    """只对长 narrative 做细分；短 narrative 直接保留。"""

    narrative_block = block if block.type == "narrative" else replace(block, type="narrative")
    text = normalize_text(narrative_block.text)
    if len(text) < min_chars or estimate_tokens(text) <= max_tokens:
        return [replace(narrative_block, text=text)] if text else []

    if use_semantic_split:
        texts = split_long_narrative(text, max_tokens=max_tokens, sim_threshold=sim_threshold, encoder=encoder)
    else:
        texts = pack_by_tokens(split_sentences(text), max_tokens=max_tokens)
    return split_text_to_blocks(narrative_block, texts)


def append_narrative_children(
    children: List[Chunk],
    doc: ParsedDocument,
    block: StructuredBlock,
    parent: Chunk,
    start_index: int,
    max_tokens: int,
    min_chars: int,
    sim_threshold: float,
    encoder: Optional[MedCPTEncoder],
    use_semantic_split: bool,
) -> int:
    """切分并追加 narrative child，返回下一个 narrative 序号。"""

    next_index = start_index
    for split_block in split_narrative_block(block, max_tokens, min_chars, sim_threshold, encoder, use_semantic_split):
        if len(normalize_text(split_block.text)) < min_chars:
            continue
        children.append(make_narrative_child(doc, split_block, parent, next_index))
        next_index += 1
    return next_index


def chunk_record(
    record: Dict[str, Any],
    source: str,
    extractor: RecommendationExtractor,
    max_tokens: int,
    min_chars: int,
    sim_threshold: float,
    encoder: Optional[MedCPTEncoder] = None,
    use_semantic_split: bool = True,
) -> Tuple[List[Chunk], List[Chunk]]:
    """处理单条原始 record，产出 parent 和 child 两类 chunk。"""

    parsed = parse_record(record, fallback_source=source)
    sections = {section.index: section for section in parsed.sections}
    parents: List[Chunk] = []
    children: List[Chunk] = []
    parent_by_section: Dict[int, Chunk] = {}

    for section in parsed.sections:
        parent = make_parent_chunk(parsed, section)
        parents.append(parent)
        parent_by_section[section.index] = parent

    rec_counter = 0
    narr_counter = 0
    for block in parsed.blocks:
        parent = parent_by_section.get(block.section_index)
        if parent is None:
            section = sections.get(block.section_index) or SectionNode(
                section_id=stable_hash(parsed.doc_id, "section", block.section_index),
                title=block.section_title,
                path=block.section_path,
                level=1,
                index=block.section_index,
                text=block.text,
                summary=block.text[:500],
            )
            parent = make_parent_chunk(parsed, section)
            parents.append(parent)
            parent_by_section[block.section_index] = parent

        if block.type == "recommendation":
            recs = extractor.extract(block)
            if recs:
                children.append(make_recommendation_child(parsed, block, parent, recs, rec_counter))
                rec_counter += 1
                continue
            logger.debug("Recommendation-like block downgraded to narrative: %s", block.text[:120])

        if block.type in {"narrative", "recommendation"}:
            narr_counter = append_narrative_children(
                children,
                parsed,
                block,
                parent,
                narr_counter,
                max_tokens,
                min_chars,
                sim_threshold,
                encoder,
                use_semantic_split,
            )

    return parents, children


def expand_inputs(pattern: str) -> List[Path]:
    """展开单文件路径或 glob 输入。"""

    matches = [Path(path) for path in glob.glob(pattern)]
    if matches:
        return sorted(matches)
    path = Path(pattern)
    return [path] if path.exists() else []


def default_outputs(input_path: Path, out_dir: Optional[Path]) -> Tuple[Path, Path]:
    """根据输入文件和输出目录推导 parents/children 文件名。"""

    base = input_path.stem
    root = out_dir or input_path.parent
    return root / f"{base}.parents.jsonl", root / f"{base}.children.jsonl"


def prepare_records(input_path: Path, clean_input: bool, deduplicate_input: bool) -> List[Dict[str, Any]]:
    """读取 JSONL 后用 record-first cleaner/deduplicator 做预处理。"""

    raw_records = list(iter_jsonl(input_path))
    records = clean_records(raw_records) if clean_input else raw_records
    records = deduplicate_records(records) if deduplicate_input else records
    logger.info(
        "Prepared %s records from %s: raw=%s clean=%s dedup=%s",
        len(records),
        input_path,
        len(raw_records),
        clean_input,
        deduplicate_input,
    )
    return records


def process_file(
    input_path: Path,
    source: str,
    out_parents: Path,
    out_children: Path,
    extractor: RecommendationExtractor,
    max_tokens: int,
    min_chars: int,
    sim_threshold: float,
    encoder: Optional[MedCPTEncoder],
    use_semantic_split: bool,
    clean_input: bool,
    deduplicate_input: bool,
) -> ProcessingStats:
    """处理一个 JSONL 文件并写出 parents/children 两个 JSONL 文件。"""

    out_parents.parent.mkdir(parents=True, exist_ok=True)
    out_children.parent.mkdir(parents=True, exist_ok=True)
    stats = ProcessingStats()
    records = prepare_records(input_path, clean_input=clean_input, deduplicate_input=deduplicate_input)

    with out_parents.open("w", encoding="utf-8") as parents_handle, out_children.open("w", encoding="utf-8") as children_handle:
        for record in records:
            stats.records += 1
            try:
                parents, children = chunk_record(
                    record,
                    source,
                    extractor,
                    max_tokens,
                    min_chars,
                    sim_threshold,
                    encoder,
                    use_semantic_split,
                )
            except Exception as exc:
                stats.failed += 1
                logger.exception("Failed record in %s: %s", input_path, exc)
                continue
            for parent in parents:
                write_chunk(parents_handle, parent)
                stats.parents += 1
            for child in children:
                write_chunk(children_handle, child)
                stats.count_child(child)
    return stats


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Build structure-first parent/child medical guideline chunks")
    parser.add_argument("--in", dest="input_pattern", required=True, help="Input JSONL path or glob")
    parser.add_argument("--source", default="", help="Fallback source name")
    parser.add_argument("--out-parents", default="", help="Parents output path for single input")
    parser.add_argument("--out-children", default="", help="Children output path for single input")
    parser.add_argument("--out-dir", default="", help="Output directory for glob/batch mode")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-chars", dest="max_tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sim-threshold", type=float, default=20.0, help="<=1 means cosine cutoff; >1 means percentile")
    parser.add_argument("--use-medcpt", action="store_true", help="Use MedCPT semantic split for long narrative blocks")
    parser.add_argument("--medcpt-model", default="ncbi/MedCPT-Article-Encoder")
    parser.add_argument("--use-rec-model", action="store_true", help="Use optional HF sequence classifier")
    parser.add_argument("--rec-model", default="", help="Local recommendation classifier path")
    parser.add_argument("--no-clean", action="store_true", help="Skip record cleaner before structure parsing")
    parser.add_argument("--no-dedup", action="store_true", help="Skip record deduplicator before structure parsing")
    parser.add_argument("--log-level", default="INFO")
    
    return parser.parse_args()


def main() -> None:
    """CLI 入口：批量执行 parent-child chunking。"""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")

    inputs = expand_inputs(args.input_pattern)
    if not inputs:
        raise FileNotFoundError(args.input_pattern)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if len(inputs) > 1 and (args.out_parents or args.out_children) and not out_dir:
        raise ValueError("--out-parents/--out-children are only valid for one input; use --out-dir for glob mode")

    encoder = load_encoder(args.medcpt_model) if args.use_medcpt else None
    extractor = RecommendationExtractor(source=args.source, model_name_or_path=args.rec_model or None, use_model=args.use_rec_model)

    total = ProcessingStats()
    for input_path in inputs:
        parents_path, children_path = default_outputs(input_path, out_dir)
        if len(inputs) == 1:
            if args.out_parents:
                parents_path = Path(args.out_parents)
            if args.out_children:
                children_path = Path(args.out_children)
        stats = process_file(
            input_path=input_path,
            source=args.source or input_path.stem,
            out_parents=parents_path,
            out_children=children_path,
            extractor=extractor,
            max_tokens=args.max_tokens,
            min_chars=args.min_chars,
            sim_threshold=args.sim_threshold,
            encoder=encoder,
            use_semantic_split=args.use_medcpt,
            clean_input=not args.no_clean,
            deduplicate_input=not args.no_dedup,
        )
        logger.info("Processed %s -> parents=%s children=%s stats=%s", input_path, parents_path, children_path, stats.to_dict())
        total.add(stats)
    logger.info("Total stats: %s", total.to_dict())


if __name__ == "__main__":
    main()

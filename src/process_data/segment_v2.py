from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    title: str
    text: str
    level: int
    ordinal: int
    parent_titles: List[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        parts = [*self.parent_titles, self.title]
        return " > ".join(part for part in parts if part)


@dataclass
class ChunkConfig:
    max_chars: int = 1400
    min_chars: int = 50
    parent_max_chars: int = 2800
    semantic_threshold_pct: int = 20
    recommendation_context_sentences: int = 4


RECOMMENDATION_DIRECT_RE = re.compile(
    r"\b("
    r"we\s+recommend|we\s+suggest|recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|"
    r"should|should\s+not|must|must\s+not|"
    r"do\s+not\s+(?:use|administer|offer|provide|initiate|start|continue|treat|screen|monitor|perform)|"
    r"avoid|contraindicat|not\s+recommended|not\s+indicated|may\s+be\s+considered|"
    r"it\s+is\s+reasonable\s+to"
    r")\b|"
    r"(recommendation\s+\d+|class\s+(?:i|ii|iii|1|2|3)|grade\s+[abc]|"
    r"strong\s+recommendation|conditional\s+recommendation|weak\s+recommendation)",
    re.IGNORECASE,
)

RECOMMENDATION_ACTION_RE = re.compile(
    r"\b("
    r"offer|provide|administer|initiate|start|stop|continue|use|treat|screen|"
    r"monitor|refer|perform|assess|consider"
    r")\b",
    re.IGNORECASE,
)

ACTION_AT_START_RE = re.compile(
    r"^(?:"
    r"offer|provide|administer|initiate|start|stop|continue|use|treat|screen|"
    r"monitor|refer|perform|assess|consider"
    r")\b",
    re.IGNORECASE,
)

RECOMMENDATION_SUPPORT_RE = re.compile(
    r"\b("
    r"recommendation\s+strength|strength\s+of\s+recommendation|class\s+of\s+recommendation|"
    r"level\s+of\s+evidence|certainty\s+of\s+evidence|quality\s+of\s+evidence|"
    r"evidence\s+level|evidence\s+grade|rationale|remarks?|implementation|"
    r"benefits?|harms?|adverse|toxicity|safety|contraindications?|cautions?|"
    r"not\s+recommended|not\s+indicated|does\s+not\s+apply|exceptions?|"
    r"population|patients?\s+with|for\s+patients?|in\s+patients?|because|"
    r"high\s+quality|moderate\s+quality|low\s+quality|very\s+low\s+quality|"
    r"high\s+certainty|moderate\s+certainty|low\s+certainty|very\s+low\s+certainty"
    r")\b",
    re.IGNORECASE,
)

CLINICAL_SIGNAL_RE = re.compile(
    r"\b("
    r"patient|patients|adult|adults|child|children|infant|neonate|pregnan|"
    r"disease|syndrome|infection|cancer|diabetes|hypertension|stroke|asthma|"
    r"diagnos|treat|therapy|management|screening|prevention|dose|dosage|mg|ml|kg|"
    r"risk|symptom|clinical|surgery|vaccine|drug|antibiotic|blood pressure|"
    r"renal|kidney|cardiovascular|mortality|morbidity"
    r")\b",
    re.IGNORECASE,
)

BACKGROUND_SECTION_RE = re.compile(
    r"\b("
    r"background|overview|introduction|summary|evidence|rationale|discussion|"
    r"epidemiology|pathophysiology|methods?|literature|review|appendix"
    r")\b",
    re.IGNORECASE,
)

RECOMMENDATION_SECTION_RE = re.compile(
    r"\b(recommendation|recommendations|guideline statements?|statements?)\b",
    re.IGNORECASE,
)

NOISE_RE = re.compile(
    r"\b("
    r"references?|bibliography|acknowledg|copyright|permission|appendix|"
    r"supplementary|table\s+of\s+contents|figure|doi|pmid|isbn|citation"
    r")\b",
    re.IGNORECASE,
)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    records.extend(iter_jsonl(path))
    return records


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Line {line_no} JSON parse error: {exc}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                yield item


def keep_year(value: Any) -> str:
    if not value:
        return ""
    match = re.search(r"(?:19|20)\d{2}", str(value).strip())
    return match.group(0) if match else ""


def normalize_topic(value: Any) -> Any:
    if isinstance(value, list):
        return value
    return value or ""


def process_data(data: Sequence[Dict[str, Any]]) -> List[Document]:
    return [process_record(sample) for sample in data]


def process_record(sample: Dict[str, Any]) -> Document:
    content = sample.get("content_markdown") or sample.get("content") or sample.get("abstract") or ""
    return Document(
        page_content=content,
        metadata={
            "published_date": keep_year(sample.get("published_date") or sample.get("last_updated")),
            "title": sample.get("title") or "",
            "medical_topic": normalize_topic(sample.get("medical_topics") or sample.get("keywords")),
            "url": sample.get("url") or "",
            "source": sample.get("source") or "",
            "issuer": sample.get("issuer") or "",
            "quality": sample.get("quality"),
            "recommendation": sample.get("recommendations") or sample.get("recommendation"),
            "has_recommendations": bool(sample.get("has_recommendations") or sample.get("recommendation_count")),
        },
    )


class DocumentCleaner:
    MOJIBAKE_REPLACEMENTS = {
        "\ufffd": "",
        "\u920d?": ">=",
        "\u920d\ufffd": ">=",
        "\u920d\u6a9a": "'s",
        "\u920d\u6a9b": "'t",
        "\u920d\u6a99": "'r",
        "\u920d\u6a9d": "'v",
        "\u920d\u6a91": "'l",
        "\u920d\uff1f": "'",
    }

    def clean_text(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for bad, good in self.MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)

        cleaned_lines: List[str] = []
        for line in text.split("\n"):
            line = re.sub(r"^\s*#+\s*", "", line)
            line = re.sub(r"[ \t]+", " ", line).strip()
            line = re.sub(r"\s+([,.;:])", r"\1", line)
            line = re.sub(r"([(\[])\s+", r"\1", line)
            line = re.sub(r"\s+([)\]])", r"\1", line)
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean(self, documents: Sequence[Document]) -> List[Document]:
        cleaned: List[Document] = []
        for doc in documents:
            text = self.clean_text(doc.page_content)
            if len(collapse_ws(text)) < 20:
                continue
            cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
        return cleaned


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def hash_text(text: str) -> str:
    normalized = collapse_ws(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicator(data: Sequence[Document]) -> List[Document]:
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in data:
        digest = hash_text(doc.page_content)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(doc)
    return unique


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 160:
        return False
    if RECOMMENDATION_DIRECT_RE.search(line) and len(line) > 80:
        return False
    if line.endswith((".", ";", ",")) and not re.match(r"^\d+(?:\.\d+)*\.?\s+", line):
        return False
    if re.match(r"^(?:chapter|section|part|appendix)\s+\d+", line, flags=re.I):
        return True
    if re.match(r"^\d+(?:\.\d+){0,4}\.?\s+\S+", line):
        return True
    if re.match(r"^[A-Z][A-Z0-9 /,&():'-]{3,}$", line) and len(line.split()) <= 14:
        return True
    if re.search(
        r"\b("
        r"quick reference|recommendations?|summary|background|evidence|rationale|"
        r"management|diagnosis|treatment|prevention|screening|monitoring|"
        r"clinical policy|patient resources|related guidelines|apps and tools"
        r")\b",
        line,
        flags=re.I,
    ) and len(line.split()) <= 12:
        return True
    return False


def heading_level(line: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", line)
    if match:
        return min(match.group(1).count(".") + 1, 6)
    if re.match(r"^(?:chapter|part)\s+\d+", line, flags=re.I):
        return 1
    if re.match(r"^[A-Z][A-Z0-9 /,&():'-]{3,}$", line):
        return 2
    return 3


def split_inline_heading_text(line: str) -> List[str]:
    patterns = [
        r"(Quick Reference|Related Clinical Policy|Related Guidelines|Perspectives|Education|"
        r"Apps and Tools|Patient Resources|Slides|Recommendations?|Background|Evidence|"
        r"Rationale|Summary|Diagnosis|Treatment|Management|Prevention|Screening)",
    ]
    result = [line]
    for pattern in patterns:
        new_result: List[str] = []
        for item in result:
            parts = re.split(pattern, item, flags=re.I)
            if len(parts) == 1:
                new_result.append(item)
                continue
            prefix = parts[0].strip()
            if prefix:
                new_result.append(prefix)
            for i in range(1, len(parts), 2):
                heading = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                new_result.append(heading)
                if body:
                    new_result.append(body)
        result = new_result
    return [part.strip() for part in result if part.strip()]


def iter_structural_lines(text: str) -> List[str]:
    raw_lines = [line.strip() for line in str(text or "").split("\n")]
    if len([line for line in raw_lines if line]) > 1:
        return [line for line in raw_lines if line]

    line = collapse_ws(text)
    if not line:
        return []
    return split_inline_heading_text(line)


def parse_sections(doc: Document) -> List[Section]:
    lines = iter_structural_lines(doc.page_content)
    root_title = str(doc.metadata.get("title") or "Untitled guideline").strip()
    sections: List[Section] = []
    stack: List[Tuple[int, str]] = [(0, root_title)]
    current_title = "Document"
    current_level = 1
    current_parent_titles = [root_title]
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(
                Section(
                    title=current_title,
                    text=text,
                    level=current_level,
                    ordinal=len(sections),
                    parent_titles=list(current_parent_titles),
                )
            )
        current_lines = []

    for line in lines:
        if looks_like_heading(line):
            flush()
            level = heading_level(line)
            while stack and stack[-1][0] >= level:
                stack.pop()
            current_parent_titles = [title for _, title in stack]
            current_title = line.strip(" :")
            current_level = level
            stack.append((level, current_title))
            continue
        current_lines.append(line)

    flush()
    if sections:
        return sections
    return [Section(title="Document", text=doc.page_content, level=1, ordinal=0, parent_titles=[root_title])]


def split_sentences(text: str) -> List[str]:
    text = str(text or "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+|(?<=:)\s+(?=[A-Z])", text)
    return [part.strip(" \t\r\n-*") for part in parts if part.strip(" \t\r\n-*")]


def section_units(section: Section) -> List[str]:
    units: List[str] = []
    for line in section.text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:[-*]|\d+\.|[A-Z]\.)\s+", line):
            units.append(re.sub(r"^(?:[-*]|\d+\.|[A-Z]\.)\s+", "", line).strip())
        else:
            units.extend(split_sentences(line))
    return [unit for unit in units if unit]


def is_noise(text: str) -> bool:
    normalized = collapse_ws(text)
    if not normalized:
        return True
    urls = re.findall(r"https?://\S+|www\.\S+|doi\.org/\S+", normalized, flags=re.I)
    has_clinical_signal = bool(CLINICAL_SIGNAL_RE.search(normalized))
    if urls and not has_clinical_signal and len(normalized) < 260:
        return True
    if NOISE_RE.search(normalized) and not has_clinical_signal and not RECOMMENDATION_DIRECT_RE.search(normalized):
        return True
    return False


def is_low_value_chunk(text: str, min_chars: int) -> bool:
    normalized = collapse_ws(text)
    if len(normalized) < min_chars:
        return True
    if is_noise(normalized):
        return True
    if re.fullmatch(r"(references?\s*)?\d+\.?", normalized, flags=re.I):
        return True
    return False


def is_recommendation_start(unit: str, section: Section) -> bool:
    has_clinical_signal = bool(CLINICAL_SIGNAL_RE.search(unit))
    in_recommendation_section = bool(RECOMMENDATION_SECTION_RE.search(section.title))

    if RECOMMENDATION_DIRECT_RE.search(unit) and has_clinical_signal:
        return True
    if in_recommendation_section and RECOMMENDATION_ACTION_RE.search(unit):
        return True
    if ACTION_AT_START_RE.search(unit) and has_clinical_signal and in_recommendation_section:
        return True
    if re.match(r"^(?:recommendation|statement)\s+\d+", unit, flags=re.I):
        return True
    return False


def is_recommendation_support(unit: str) -> bool:
    return bool(RECOMMENDATION_SUPPORT_RE.search(unit) or CLINICAL_SIGNAL_RE.search(unit))


def build_parent_context(doc: Document, section: Section, max_chars: int) -> str:
    title = str(doc.metadata.get("title") or "").strip()
    topic = doc.metadata.get("medical_topic")
    topic_text = ", ".join(map(str, topic)) if isinstance(topic, list) else str(topic or "")
    header_parts = [title, section.path]
    if topic_text:
        header_parts.append(f"Topic: {topic_text}")
    header = " | ".join(part for part in header_parts if part)
    body = collapse_ws(section.text)
    context = f"{header}\n{body}".strip()
    if len(context) <= max_chars:
        return context
    keep = max(max_chars - len(header) - 4, 200)
    return f"{header}\n{body[:keep].rstrip()}..."


def make_chunk_id(metadata: Dict[str, Any], section: Section, text: str, chunk_type: str) -> str:
    raw = "|".join(
        [
            str(metadata.get("url") or ""),
            str(metadata.get("title") or ""),
            str(section.ordinal),
            chunk_type,
            collapse_ws(text)[:400],
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def make_chunk(
    doc: Document,
    section: Section,
    text: str,
    chunk_index: int,
    chunk_type: str,
    parent_context: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Document:
    normalized = collapse_ws(text)
    metadata = {
        **doc.metadata,
        "chunk_index": chunk_index,
        "chunk_id": make_chunk_id(doc.metadata, section, normalized, chunk_type),
        "chunk_type": chunk_type,
        "char_len": len(normalized),
        "section_title": section.title,
        "section_path": section.path,
        "section_level": section.level,
        "section_index": section.ordinal,
        "parent_title": section.parent_titles[-1] if section.parent_titles else "",
        "parent_path": " > ".join(section.parent_titles),
        "parent_context": parent_context,
    }
    if extra:
        metadata.update(extra)
    return Document(page_content=normalized, metadata=metadata)


def extract_recommendation_blocks(
    doc: Document,
    section: Section,
    config: ChunkConfig,
    start_chunk_index: int,
) -> Tuple[List[Document], List[int]]:
    units = section_units(section)
    chunks: List[Document] = []
    consumed: set[int] = set()
    parent_context = build_parent_context(doc, section, config.parent_max_chars)
    chunk_index = start_chunk_index
    i = 0

    while i < len(units):
        unit = units[i]
        if not is_recommendation_start(unit, section):
            i += 1
            continue

        block = [unit]
        consumed.add(i)
        j = i + 1
        support_count = 0
        while j < len(units):
            next_unit = units[j]
            if is_recommendation_start(next_unit, section):
                break
            would_be = collapse_ws(" ".join([*block, next_unit]))
            if len(would_be) > config.max_chars and support_count >= 1:
                break
            if is_recommendation_support(next_unit) or support_count < config.recommendation_context_sentences:
                block.append(next_unit)
                consumed.add(j)
                support_count += 1
                j += 1
                continue
            break

        text = " ".join(block)
        if not is_low_value_chunk(text, config.min_chars):
            chunks.append(
                make_chunk(
                    doc,
                    section,
                    text,
                    chunk_index,
                    "recommendation_block",
                    parent_context,
                    {
                        "recommendation_atomic": True,
                        "recommendation_unit_count": len(block),
                    },
                )
            )
            chunk_index += 1
        i = max(j, i + 1)

    return chunks, sorted(consumed)


def load_model() -> Tuple[Any, Any, str]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("MedCPT semantic splitting requires torch and transformers.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device).eval()
    return tokenizer, model, device


def embed_sentences(sentences: Sequence[str], tokenizer: Any, model: Any, device: str, batch: int = 32) -> Any:
    import numpy as np
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            enc = tokenizer(
                list(sentences[i : i + batch]),
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            vecs = model(**enc).last_hidden_state[:, 0, :]
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            out.append(vecs.cpu().numpy())
    return np.vstack(out)


def pack_units_by_length(units: Sequence[str], max_chars: int, min_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    for unit in units:
        unit = collapse_ws(unit)
        if not unit:
            continue
        candidate = " ".join([*current, unit]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [unit]
        else:
            current.append(unit)
    if current:
        chunks.append(" ".join(current).strip())

    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)
    return merged


def semantic_split_units(
    units: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_chars: int,
    min_chars: int,
    threshold_pct: int,
) -> List[str]:
    import numpy as np

    units = [collapse_ws(unit) for unit in units if collapse_ws(unit)]
    if len(" ".join(units)) <= max_chars or len(units) <= 1:
        return [" ".join(units).strip()] if units else []

    emb = embed_sentences(units, tokenizer, model, device)
    sims = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
    cutoff = np.percentile(sims, threshold_pct)
    chunks: List[str] = []
    current = [units[0]]

    for i, sim in enumerate(sims):
        candidate = " ".join([*current, units[i + 1]])
        if (sim < cutoff and len(" ".join(current)) >= min_chars) or len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [units[i + 1]]
        else:
            current.append(units[i + 1])
    if current:
        chunks.append(" ".join(current).strip())

    return pack_units_by_length(chunks, max_chars=max_chars, min_chars=min_chars)


def split_non_recommendation_section(
    doc: Document,
    section: Section,
    consumed_indexes: Iterable[int],
    config: ChunkConfig,
    start_chunk_index: int,
    semantic_model: Optional[Tuple[Any, Any, str]] = None,
) -> List[Document]:
    units = section_units(section)
    consumed = set(consumed_indexes)
    residual_units = [unit for idx, unit in enumerate(units) if idx not in consumed]
    if not residual_units:
        return []

    parent_context = build_parent_context(doc, section, config.parent_max_chars)
    text = " ".join(residual_units)
    use_semantic = semantic_model is not None and (
        BACKGROUND_SECTION_RE.search(section.title) or len(text) > config.max_chars
    )
    if use_semantic:
        tokenizer, model, device = semantic_model
        texts = semantic_split_units(
            residual_units,
            tokenizer,
            model,
            device,
            max_chars=config.max_chars,
            min_chars=config.min_chars,
            threshold_pct=config.semantic_threshold_pct,
        )
    else:
        texts = pack_units_by_length(residual_units, config.max_chars, config.min_chars)

    chunks: List[Document] = []
    chunk_index = start_chunk_index
    default_type = "section_parent_chunk" if RECOMMENDATION_SECTION_RE.search(section.title) else "section_chunk"
    for text in texts:
        if is_low_value_chunk(text, config.min_chars):
            continue
        chunks.append(make_chunk(doc, section, text, chunk_index, default_type, parent_context))
        chunk_index += 1
    return chunks


def structured_recommendation_chunks(
    doc: Document,
    config: ChunkConfig,
    start_chunk_index: int,
) -> List[Document]:
    raw_recommendations = doc.metadata.get("recommendation")
    if not isinstance(raw_recommendations, list):
        return []

    section = Section(
        title="Structured Recommendations",
        text="",
        level=1,
        ordinal=-1,
        parent_titles=[str(doc.metadata.get("title") or "Untitled guideline")],
    )
    parent_context = build_parent_context(doc, section, config.parent_max_chars)
    chunks: List[Document] = []
    chunk_index = start_chunk_index
    for item in raw_recommendations:
        if isinstance(item, dict):
            parts = [
                item.get("text") or item.get("recommendation") or item.get("statement") or "",
                item.get("strength") or item.get("recommendation_strength") or "",
                item.get("quality") or item.get("evidence_quality") or item.get("evidence") or "",
                item.get("remarks") or item.get("note") or "",
            ]
            text = ". ".join(collapse_ws(part) for part in parts if collapse_ws(part))
        else:
            text = collapse_ws(str(item))
        if is_low_value_chunk(text, config.min_chars):
            continue
        chunks.append(
            make_chunk(
                doc,
                section,
                text,
                chunk_index,
                "recommendation_block",
                parent_context,
                {"recommendation_atomic": True, "structured_recommendation": True},
            )
        )
        chunk_index += 1
    return chunks


def splitting(
    data: Sequence[Document],
    config: ChunkConfig,
    semantic_model: Optional[Tuple[Any, Any, str]] = None,
) -> List[Document]:
    all_chunks: List[Document] = []
    seen_chunk_ids: set[str] = set()

    for doc in data:
        for chunk in split_document(doc, config, semantic_model=semantic_model):
            chunk_id = str(chunk.metadata.get("chunk_id") or "")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            all_chunks.append(chunk)

    for index, chunk in enumerate(all_chunks):
        chunk.metadata["global_chunk_index"] = index
    return all_chunks


def split_document(
    doc: Document,
    config: ChunkConfig,
    semantic_model: Optional[Tuple[Any, Any, str]] = None,
    sections: Optional[Sequence[Section]] = None,
) -> List[Document]:
    doc_chunks: List[Document] = structured_recommendation_chunks(doc, config, start_chunk_index=0)
    next_index = len(doc_chunks)

    for section in sections if sections is not None else parse_sections(doc):
        rec_chunks, consumed = extract_recommendation_blocks(doc, section, config, next_index)
        doc_chunks.extend(rec_chunks)
        next_index += len(rec_chunks)

        section_chunks = split_non_recommendation_section(
            doc,
            section,
            consumed,
            config,
            next_index,
            semantic_model=semantic_model,
        )
        doc_chunks.extend(section_chunks)
        next_index += len(section_chunks)

    return doc_chunks


def save_jsonl(result: Sequence[Document], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for doc in result:
            f.write(
                json.dumps(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Structure-first medical guideline chunker. Recommendation blocks are kept atomic; "
            "MedCPT semantic splitting is only used inside long non-recommendation sections."
        )
    )
    parser.add_argument("--input", default="data/raw/data.jsonl", help="Input guideline JSONL file")
    parser.add_argument("--output", default="./outputs/chunk_v2.jsonl", help="Output chunk JSONL file")
    parser.add_argument("--min-chars", type=int, default=50, help="Drop chunks shorter than this many chars")
    parser.add_argument("--max-chars", type=int, default=1400, help="Soft max chars per child chunk")
    parser.add_argument("--parent-max-chars", type=int, default=2800, help="Max chars stored as parent_context")
    parser.add_argument(
        "--semantic",
        choices=["auto", "always", "never"],
        default="auto",
        help="Use MedCPT only inside long/background sections by default",
    )
    return parser.parse_args()


def should_load_semantic_model(documents: Sequence[Document], config: ChunkConfig, mode: str) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    for doc in documents:
        for section in parse_sections(doc):
            if BACKGROUND_SECTION_RE.search(section.title) and len(collapse_ws(section.text)) > config.max_chars:
                return True
            if len(collapse_ws(section.text)) > config.max_chars * 2:
                return True
    return False


def section_needs_semantic(section: Section, config: ChunkConfig) -> bool:
    text_len = len(collapse_ws(section.text))
    return bool(BACKGROUND_SECTION_RE.search(section.title) and text_len > config.max_chars) or text_len > config.max_chars * 2


def main() -> None:
    args = parse_args()
    config = ChunkConfig(
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        parent_max_chars=args.parent_max_chars,
    )

    semantic_model: Optional[Tuple[Any, Any, str]] = None
    if args.semantic == "always":
        semantic_model = load_model()

    cleaner = DocumentCleaner()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_docs: set[str] = set()
    seen_chunk_ids: set[str] = set()
    total_chunks = 0
    total_records = 0

    with output.open("w", encoding="utf-8") as f:
        for sample in iter_jsonl(args.input):
            total_records += 1
            cleaned_docs = cleaner.clean([process_record(sample)])
            if not cleaned_docs:
                continue
            doc = cleaned_docs[0]

            doc_hash = hash_text(doc.page_content)
            if doc_hash in seen_docs:
                continue
            seen_docs.add(doc_hash)

            sections = parse_sections(doc)
            if args.semantic == "auto" and semantic_model is None:
                if any(section_needs_semantic(section, config) for section in sections):
                    semantic_model = load_model()

            for chunk in split_document(doc, config, semantic_model=semantic_model, sections=sections):
                chunk_id = str(chunk.metadata.get("chunk_id") or "")
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunk.metadata["global_chunk_index"] = total_chunks
                f.write(
                    json.dumps(
                        {
                            "page_content": chunk.page_content,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_chunks += 1

            if total_records % 1000 == 0:
                print(f"Processed {total_records} records, wrote {total_chunks} chunks")

    print(f"Wrote {total_chunks} chunks to {args.output}")


if __name__ == "__main__":
    main()

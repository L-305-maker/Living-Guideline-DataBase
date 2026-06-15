import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# 读取文件 ###############################################
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Line {line_no} JSON parse error: {exc}")
    return records
###########################################################


# 将数据初步处理为Document形式 ##############################
def keep_year(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"(19|20)\d{2}", str(value).strip())
    return match.group(0) if match else ""


def process_data(data: List[Dict[str, Any]]) -> List[Document]:
    result = []
    for sample in data:
        result.append(
            Document(
                page_content=sample.get("content") or "",
                metadata={
                    "published_date": keep_year(sample.get("published_date")),
                    "title": sample.get("title") or "",
                    "medical_topic": sample.get("medical_topics") or "",
                    "url": sample.get("url") or "",
                    "source": sample.get("source") or ""
                },
            )
        )
    return result
###########################################################


# 数据清洗 #################################################
class DocumentCleaner:
    MOJIBAKE_REPLACEMENTS = {
        "\u920d?": ">=",
        "\u920d\ufffd": ">=",
        "\u920d\u6a9a": "'s",
        "\u920d\u6a9b": "'t",
        "\u920d\u6a99": "'r",
        "\u920d\u6a9d": "'v",
        "\u920d\u6a91": "'l",
        "\u920d\uff1f": "'",
        "\u76f2": "a",
        "\ufffd": "",
    }

    def clean_text(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        for bad, good in self.MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)

        text = re.sub(r"(?im)^\s*#+\s*", "", text)
        text = re.sub(r"(?im)^\s*[-*•]\s+", "", text)
        text = re.sub(r"(?im)^\s*(Path|Citation|Footnotes?|References?)\s*$", "", text)
        text = re.sub(r"(?im)^\s*(\[\d+\]|\(\w\)|[a-z]|\d+)\s*$", "", text)
        text = re.sub(r"(?i)\bCitation\s+", "", text)
        text = re.sub(r"\s+\[\d+\]\s+", " ", text)
        text = re.sub(r"\s+\([a-z]\)\s+", " ", text)
        text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]])", r"\1", text)
        return text.strip()

    def clean(self, documents: List[Document]) -> List[Document]:
        cleaned_docs = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if len(cleaned_text) < 20:
                continue
            doc.page_content = cleaned_text
            cleaned_docs.append(doc)
        return cleaned_docs
###########################################################


# 指南去重 #################################################
def hash_text(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicator(data: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for document in data:
        content_hash = hash_text(document.page_content)
        if content_hash in seen:
            continue
        seen.add(content_hash)
        unique.append(document)
    return unique
###########################################################


def load_model():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("transformers not installed")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device).eval()
    return tokenizer, model, device


# 将数据切分为chunk，并对chunk进行检查 #####################
@torch.no_grad()
def embed_sentences(sentences, tokenizer, model, device, batch=32):
    out = []
    for i in range(0, len(sentences), batch):
        enc = tokenizer(
            sentences[i : i + batch],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        vecs = model(**enc).last_hidden_state[:, 0, :]
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        out.append(vecs.cpu().numpy())
    return np.vstack(out)


def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [sent.strip() for sent in sents if sent.strip()]


# 将太长的chunk进行二次切分 ##################################
def split_toolong(text: str) -> List[str]:
    units = split_sentences(text)
    if len(units) > 1 and max(len(unit) for unit in units) <= 1200:
        return units

    units = re.split(
        r"(?<=[.!?;:])\s+|"
        r"(?=\b(?:Children|Adults|Neonates|First-line|Second-line|Alternative|"
        r"Treatment|Clinical features|Diagnosis|Management|Prevention|"
        r"No associated|Associated)\b)",
        text,
    )
    return [unit.strip() for unit in units if unit.strip()]


# 清洗chunk包含的URL、参考文献等内容 ##########################
def is_noise(text: str) -> bool:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return True

    lower = normalized.lower()
    urls = re.findall(r"https?://\S+|www\.\S+|doi\.org/\S+", normalized, flags=re.I)
    doi_refs = re.findall(r"\bdoi\s*:?\s*10\.\d{4,9}/\S+|10\.\d{4,9}/\S+", normalized, flags=re.I)
    citation_markers = re.findall(
        r"\b(accessed|published|journal|vol\.|volume|issue|et al\.|"
        r"doi|pmid|pmcid|copyright|available from|retrieved|"
        r"press|publisher|database|supplement|appendix)\b",
        lower,
    )
    journal_markers = re.findall(
        r"\b(jama|lancet|nejm|bmj|cochrane|pediatrics|epilepsia|"
        r"clin infect dis|intensive care med|n engl j med|mmwr|"
        r"world j|crit care|paediatrics|pediatric|journal)\b",
        lower,
    )
    years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
    numbered_refs = re.findall(r"(?:^|\s)\d+\.\s+[A-Z][A-Za-z-]+", normalized)
    word_count = len(re.findall(r"[A-Za-z][A-Za-z-]*", normalized))
    url_chars = sum(len(url) for url in urls)
    url_ratio = url_chars / max(len(normalized), 1)

    has_clinical_signal = bool(
        re.search(
            r"\b(administer|treat|treatment|diagnos|monitor|refer|manage|"
            r"recommend|should|dose|dosage|mg|ml|kg|contraindicat|symptom|"
            r"patient|patients|children|adults|neonates|infection|disease|"
            r"severe|therapy|clinical features|management|prevention|"
            r"first-line|second-line|alternative|oral|iv|im|po|signs|"
            r"blood pressure|fever|pain|antibiotic|vaccine)\b",
            lower,
        )
    )

    has_action_signal = bool(
        re.search(
            r"\b(administer|give|start|stop|avoid|continue|monitor|perform|"
            r"transfer|refer|use|apply|insert|remove|repeat|check|assess|"
            r"reassess|treat|manage)\b",
            lower,
        )
    )

    reference_score = 0
    reference_score += 2 * len(urls)
    reference_score += 2 * len(doi_refs)
    reference_score += len(citation_markers)
    reference_score += len(journal_markers)
    reference_score += len(numbered_refs)
    reference_score += 1 if years else 0

    if re.fullmatch(r"(?:\d+\.\s*)?(?:https?://\S+|www\.\S+|doi\.org/\S+)", normalized, flags=re.I):
        return True

    if url_ratio > 0.45 and not has_clinical_signal:
        return True

    if (urls or doi_refs) and word_count < 22 and not has_clinical_signal:
        return True

    if reference_score >= 6 and not has_clinical_signal:
        return True

    if reference_score >= 8 and not has_action_signal:
        return True

    if len(numbered_refs) >= 2 and (urls or doi_refs or journal_markers) and not has_action_signal:
        return True

    if re.match(r"^\d+\.\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*,?\s+[A-Z]", normalized):
        if reference_score >= 4 and not has_action_signal:
            return True

    return False


def is_low_value_chunk(text: str, min_chars: int = 50) -> bool:
    normalized = " ".join((text or "").split()).strip()
    if len(normalized) < min_chars:
        return True
    if is_noise(normalized):
        return True

    lower = normalized.lower().strip(" .:;,-")
    if lower in {"path", "references", "references 1", "citation"}:
        return True
    if re.fullmatch(r"(references?\s*)?\d+\.?", lower):
        return True
    if re.fullmatch(r"[a-z]\.?", lower):
        return True

    return False

 
def trim_reference_tail(text: str, min_remaining_chars: int = 80) -> str:
    normalized = " ".join((text or "").split()).strip()
    if len(normalized) < min_remaining_chars:
        return normalized

    tail_patterns = [
        r"\s+\d+\.\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*,?\s+[A-Z][^.]{0,220}"
        r"(?:et al\.|https?://|doi\.org|10\.\d{4,9}/|Accessed|Published|Journal|"
        r"JAMA|Lancet|BMJ|Cochrane|MMWR|N Engl J Med|Clin Infect Dis).*$",
        r"\s+(?:References?|Bibliography)\s+\d+\..*$",
        r"\s+(?:Accessed|Published online|Available from|Retrieved)\s+[^.]{0,180}"
        r"(?:https?://|doi\.org|10\.\d{4,9}/).*$",
    ]

    clinical_tail_guard = re.compile(
        r"\b(administer|give|start|monitor|treat|dose|mg|ml|kg|children|adults|"
        r"patient|patients|diagnos|management|severe|infection|therapy|iv|im|po)\b",
        re.I,
    )

    for pattern in tail_patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if not match:
            continue

        head = normalized[: match.start()].strip()
        tail = normalized[match.start() :].strip()
        if len(head) < min_remaining_chars:
            continue
        if clinical_tail_guard.search(tail) and len(tail) > 220:
            continue
        return head.rstrip(" ,;:")

    return normalized


def trim_reference_prefix(text: str, min_remaining_chars: int = 80) -> str:
    normalized = " ".join((text or "").split()).strip()
    if len(normalized) < min_remaining_chars:
        return normalized

    clinical_signal = re.compile(
        r"\b(administer|give|start|monitor|treat|dose|mg|ml|kg|children|adults|"
        r"patient|patients|diagnos|clinical features|management|prevention|"
        r"severe|infection|therapy|first-line|second-line|alternative|signs|"
        r"symptoms|fever|pain|po|iv|im|antibiotic|penicillin|benzathine|"
        r"ceftriaxone|azithromycin|erythromycin|tetracycline|vaccine|"
        r"available|choice|suspected)\b",
        re.I,
    )

    starts_like_reference = re.match(
        r"^(https?://|www\.|doi\.org/|10\.\d{4,9}/|Published|Accessed|"
        r"Available from|Retrieved|[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*,)",
        normalized,
        flags=re.I,
    )
    if not starts_like_reference:
        return normalized

    url_match = re.search(r"https?://\S+|www\.\S+|doi\.org/\S+|10\.\d{4,9}/\S+", normalized[:320], flags=re.I)
    if not url_match:
        return normalized

    cut = url_match.end()
    accessed = re.match(r"\s+\[?Accessed[^\]]*\]?", normalized[cut:], flags=re.I)
    if accessed:
        cut += accessed.end()

    while cut < len(normalized) and normalized[cut] in " .;:,":
        cut += 1

    rest = normalized[cut:].strip(" ,;:")
    if len(rest) >= min_remaining_chars and clinical_signal.search(rest):
        return rest

    return normalized


def split_long_similarity(text: str,tokenizer,model,device,max_chars: int = 1200,min_chars: int = 50,) -> List[str]:
    
    if len(text) <= max_chars:
        return [text]

    sents = split_toolong(text)
    if len(sents) <= 1:
        return [text]

    emb = embed_sentences(sents, tokenizer, model, device)
    sims = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]

    def join_range(start: int, end: int) -> str:
        return " ".join(sents[start:end]).strip()

    def split_range(start: int, end: int) -> List[str]:
        current = join_range(start, end)
        if len(current) <= max_chars or end - start <= 1:
            return [current]

        candidates = []
        for split_at in range(start + 1, end):
            left = join_range(start, split_at)
            right = join_range(split_at, end)
            if len(left) >= min_chars and len(right) >= min_chars:
                candidates.append(split_at)

        if not candidates:
            candidates = list(range(start + 1, end))

        split_at = min(candidates, key=lambda idx: sims[idx - 1])
        return split_range(start, split_at) + split_range(split_at, end)

    return split_range(0, len(sents))


def splitting(data: List[Document],tokenizer,model,device,threshold_pct: int = 20,max_chars: int = 1200,min_chars: int = 50,) -> List[Document]:
    all_chunks: List[Document] = []
    for doc in data:
        content = doc.page_content or ""
        sents = split_sentences(content)

        if len(sents) <= 1:
            if content and not is_low_value_chunk(content, min_chars):
                all_chunks.append(
                    Document(
                        page_content=content,
                        metadata={**doc.metadata, "chunk_index": 0, "char_len": len(content)},
                    )
                )
            continue

        emb = embed_sentences(sents, tokenizer, model, device)
        sims = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
        cutoff = np.percentile(sims, threshold_pct)
        chunks, cur = [], [sents[0]]

        for i, sim in enumerate(sims):
            too_long = len(" ".join(cur)) > max_chars
            if sim < cutoff or too_long:
                chunks.append(" ".join(cur))
                cur = [sents[i + 1]]
            else:
                cur.append(sents[i + 1])

        if cur:
            chunks.append(" ".join(cur))

        accepted_idx = 0
        for text in chunks:
            text = text.strip()
            for sub_text in split_long_similarity(text, tokenizer, model, device, max_chars, min_chars):
                sub_text = trim_reference_tail(sub_text)
                sub_text = trim_reference_prefix(sub_text)
                if is_low_value_chunk(sub_text, min_chars):
                    continue
                all_chunks.append(
                    Document(
                        page_content=sub_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": accepted_idx,
                            "char_len": len(sub_text),
                        },
                    )
                )
                accepted_idx += 1

    return all_chunks


def save_jsonl(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for doc in result:
            item = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/asa_guidelines.jsonl", help="choose the input file")
    parser.add_argument("--output", default="./outputs/temp_result.jsonl", help="choose the output file")
    parser.add_argument("--min-chars", type=int, default=50, help="drop chunks shorter than this many characters")
    args = parser.parse_args()

    data = process_data(read_jsonl(args.input))
    cleaner = DocumentCleaner()
    data = cleaner.clean(data)
    data = deduplicator(data)

    tokenizer, model, device = load_model()
    result = splitting(data, tokenizer, model, device, min_chars=args.min_chars)

    save_jsonl(result, args.output)


if __name__ == "__main__":
    main()

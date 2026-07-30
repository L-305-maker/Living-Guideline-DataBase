"""Controlled text normalization for evidence span checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedTextView:
    original_text: str
    normalized_text: str
    normalized_to_original_offsets: list[int]
    normalization_steps: list[str]


def normalize_for_span_match(text: str) -> NormalizedTextView:
    steps = ["unicode_nfkc", "strip_markdown_emphasis", "repair_hyphen_linebreak", "collapse_whitespace"]
    original = text or ""
    normalized_chars: list[str] = []
    offsets: list[int] = []
    index = 0
    while index < len(original):
        char = original[index]
        if char in "*_`":
            index += 1
            continue
        if char == "-" and index + 1 < len(original) and original[index + 1] in "\r\n":
            index += 1
            while index < len(original) and original[index] in "\r\n":
                index += 1
            continue
        if char.isspace():
            if normalized_chars and normalized_chars[-1] != " ":
                normalized_chars.append(" ")
                offsets.append(index)
            index += 1
            continue
        norm = unicodedata.normalize("NFKC", char)
        for norm_char in norm:
            normalized_chars.append(norm_char)
            offsets.append(index)
        index += 1
    normalized = "".join(normalized_chars).strip()
    if normalized != "".join(normalized_chars):
        normalized = re.sub(r"^\s+|\s+$", "", normalized)
    return NormalizedTextView(original, normalized, offsets[: len(normalized)], steps)


def find_unique(text: str, quote: str) -> tuple[int, int] | None:
    if not quote:
        return None
    positions = []
    start = text.find(quote)
    while start >= 0:
        positions.append((start, start + len(quote)))
        start = text.find(quote, start + 1)
    return positions[0] if len(positions) == 1 else None


def normalized_unique_match(text: str, quote: str) -> tuple[int, int, list[str]] | None:
    source = normalize_for_span_match(text)
    target = normalize_for_span_match(quote)
    span = find_unique(source.normalized_text, target.normalized_text)
    if span is None or not source.normalized_to_original_offsets:
        return None
    start_norm, end_norm = span
    if end_norm - 1 >= len(source.normalized_to_original_offsets):
        return None
    start = source.normalized_to_original_offsets[start_norm]
    end = source.normalized_to_original_offsets[end_norm - 1] + 1
    return start, end, source.normalization_steps

# 文档级表征（document card / views）构建与触发。
#
# 主要符号：
# - build_document_card：单篇文档的"卡片"（多字段聚合）；
# - build_document_views：从 card 派生 7 类 view（title_abstract / scope_population ...）；
# - build_document_representations：批量入口（扫 markdown_clean → 写 cards + views）；
# - classify_section / classify_chunk / chunk_type_weight：分类器与权重。
"""Document-level representations for guideline search."""

from src.retrieval.document_repr.builder import (
    build_document_card,
    build_document_representations,
    build_document_views,
)
from src.retrieval.document_repr.section_classifier import classify_chunk, classify_section, chunk_type_weight

__all__ = [
    "build_document_card",
    "build_document_representations",
    "build_document_views",
    "classify_chunk",
    "classify_section",
    "chunk_type_weight",
]
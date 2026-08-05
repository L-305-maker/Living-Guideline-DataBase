# 面向临床指南检索的标题感知 Markdown chunking。
#
# 入口与构造：build_atomic_chunks / build_chunk_links / parse_markdown_document / build_section_chunks /
# build_table_chunks / build_structural_chunks / expand_query / GuidelineRAGIndex。
"""Heading-aware Markdown chunking for clinical guideline retrieval."""

from src.guideline_chunking.atomic_chunker import build_atomic_chunks
from src.guideline_chunking.chunk_linker import build_chunk_links
from src.guideline_chunking.markdown_parser import parse_markdown_document
from src.guideline_chunking.section_chunker import build_section_chunks
from src.guideline_chunking.structural_rag import GuidelineRAGIndex, build_structural_chunks, expand_query
from src.guideline_chunking.table_chunker import build_table_chunks

__all__ = [
    "GuidelineRAGIndex",
    "build_atomic_chunks",
    "build_chunk_links",
    "build_section_chunks",
    "build_structural_chunks",
    "build_table_chunks",
    "expand_query",
    "parse_markdown_document",
]
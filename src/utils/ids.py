# 稳定 ID 与哈希工具。
#
# 设计原则：
# - sha256_file / sha256_text：跨阶段内容指纹，用于文档去重与 doc_id 生成；
# - slug：把任意字符串转成文件名安全的小写短串（保留 - _ ，去除重音符号）；
# - make_doc_id / make_chunk_id：稳定的对外可见主键，跨 JSONL / PG / MCP 全链路一致。
"""Stable IDs and hashing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """大文件分块读取并计算 SHA-256，避免一次性读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """对字符串直接计算 SHA-256（UTF-8 编码）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(value: object | None, max_len: int = 80) -> str:
    """把任意字符串转成文件名安全的小写短串。

    处理：
    - NFKD 规范化后过滤组合标记（去重音符号）；
    - 仅保留字母数字、下划线、连字符、空格；
    - 空格与连字符合并为 _，连续 _ 合并；
    - 截断 max_len（默认 80）并去除尾部 _。
    返回值保证非空：空输入返回 "unknown"。
    """
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).strip().lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_len].strip("_") or "unknown")


def make_doc_id(source_institution: str, publication_date: str, file_sha256: str) -> str:
    """生成跨阶段稳定的 doc_id：<机构slug>_<年份>_<sha256 前 12 位>。

    年份取自 publication_date 的正则匹配（19xx/20xx），缺失时回退 "unknown"。
    文件指纹前缀 12 位足够避免碰撞且保持 doc_id 简洁。
    """
    year_match = re.search(r"(19\d{2}|20\d{2})", publication_date or "")
    year = year_match.group(1) if year_match else "unknown"
    return f"{slug(source_institution)}_{year}_{file_sha256[:12]}"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """生成 chunk_id：<doc_id>#chunk_<5 位序号>。"""
    return f"{doc_id}#chunk_{chunk_index:05d}"
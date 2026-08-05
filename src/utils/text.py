# 跨处理阶段共享的纯文本规范化工具。
#
# 模块职责：
# - SEARCH_TOKEN_RE：英文按单词切、中文按字符切（简化 token 估算）；
# - tokenize_search_text / estimate_tokens：BM25 与 token 估算共用规则；
# - normalize_space：压缩连续空白。
"""跨处理阶段共享的纯文本规范化工具。"""

from __future__ import annotations

import re


# 搜索 token 正则：英文按 [A-Za-z0-9]+（含连字符/撇号分词），中文按单字。
# 该规则同时用于 BM25 分词与 token 数估算，保持口径一致。
SEARCH_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]"
)


def tokenize_search_text(text: str) -> list[str]:
    """按 BM25 与粗略 token 估算共用的规则切分中英文文本。"""
    return [token.lower() for token in SEARCH_TOKEN_RE.findall(text or "")]


def estimate_tokens(text: str) -> int:
    """使用稳定的轻量规则估算 token 数，不触发模型加载。

    实现：直接复用 SEARCH_TOKEN_RE.findall 计数，避免引入重型分词器。
    """
    return len(SEARCH_TOKEN_RE.findall(text or ""))


def normalize_space(text: str) -> str:
    """把连续空白压缩为单个空格并清理首尾空白。"""
    return re.sub(r"\s+", " ", text or "").strip()
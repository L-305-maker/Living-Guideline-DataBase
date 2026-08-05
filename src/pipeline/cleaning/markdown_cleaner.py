# evidence 文档的 Markdown 清洗入口。
#
# 仅做一层 alias 转发：真正的实现位于 cleaner.clean_all / clean_file，
# 这里只负责暴露 src.pipeline.cleaning 统一接口。
"""Markdown cleanup entry points for evidence documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.cleaner import clean_all as _clean_all
from src.pipeline.cleaning.cleaner import clean_file, clean_markdown_text


def clean_markdown_dir(
    markdown_raw_dir: str | Path,
    markdown_clean_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """批量清洗 markdown_raw/ → markdown_clean/，并把 manifest 写到 manifest_path。

    返回 _clean_all 的 manifest 字典（行数、按清洗质量的分布等）。
    """
    return _clean_all(markdown_raw_dir, markdown_clean_dir, manifest_path)


# 兼容旧调用方：clean_all 与 clean_markdown_dir 等价。
clean_all = clean_markdown_dir
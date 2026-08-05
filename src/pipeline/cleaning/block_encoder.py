# 把清洗后的 Markdown 编码为完整源块（heading-aware 段落）。
#
# 注意：磁盘上的目录名仍然是 sections/，因为 src.storage 入库和 PostgreSQL 表
# （sections）已经按这个名称固定；改名字段会破坏下游契约。
"""Block encoding for clean Markdown evidence documents.

In the evidence pipeline, a block is the complete Markdown section produced
from heading-aware parsing. The on-disk directory remains named ``sections`` so
it can be ingested by the PostgreSQL storage builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.cleaning.encoder import encode_all as _encode_all
from src.pipeline.cleaning.encoder import encode_file
from src.pipeline.cleaning.encoder import encode_markdown


def encode_blocks(
    markdown_clean_dir: str | Path,
    blocks_dir: str | Path,
) -> dict[str, Any]:
    """编码所有 clean Markdown 为完整源块（sections），写到 blocks_dir。

    blocks_dir 在磁盘上即为 sections/；函数返回 {"documents", "blocks", "blocks_dir"}。
    注意这里用 "blocks" 而不是 "sections" 是因为函数语义是"块编码"，让上游脚本
    区分 blocks_dir 与下游 chunk_blocks 写入的 chunks_dir。
    """
    result = _encode_all(markdown_clean_dir, blocks_dir)
    return {
        "documents": result.get("documents", 0),
        "blocks": result.get("sections", 0),
        "blocks_dir": str(blocks_dir),
    }


# 旧名兼容：encode_all 等价于 encode_blocks。
encode_all = encode_blocks
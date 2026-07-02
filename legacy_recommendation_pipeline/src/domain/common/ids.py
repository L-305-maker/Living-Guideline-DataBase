"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import hashlib
from typing import Any


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


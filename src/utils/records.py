# 跨存储后端共享的记录字段规范化工具。
#
# 模块职责：
# - publication_year：从日期字符串提取 4 位出版年份（兼容多种日期格式）；
# - truthy：把 JSON / SQLite / PostgreSQL 中的布尔表示统一解析；
# - department_text：把 clinical_departments list 规范为去重后 '|' 分隔字符串；
# - record_text：兼容旧版只有 content 字段的检索记录。
"""跨存储后端共享的记录字段规范化工具。"""

from __future__ import annotations

import re
from typing import Any


def publication_year(publication_date: str | None) -> int | None:
    """从日期或日期样式文本中提取四位出版年份。"""
    match = re.search(r"(19\d{2}|20\d{2})", publication_date or "")
    return int(match.group(1)) if match else None


def truthy(value: Any) -> bool:
    """统一解析 JSON、SQLite 与 PostgreSQL 中常见的布尔表示。

    - bool True：直接返回 True；
    - str.strip().lower() ∈ {"1","true","yes","y"}：返回 True；
    - 其它（含 None、空串、False）：返回 False。
    """
    return value is True or str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def department_text(record: dict[str, Any]) -> str:
    """把单科室或多科室字段规范为去重后的竖线分隔文本。

    输入优先级：
    1. clinical_departments（list）
    2. clinical_department（str，list 缺失时回退）
    3. 兜底 "未分类"

    返回非空字符串；dict.fromkeys 保证顺序稳定去重。
    """
    labels = record.get("clinical_departments") or [
        record.get("clinical_department") or "未分类"
    ]
    return (
        "|".join(dict.fromkeys(str(label) for label in labels if label))
        or "未分类"
    )


def record_text(record: dict[str, Any], text_field: str) -> str:
    """读取指定检索字段，并兼容只有 content 的旧记录。"""
    return str(record.get(text_field) or record.get("content") or "")
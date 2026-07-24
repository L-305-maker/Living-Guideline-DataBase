"""跨存储后端共享的记录字段规范化工具。"""

from __future__ import annotations

import re
from typing import Any


def publication_year(publication_date: str | None) -> int | None:
    """从日期或日期样式文本中提取四位出版年份。"""

    match = re.search(r"(19\d{2}|20\d{2})", publication_date or "")
    return int(match.group(1)) if match else None


def truthy(value: Any) -> bool:
    """统一解析 JSON、SQLite 与 PostgreSQL 中常见的布尔表示。"""

    return value is True or str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def department_text(record: dict[str, Any]) -> str:
    """把单科室或多科室字段规范为去重后的竖线分隔文本。"""

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

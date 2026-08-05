# 简易 YAML-like front matter 解析器与写入器。
#
# 设计：仅支持扁平 key-value 的单行 front matter（与仓库现有 Markdown 文档约定匹配）；
# 不引入 PyYAML 依赖，避免把重型库拖入冷启动路径。
#
# 写入：每个字段占一行，字符串加双引号包裹（值含双引号时用反斜杠转义）；
# 读取：容忍缺失末尾换行、空行、注释，list/dict 值用 json.loads 反序列化。
"""Tiny YAML-like front matter parser/writer for flat metadata."""

from __future__ import annotations

import json
from collections import OrderedDict


# 写入时的字段顺序：保持稳定输出，方便 audit/版本对比。
FRONT_MATTER_KEYS = [
    "id",
    "title",
    "publication_date",
    "source_institution",
    "source_file",
    "clinical_department",
    "clinical_departments",
    "department_scope",
    "document_kind",
    "source_pdf_text_quality",
    "source_pdf_needs_ocr",
    "source_pdf_is_scanned",
    "pdf_page_count",
    "pdf_text_chars",
    "pdf_text_chars_per_page",
    "pdf_low_text_page_ratio",
    "pdf_image_page_ratio",
    "pdf_scanned_page_ratio",
    "pdf_is_scanned",
    "pdf_needs_ocr",
    "pdf_text_quality",
    "ocr_engine",
    "ocr_applied",
    "ocr_status",
    "ocr_error",
    "cleaning_quality",
    "cleaning_flags",
]


def dump_front_matter(metadata: dict[str, object], body: str) -> str:
    """按 FRONT_MATTER_KEYS 顺序写出 --- 包围的 front matter + body。

    list/dict 值用 json.dumps 序列化（ensure_ascii=False 保留中文）；
    字符串值加双引号包裹，内部双引号用反斜杠转义；
    body 在最末尾补 \\n，保证后续 append 行可正常拼到末尾。
    """
    lines = ["---"]
    for key in FRONT_MATTER_KEYS:
        raw_value = metadata.get(key, "")
        value = (
            json.dumps(raw_value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(raw_value, (list, dict))
            else str(raw_value or "")
        )
        value = value.replace('"', '\\"')
        lines.append(f'{key}: "{value}"')
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.strip() + "\n"


def parse_front_matter(markdown: str) -> tuple[dict[str, object], str]:
    """解析 --- 包围的 front matter，返回 (metadata, body)。

    容错处理：
    - 文件不以 ---\\n 起头 → 返回 ({}, 全文)；
    - 找不到闭合 --- → 返回 ({}, 全文)；
    - 字段值以 [ / { 开头 → 尝试 json.loads（list / dict）；
    - 解析失败的字段原样保留为字符串。
    """
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end].strip().splitlines()
    metadata: dict[str, object] = OrderedDict()
    for line in raw:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace('\\"', '"')
        if value.startswith(("[", "{")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        metadata[key.strip()] = value
    body = text[text.find("\n", end + 1) + 1 :]
    return metadata, body
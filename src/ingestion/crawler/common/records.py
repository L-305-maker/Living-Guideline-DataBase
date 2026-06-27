"""爬虫公共工具文件：封装 HTTP、JSONL、PDF 和文本处理等 ingestion 阶段共享能力。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PdfCandidate:
    source: str
    url: str
    title: str = ""
    published_year: str = ""
    landing_url: str = ""


@dataclass
class PdfMetadata:
    source: str
    url: str
    title: str
    published_year: str
    pdf_path: str
    landing_url: str = ""
    sha256: str = ""

    def to_json(self) -> dict[str, Any]:
        item = asdict(self)
        item["pdf_path"] = str(Path(self.pdf_path))
        return item



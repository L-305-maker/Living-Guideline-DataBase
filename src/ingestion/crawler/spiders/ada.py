"""爬虫来源适配文件：面向特定医学资料来源抓取或解析原始记录，并为 ingestion 阶段提供统一输入。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from src.ingestion.crawler.common.records import PdfCandidate
from src.ingestion.crawler.spiders.base import BaseSpider


class AdaSpider(BaseSpider):
    source = "ada"
    allowed_domains = ("diabetesjournals.org", "diabetes.org")
    start_urls = (
        "https://diabetesjournals.org/care/issue",
        "https://diabetesjournals.org/care/pages/standards-of-care",
        "https://professional.diabetes.org/standards-of-care",
    )
    max_pages = 250

    def make_candidate(self, pdf_url: str, link_text: str, landing_url: str, page_meta: dict[str, str]) -> PdfCandidate | None:
        candidate = super().make_candidate(pdf_url, link_text, landing_url, page_meta)
        if candidate:
            return candidate
        haystack = f"{link_text} {landing_url} {pdf_url}".lower()
        if "standards-of-care" in haystack or "standards of care" in haystack:
            return PdfCandidate(
                source=self.source,
                url=pdf_url,
                title=link_text or page_meta.get("title", "ADA Standards of Care"),
                published_year=page_meta.get("published_year", ""),
                landing_url=landing_url,
            )
        return None



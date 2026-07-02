"""爬虫来源适配文件：面向特定医学资料来源抓取或解析原始记录，并为 ingestion 阶段提供统一输入。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable
from urllib.parse import urlparse

from src.ingestion.crawler.common.http import HttpClient, PdfDownloader
from src.ingestion.crawler.common.records import PdfCandidate, PdfMetadata
from src.ingestion.crawler.common.text import (
    absolute_url,
    extract_links,
    extract_page_metadata,
    extract_year,
    normalize_space,
    same_domain_or_subdomain,
)


logger = logging.getLogger(__name__)


GUIDELINE_KEYWORDS = (
    "guideline",
    "guidelines",
    "recommendation",
    "recommendations",
    "clinical practice",
    "practice guideline",
    "standards of care",
    "statement",
)
EXCLUDE_KEYWORDS = (
    "slide",
    "slides",
    "poster",
    "form",
    "flyer",
    "leaflet",
    "press release",
    "news",
    "brochure",
    "infographic",
)


class BaseSpider:
    source = ""
    start_urls: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    max_pages = 200

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def crawl(self, limit: int | None = None) -> Iterable[PdfCandidate]:
        seen_pages: set[str] = set()
        seen_pdfs: set[str] = set()
        queue = deque(self.start_urls)

        while queue and len(seen_pages) < self.max_pages:
            page_url = queue.popleft()
            if page_url in seen_pages or not self.should_visit_page(page_url):
                continue
            seen_pages.add(page_url)
            try:
                html = self.client.get(page_url).text
            except Exception as exc:
                logger.warning("Skip page %s: %s", page_url, exc)
                continue
            page_meta = extract_page_metadata(html)
            for href, text in extract_links(page_url, html):
                if self.is_pdf_url(href):
                    candidate = self.make_candidate(href, text, page_url, page_meta)
                    if candidate and candidate.url not in seen_pdfs:
                        seen_pdfs.add(candidate.url)
                        yield candidate
                        if limit is not None and len(seen_pdfs) >= limit:
                            return
                elif self.should_enqueue_link(href, text):
                    queue.append(href)

    def run(
        self,
        raw_root: str = "data/raw_pdf",
        limit: int | None = None,
        source_dir_name: str | None = None,
    ) -> list[PdfMetadata]:
        downloader = PdfDownloader(raw_root=raw_root, client=self.client)
        return downloader.download_all(
            self.crawl(limit=limit),
            source=self.source,
            limit=limit,
            source_dir_name=source_dir_name,
        )

    def should_visit_page(self, url: str) -> bool:
        return same_domain_or_subdomain(url, self.allowed_domains)

    def should_enqueue_link(self, url: str, text: str) -> bool:
        if not same_domain_or_subdomain(url, self.allowed_domains):
            return False
        path = urlparse(url).path.lower()
        haystack = f"{path} {text}".lower()
        return any(keyword in haystack for keyword in GUIDELINE_KEYWORDS)

    def is_pdf_url(self, url: str) -> bool:
        return ".pdf" in urlparse(url).path.lower()

    def make_candidate(
        self,
        pdf_url: str,
        link_text: str,
        landing_url: str,
        page_meta: dict[str, str],
    ) -> PdfCandidate | None:
        title = normalize_space(link_text) or page_meta.get("title", "")
        haystack = f"{title} {landing_url} {pdf_url}".lower()
        if any(keyword in haystack for keyword in EXCLUDE_KEYWORDS):
            return None
        if not any(keyword in haystack for keyword in GUIDELINE_KEYWORDS):
            return None
        return PdfCandidate(
            source=self.source,
            url=absolute_url(landing_url, pdf_url),
            title=title,
            published_year=extract_year(page_meta.get("published_year"), title, pdf_url, landing_url),
            landing_url=landing_url,
        )


"""爬虫来源适配文件：面向特定医学资料来源抓取或解析原始记录，并为 ingestion 阶段提供统一输入。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlparse

import requests

from src.ingestion.crawler.common.records import PdfCandidate
from src.ingestion.crawler.common.text import absolute_url, extract_year, normalize_space, same_domain_or_subdomain
from src.ingestion.crawler.spiders.base import BaseSpider


logger = logging.getLogger(__name__)


class IdsaSpider(BaseSpider):
    source = "idsa"
    allowed_domains = (
        "idsociety.org",
        "academic.oup.com",
        "oup.silverchair-cdn.com",
        "doi.org",
        "cambridge.org",
        "journals.lww.com",
        "atsjournals.org",
        "aaos.org",
        "onlinelibrary.wiley.com",
        "ascopubs.org",
        "heartrhythmjournal.com",
        "aappublications.org",
    )
    start_urls = ("https://www.idsociety.org/practice-guideline/all-practice-guidelines",)
    search_url = "https://www.idsociety.org/practice-guideline/practice-guidelines/"
    bravo_base_url = "https://api.bravosquared.com"
    bravo_subscription_id = "98822ae0-5ab0-4a4b-8c4a-7f2805d592d0"
    bravo_ledger_id = "9b67932c-af7f-4119-ab99-0abc4967e52d"
    bravo_api_key = "0774909c-e9dd-4101-8c65-87c1ade4c76f"
    date_start = "2012-01-01"
    date_end = "2026-12-31"
    exclude_pdf_terms = (
        "executive",
        "summary",
        "supplement",
        "supplemental",
        "data",
        "figure",
        "figures",
        "table",
        "tables",
        "pathway",
        "overview",
        "slide",
        "appendix",
        "infographic",
    )

    def crawl(self, limit: int | None = None) -> Iterable[PdfCandidate]:
        emitted = 0
        for detail_url in self.iter_guideline_pages():
            candidate = self.make_detail_candidate(detail_url)
            if candidate is None:
                continue
            yield candidate
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def iter_guideline_pages(self) -> Iterable[str]:
        yielded = False
        for url in self.iter_search_guideline_pages():
            yielded = True
            yield url
        if yielded:
            return

        try:
            html = self.client.get(self.start_urls[0]).text
        except Exception as exc:
            logger.warning("IDSA guideline list failed: %s", exc)
            return
        parser = IdsaListParser(self.start_urls[0])
        parser.feed(html)
        seen: set[str] = set()
        for url in parser.guideline_urls:
            if url in seen:
                continue
            seen.add(url)
            yield url

    def iter_search_guideline_pages(self) -> Iterable[str]:
        headers = {
            "SubscriptionId": self.bravo_subscription_id,
            "ApiKey": self.bravo_api_key,
            "Content-Type": "application/json",
            "Origin": "https://www.idsociety.org",
            "Referer": self.search_url,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }
        seen: set[str] = set()
        page = 0
        total = None
        while total is None or page * 100 < total:
            req = self._search_request(page)
            try:
                response = requests.post(
                    f"{self.bravo_base_url}/api/search/v2/ledger/{self.bravo_ledger_id}/query",
                    headers=headers,
                    json=req,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                logger.warning("IDSA Bravo search failed at page %s: %s", page, exc)
                return
            total = int(data.get("numFound") or 0)
            docs = data.get("documents") or []
            if not docs:
                return
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                url = normalize_space(str(doc.get("url") or doc.get("url_na_str") or ""))
                if url and url not in seen:
                    seen.add(url)
                    yield url
            page += 1

    def _search_request(self, page: int) -> dict[str, object]:
        return {
            "boostFields": [{"field": "title", "value": 100}, {"field": "body", "value": 10}],
            "debug": False,
            "excludeFields": [],
            "facetFields": ["status_na_str", "organism_na_str", "idTopic_na_str"],
            "facetFilters": [],
            "facetMax": 50,
            "fuzzy": False,
            "highlight": False,
            "includeFields": [],
            "language": 0,
            "pinned": False,
            "rangeFields": ["date_na_dt"],
            "rangeFilters": [
                {
                    "facetName": "date_na_dt",
                    "greaterThanEqual": self.date_start,
                    "lessThanEqual": self.date_end,
                }
            ],
            "resultMax": 100,
            "searchFields": ["title", "name", "body_no_src", "source", "keywords", "doi", "journal", "format"],
            "searchQueryType": "multimatch",
            "searchTermOperator": 1,
            "searchTermType": "full",
            "searchTerm": "",
            "spellcheck": False,
            "startPage": page,
            "types": ["PracticeGuideline", "PicoPracticeGuideline"],
            "sortBy": "date_na_dt",
            "sortDirection": "desc",
        }

    def make_detail_candidate(self, detail_url: str) -> PdfCandidate | None:
        try:
            html = self.client.get(detail_url).text
        except Exception as exc:
            logger.warning("IDSA guideline detail failed: %s: %s", detail_url, exc)
            return None
        parser = IdsaDetailParser(detail_url)
        parser.feed(html)
        title = parser.title or detail_url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
        pdf_url = self.resolve_primary_pdf(parser.primary_href, parser.pdf_links)
        if not pdf_url:
            logger.warning("No strict primary PDF found for IDSA guideline: %s", detail_url)
            return None
        return PdfCandidate(
            source=self.source,
            url=pdf_url,
            title=title,
            published_year=extract_year(title, html[:8000], detail_url),
            landing_url=detail_url,
        )

    def resolve_primary_pdf(self, primary_href: str, pdf_links: list[tuple[str, str]]) -> str:
        if primary_href:
            if _looks_like_pdf(primary_href):
                return primary_href
            pdf = self.find_external_article_pdf(primary_href)
            if pdf:
                return pdf
        for href, text in pdf_links:
            haystack = f"{href} {text}".lower()
            if any(term in haystack for term in self.exclude_pdf_terms):
                continue
            if not _is_likely_guideline_pdf(href):
                continue
            return href
        return ""

    def find_external_article_pdf(self, url: str) -> str:
        if not same_domain_or_subdomain(url, self.allowed_domains):
            return ""
        try:
            response = self.client.get(url)
        except Exception as exc:
            logger.warning("IDSA external article page failed: %s: %s", url, exc)
            return ""
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or _looks_like_pdf(response.url):
            return response.url

        html = response.text
        parser = ExternalPdfParser(response.url)
        parser.feed(html)
        for href, text in parser.pdf_links:
            haystack = f"{href} {text}".lower()
            if any(term in haystack for term in self.exclude_pdf_terms):
                continue
            if _is_external_pdf_candidate(href):
                return href
        return ""


class IdsaListParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.guideline_urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        href = attrs_dict.get("href", "")
        classes = attrs_dict.get("class", "")
        if "list-pages__link" in classes and href.startswith("/practice-guideline/"):
            self.guideline_urls.append(absolute_url(self.base_url, href))


class IdsaDetailParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.title = ""
        self.primary_href = ""
        self.pdf_links: list[tuple[str, str]] = []
        self._capture_title = False
        self._title_parts: list[str] = []
        self._active_pdf_href = ""
        self._active_pdf_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() in {"h1", "title"} and attrs_dict.get("id") == "PageTitle":
            self._capture_title = True
            self._title_parts = []
        if tag.lower() != "a":
            return
        href = attrs_dict.get("href", "")
        if not href:
            return
        full_url = absolute_url(self.base_url, href)
        if attrs_dict.get("id") == "GuidelinePDFLink":
            self.primary_href = full_url
        if _looks_like_pdf(full_url):
            self._active_pdf_href = full_url
            self._active_pdf_text = []

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._title_parts.append(data)
        if self._active_pdf_href:
            self._active_pdf_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"h1", "title"} and self._capture_title:
            self.title = normalize_space(" ".join(self._title_parts))
            self._capture_title = False
        if tag.lower() == "a" and self._active_pdf_href:
            self.pdf_links.append((self._active_pdf_href, normalize_space(" ".join(self._active_pdf_text))))
            self._active_pdf_href = ""
            self._active_pdf_text = []


class ExternalPdfParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.pdf_links: list[tuple[str, str]] = []
        self._active_pdf_href = ""
        self._active_pdf_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            key = attrs_dict.get("name") or attrs_dict.get("property")
            content = attrs_dict.get("content")
            if key and key.lower() == "citation_pdf_url" and content:
                self.pdf_links.append((absolute_url(self.base_url, content), "citation_pdf_url"))
            return
        if tag.lower() != "a":
            return
        href = attrs_dict.get("href") or ""
        if not href:
            return
        full_url = absolute_url(self.base_url, href)
        if _is_external_pdf_candidate(full_url):
            self._active_pdf_href = full_url
            self._active_pdf_text = []

    def handle_data(self, data: str) -> None:
        if self._active_pdf_href:
            self._active_pdf_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._active_pdf_href:
            self.pdf_links.append((self._active_pdf_href, normalize_space(" ".join(self._active_pdf_text))))
            self._active_pdf_href = ""
            self._active_pdf_text = []


def _looks_like_pdf(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or ".pdf/" in path or ".pdf" in path


def _is_likely_guideline_pdf(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host.endswith("idsociety.org") and "/globalassets/idsa/practice-guidelines/" in path:
        return True
    if host.endswith("academic.oup.com") and ("article-pdf" in path or "advance-article-pdf" in path):
        return True
    if host.endswith("oup.silverchair-cdn.com") and path.endswith(".pdf"):
        return True
    if host.endswith("downloads.aap.org") and path.endswith(".pdf"):
        return True
    return False


def _is_external_pdf_candidate(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    return (
        _looks_like_pdf(url)
        or "article-pdf" in path
        or "advance-article-pdf" in path
        or "/doi/pdf/" in path
        or "/doi/epdf/" in path
        or path.endswith("/pdf")
        or query.startswith("download=true")
    )


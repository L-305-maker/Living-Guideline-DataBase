from __future__ import annotations

import logging
import re
from typing import Iterable
from urllib.parse import urlparse

from src.ingestion.crawler.common.text import absolute_url, extract_links, extract_year, normalize_space
from src.ingestion.crawler.common.records import PdfCandidate
from src.ingestion.crawler.spiders.base import BaseSpider


logger = logging.getLogger(__name__)


class WhoSpider(BaseSpider):
    source = "who"
    allowed_domains = ("who.int",)
    start_urls = (
        "https://www.who.int/publications/who-guidelines",
        "https://www.who.int/publications/i",
    )
    max_pages = 300
    api_url = "https://www.who.int/api/hubs/publications"
    guideline_office_id = "c09761c0-ab8e-4cfa-9744-99509c4d306b"
    page_size = 100
    strict_tags = {"Guideline", "Guidance (normative)"}
    strict_title_re = re.compile(r"\b(guideline|guidelines|recommendation|recommendations)\b", re.I)
    exclude_title_re = re.compile(
        r"\b(executive summary|summary of|at a glance|rapid communication|corrigendum|"
        r"resource kit|annex|references cited|contributors|fact sheets?|introduction)\b",
        re.I,
    )

    def crawl(self, limit: int | None = None) -> Iterable[PdfCandidate]:
        emitted = 0
        for item in self.iter_api_items():
            candidate = self.make_api_candidate(item)
            if candidate is None:
                continue
            yield candidate
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def iter_api_items(self) -> Iterable[dict[str, object]]:
        skip = 0
        total: int | None = None
        dollar = "$"
        while total is None or skip < total:
            params = {
                "sf_site": "15210d59-ad60-47ff-a542-7ed76645f0c7",
                "sf_provider": "OpenAccessProvider",
                "sf_culture": "en",
                dollar + "top": str(self.page_size),
                dollar + "skip": str(skip),
                dollar + "count": "true",
                dollar + "orderby": "PublicationDateAndTime desc",
                dollar + "select": (
                    "Title,ItemDefaultUrl,UrlName,FormatedDate,Tag,DownloadUrl,"
                    "PublicationDateAndTime,NumberOfPages"
                ),
                dollar + "filter": f"publishingoffices/any(s:s eq {self.guideline_office_id})",
            }
            try:
                data = self.client.get(self.api_url + "?" + _encode_params(params)).json()
            except Exception as exc:
                logger.warning("WHO publications API page failed at skip=%s: %s", skip, exc)
                return
            total = int(data.get("@odata.count") or 0)
            values = data.get("value") or []
            if not values:
                return
            for item in values:
                if isinstance(item, dict):
                    yield item
            skip += len(values)

    def make_api_candidate(self, item: dict[str, object]) -> PdfCandidate | None:
        title = normalize_space(str(item.get("Title") or ""))
        tag = normalize_space(str(item.get("Tag") or ""))
        if not title:
            return None
        if self.exclude_title_re.search(title):
            return None
        if tag not in self.strict_tags and not self.strict_title_re.search(title):
            return None
        landing_path = normalize_space(str(item.get("ItemDefaultUrl") or ""))
        url_name = normalize_space(str(item.get("UrlName") or ""))
        if not landing_path.startswith("/publications/"):
            landing_path = f"/publications/i/item/{url_name or landing_path.strip('/')}"
        landing_url = f"https://www.who.int{landing_path}"
        download_url = normalize_space(str(item.get("DownloadUrl") or "")) or self.find_primary_download_url(landing_url)
        if not download_url:
            logger.warning("No primary PDF download found for WHO publication: %s", landing_url)
            return None
        if not _is_who_download_host(download_url):
            logger.warning("Skip non-WHO download host for %s: %s", landing_url, download_url)
            return None
        return PdfCandidate(
            source=self.source,
            url=download_url,
            title=title,
            published_year=extract_year(str(item.get("PublicationDateAndTime") or ""), str(item.get("FormatedDate") or "")),
            landing_url=landing_url,
        )

    def find_primary_download_url(self, landing_url: str) -> str:
        try:
            html = self.client.get(landing_url).text
        except Exception as exc:
            logger.warning("WHO publication detail failed: %s: %s", landing_url, exc)
            return ""
        for href, text in extract_links(landing_url, html):
            haystack = f"{href} {text}".lower()
            if "download" in haystack and (
                "/server/api/core/bitstreams/" in href
                or ".pdf" in urlparse(href).path.lower()
            ):
                return absolute_url(landing_url, href)
        return ""

    def should_enqueue_link(self, url: str, text: str) -> bool:
        path = urlparse(url).path.lower()
        if path in {"/publications/who-guidelines", "/publications/i"}:
            return True
        return path.startswith("/publications/i/item/") or path.startswith("/publications/m/item/")

    def make_candidate(
        self,
        pdf_url: str,
        link_text: str,
        landing_url: str,
        page_meta: dict[str, str],
    ) -> PdfCandidate | None:
        landing_path = urlparse(landing_url).path.lower()
        if not (landing_path.startswith("/publications/i/item/") or landing_path.startswith("/publications/m/item/")):
            return None
        candidate = super().make_candidate(pdf_url, link_text, landing_url, page_meta)
        if candidate is None:
            return None
        haystack = f"{candidate.title} {candidate.url} {candidate.landing_url}".lower()
        if candidate.title.lower().startswith("download") and not any(
            key in haystack for key in ("guideline", "guidelines", "recommendation", "recommendations")
        ):
            return None
        return candidate


def _encode_params(params: dict[str, str]) -> str:
    from urllib.parse import urlencode

    return urlencode(params)


def _is_who_download_host(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in {"iris.who.int", "apps.who.int", "cdn.who.int", "www.who.int"}

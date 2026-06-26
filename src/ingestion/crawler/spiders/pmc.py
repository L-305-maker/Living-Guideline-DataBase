from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Iterable
from urllib.parse import urlencode, urlparse, urlunparse

from src.ingestion.crawler.common.http import HttpClient, PdfDownloader
from src.ingestion.crawler.common.records import PdfCandidate, PdfMetadata
from src.ingestion.crawler.common.text import extract_year, normalize_space
from src.ingestion.crawler.spiders.base import BaseSpider


logger = logging.getLogger(__name__)


class PmcSpider(BaseSpider):
    source = "pmc"
    allowed_domains = ("pmc.ncbi.nlm.nih.gov", "eutils.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "ftp.ncbi.nlm.nih.gov")
    start_urls: tuple[str, ...] = ()
    max_pages = 0
    query = '"Practice Guideline"[Publication Type] AND open access[filter] AND 2016:2027[pdat]'
    eutils_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    oa_base = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

    def __init__(self, client: HttpClient | None = None, retmax: int = 200) -> None:
        super().__init__(client=client)
        self.retmax = retmax
        self.api_key = os.getenv("NCBI_API_KEY", "").strip()
        self.email = os.getenv("NCBI_EMAIL", "").strip()
        self.tool = os.getenv("NCBI_TOOL", "medical_guideline_crawler").strip()
        self.client.delay = max(self.client.delay, 0.11 if self.api_key else 0.34)

    def crawl(self, limit: int | None = None) -> Iterable[PdfCandidate]:
        emitted = 0
        for docsum in self._iter_docsum_batches(limit):
            candidate = self._candidate_from_docsum(docsum)
            if candidate is None:
                continue
            yield candidate
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def run(
        self,
        raw_root: str = "data/raw_pdf",
        limit: int | None = None,
        source_dir_name: str | None = None,
    ) -> list[PdfMetadata]:
        original_respect_robots = self.client.respect_robots
        self.client.respect_robots = False
        try:
            downloader = PdfDownloader(raw_root=raw_root, client=self.client)
            return downloader.download_all(
                self.crawl(limit=limit),
                source=self.source,
                limit=limit,
                source_dir_name=source_dir_name,
            )
        finally:
            self.client.respect_robots = original_respect_robots

    def _iter_docsum_batches(self, max_items: int | None) -> Iterable[ET.Element]:
        retstart = 0
        seen = 0
        while max_items is None or seen < max_items:
            remaining = max_items - seen if max_items is not None else self.retmax
            retmax = min(self.retmax, remaining)
            search_root = self._get_xml(
                f"{self.eutils_base}/esearch.fcgi",
                {
                    "db": "pmc",
                    "retmode": "xml",
                    "retmax": str(retmax),
                    "retstart": str(retstart),
                    "sort": "pub date",
                    "term": self.query,
                },
            )
            if search_root is None:
                return
            ids = [node.text for node in search_root.findall(".//Id") if node.text]
            if not ids:
                return
            summary_root = self._get_xml(
                f"{self.eutils_base}/esummary.fcgi",
                {"db": "pmc", "retmode": "xml", "id": ",".join(ids)},
            )
            if summary_root is None:
                return
            for docsum in summary_root.findall(".//DocSum"):
                yield docsum
            returned = len(ids)
            seen += returned
            retstart += returned
            total = int((search_root.findtext(".//Count") or "0").strip() or "0")
            if retstart >= total:
                return

    def _candidate_from_docsum(self, docsum: ET.Element) -> PdfCandidate | None:
        pmcid = _normalize_pmcid(_docsum_item(docsum, "ArticleIds/pmcid") or _docsum_item(docsum, "pmcid"))
        if not pmcid:
            pmcid = _normalize_pmcid(_docsum_item(docsum, "Id"))
        if not pmcid:
            return None

        title = normalize_space(_docsum_item(docsum, "Title"))
        pub_date = _docsum_item(docsum, "PubDate")
        published_year = extract_year(pub_date, title)
        if published_year and not 2016 <= int(published_year) <= 2027:
            return None

        pdf_url = self._oa_pdf_url(pmcid)
        if not pdf_url:
            logger.info("Skip %s: no OA PDF link", pmcid)
            return None
        return PdfCandidate(
            source=self.source,
            url=pdf_url,
            title=title or pmcid,
            published_year=published_year,
            landing_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        )

    def _oa_pdf_url(self, pmcid: str) -> str:
        root = self._get_xml(self.oa_base, {"id": pmcid})
        if root is None:
            return ""
        error = root.findtext(".//error")
        if error:
            logger.warning("PMC OA lookup failed for %s: %s", pmcid, normalize_space(error))
            return ""
        record = root.find(".//record")
        if record is None or record.attrib.get("retracted", "").lower() == "yes":
            return ""
        for link in record.findall("link"):
            if link.attrib.get("format", "").lower() == "pdf":
                return _pmc_oa_download_url(link.attrib.get("href", ""))
        return ""

    def _get_xml(self, base_url: str, params: dict[str, str]) -> ET.Element | None:
        request_params = dict(params)
        if base_url.startswith(self.eutils_base):
            request_params["tool"] = self.tool
            if self.email:
                request_params["email"] = self.email
            if self.api_key:
                request_params["api_key"] = self.api_key
        url = f"{base_url}?{urlencode(request_params)}"
        try:
            return ET.fromstring(self.client.get(url).text)
        except Exception as exc:
            logger.exception("PMC API request failed: %s", exc)
            return None


def _docsum_item(docsum: ET.Element, name: str) -> str:
    if name == "ArticleIds/pmcid":
        for item in docsum.findall(".//Item[@Name='ArticleIds']/Item"):
            if item.attrib.get("Name", "").lower() == "pmcid":
                return normalize_space(item.text or "")
        return ""
    item = docsum.find(f".//Item[@Name='{name}']")
    return normalize_space(item.text or "") if item is not None else ""


def _normalize_pmcid(value: str) -> str:
    value = normalize_space(value)
    if not value:
        return ""
    return value if value.upper().startswith("PMC") else f"PMC{value}"


def _pmc_oa_download_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "ftp.ncbi.nlm.nih.gov":
        return url
    path = parsed.path
    if path.startswith("/pub/pmc/") and not path.startswith("/pub/pmc/deprecated/"):
        path = path.replace("/pub/pmc/", "/pub/pmc/deprecated/", 1)
    return urlunparse(("https", parsed.netloc, path, parsed.params, parsed.query, parsed.fragment))

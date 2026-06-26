from __future__ import annotations

from src.ingestion.crawler.spiders.base import BaseSpider


class UspstfSpider(BaseSpider):
    source = "uspstf"
    allowed_domains = ("uspreventiveservicestaskforce.org",)
    start_urls = (
        "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation-topics",
        "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation-statements",
    )
    max_pages = 350


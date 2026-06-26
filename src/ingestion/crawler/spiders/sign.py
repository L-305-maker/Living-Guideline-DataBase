from __future__ import annotations

from src.ingestion.crawler.spiders.base import BaseSpider


class SignSpider(BaseSpider):
    source = "sign"
    allowed_domains = ("sign.ac.uk",)
    start_urls = (
        "https://www.sign.ac.uk/our-guidelines/",
    )
    max_pages = 250


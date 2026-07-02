"""爬虫来源适配文件：面向特定医学资料来源抓取或解析原始记录，并为 ingestion 阶段提供统一输入。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

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



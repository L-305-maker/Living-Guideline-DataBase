"""爬虫来源适配文件：面向特定医学资料来源抓取或解析原始记录，并为 ingestion 阶段提供统一输入。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from src.ingestion.crawler.spiders.ada import AdaSpider
from src.ingestion.crawler.spiders.base import BaseSpider
from src.ingestion.crawler.spiders.idsa import IdsaSpider
from src.ingestion.crawler.spiders.pmc import PmcSpider
from src.ingestion.crawler.spiders.sign import SignSpider
from src.ingestion.crawler.spiders.uspstf import UspstfSpider
from src.ingestion.crawler.spiders.who import WhoSpider


SPIDERS: dict[str, type[BaseSpider]] = {
    "who": WhoSpider,
    "uspstf": UspstfSpider,
    "ada": AdaSpider,
    "idsa": IdsaSpider,
    "sign": SignSpider,
    "pmc": PmcSpider,
}


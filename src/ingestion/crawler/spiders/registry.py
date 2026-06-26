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

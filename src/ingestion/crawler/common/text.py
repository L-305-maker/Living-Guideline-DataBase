from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse


YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
SPACE_RE = re.compile(r"\s+")


def normalize_space(value: str | None) -> str:
    return SPACE_RE.sub(" ", value or "").strip()


def safe_filename(value: str, fallback: str = "document", max_len: int = 120) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", value.strip(), flags=re.UNICODE).strip("._")
    cleaned = cleaned[:max_len].strip("._")
    return cleaned or fallback


def extract_year(*values: str | None) -> str:
    for value in values:
        match = YEAR_RE.search(value or "")
        if match:
            return match.group(1)
    return ""


def absolute_url(base_url: str, href: str) -> str:
    joined, _ = urldefrag(urljoin(base_url, unescape(href)))
    return joined


def same_domain_or_subdomain(url: str, domains: tuple[str, ...]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)


class LinkExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._active_href = absolute_url(self.base_url, href)
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._active_href:
            self.links.append((self._active_href, normalize_space(" ".join(self._active_text))))
            self._active_href = None
            self._active_text = []


class TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta: dict[str, str] = {}
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k.lower(): v or "" for k, v in attrs}
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() == "meta":
            key = attrs_dict.get("property") or attrs_dict.get("name")
            content = attrs_dict.get("content")
            if key and content:
                self.meta[key.lower()] = normalize_space(content)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
            self.title = normalize_space(" ".join(self._title_parts))


def extract_links(base_url: str, html: str) -> list[tuple[str, str]]:
    parser = LinkExtractor(base_url)
    parser.feed(html)
    return parser.links


def extract_page_metadata(html: str) -> dict[str, str]:
    parser = TitleParser()
    parser.feed(html)
    title = parser.meta.get("og:title") or parser.meta.get("dc.title") or parser.title
    year = extract_year(
        parser.meta.get("article:published_time"),
        parser.meta.get("dc.date"),
        parser.meta.get("citation_publication_date"),
        html[:5000],
    )
    return {"title": title, "published_year": year}


from __future__ import annotations

import hashlib
import logging
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.robotparser import RobotFileParser

import requests

from src.ingestion.crawler.common.jsonl import append_jsonl, iter_jsonl
from src.ingestion.crawler.common.records import PdfCandidate, PdfMetadata
from src.ingestion.crawler.common.text import safe_filename


logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "medical-guideline-crawler/1.0 (+research corpus; respectful crawling)"
}


class HttpClient:
    def __init__(
        self,
        timeout: int = 30,
        retries: int = 3,
        delay: float = 1.0,
        headers: dict[str, str] | None = None,
        respect_robots: bool = True,
    ) -> None:
        """创建带重试、延迟和 robots 约束的礼貌 HTTP 客户端。"""

        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers.update(headers or DEFAULT_HEADERS)
        self._robots: dict[str, RobotFileParser] = {}

    def can_fetch(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception as exc:  # robots 获取失败不应中断整次爬取。
                logger.warning("Could not read robots.txt for %s: %s", base, exc)
            self._robots[base] = rp
        allowed = self._robots[base].can_fetch(self.session.headers["User-Agent"], url)
        if allowed:
            return True
        # 部分旧 robots 文件只列出被屏蔽的 bot 名称，却没有 User-agent: *。
        # 这种情况下不要把无关 bot 的屏蔽规则误判为全站禁止抓取。
        if not _has_matching_robot_group(self._robots[base], self.session.headers["User-Agent"]):
            return True
        return False

    def get(self, url: str, *, stream: bool = False) -> requests.Response:
        if not self.can_fetch(url):
            raise RuntimeError(f"robots.txt disallows fetching {url}")
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            if self.delay:
                time.sleep(self.delay + random.uniform(0, self.delay / 3))
            try:
                response = self.session.get(url, timeout=self.timeout, stream=stream)
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                last_exc = exc
                response = exc.response
                status_code = response.status_code if response is not None else None
                if status_code in {429, 503}:
                    wait = _retry_wait(response, self.delay, attempt)
                else:
                    wait = min(30.0, self.delay * (2 ** attempt))
                logger.warning("Request failed (%s/%s) %s: %s", attempt, self.retries, url, exc)
                time.sleep(wait)
            except requests.RequestException as exc:
                last_exc = exc
                wait = min(30.0, self.delay * (2 ** attempt))
                logger.warning("Request failed (%s/%s) %s: %s", attempt, self.retries, url, exc)
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url}") from last_exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PdfDownloader:
    def __init__(self, raw_root: str | Path = "data/raw_pdf", client: HttpClient | None = None) -> None:
        self.raw_root = Path(raw_root)
        self.client = client or HttpClient()

    def download_all(
        self,
        candidates: Iterable[PdfCandidate],
        source: str,
        limit: int | None = None,
        source_dir_name: str | None = None,
    ) -> list[PdfMetadata]:
        source_dir = self.raw_root / (source_dir_name or source)
        source_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = source_dir / "metadata.jsonl"
        seen_urls = {item.get("url", "") for item in iter_jsonl(metadata_path)}
        seen_hashes = {item.get("sha256", "") for item in iter_jsonl(metadata_path)}
        downloaded: list[PdfMetadata] = []

        for candidate in candidates:
            if limit is not None and len(downloaded) >= limit:
                break
            if candidate.url in seen_urls:
                logger.info("Skip already recorded PDF: %s", candidate.url)
                continue
            try:
                name_seed = candidate.title or Path(urlparse(candidate.url).path).stem
                file_name = safe_filename(name_seed, fallback=source) + ".pdf"
                pdf_path = _unique_path(source_dir / file_name)
                if urlparse(candidate.url).scheme.lower() == "ftp":
                    self._download_ftp(candidate.url, pdf_path)
                else:
                    response = self.client.get(candidate.url, stream=True)
                    with pdf_path.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 128):
                            if chunk:
                                f.write(chunk)
                digest = sha256_file(pdf_path)
                if digest in seen_hashes:
                    pdf_path.unlink(missing_ok=True)
                    logger.info("Skip duplicate PDF hash for %s", candidate.url)
                    continue
                metadata = PdfMetadata(
                    source=source,
                    url=candidate.url,
                    title=candidate.title,
                    published_year=candidate.published_year,
                    pdf_path=str(pdf_path),
                    landing_url=candidate.landing_url,
                    sha256=digest,
                )
                append_jsonl(metadata_path, metadata.to_json())
                seen_urls.add(candidate.url)
                seen_hashes.add(digest)
                downloaded.append(metadata)
                logger.info("Downloaded %s", pdf_path)
            except Exception as exc:
                logger.exception("Failed to download %s: %s", candidate.url, exc)
        return downloaded

    def _download_ftp(self, url: str, pdf_path: Path) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, self.client.retries + 1):
            if self.client.delay:
                time.sleep(self.client.delay + random.uniform(0, self.client.delay / 3))
            try:
                with urlopen(url, timeout=self.client.timeout) as response, pdf_path.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 128)
                        if not chunk:
                            break
                        f.write(chunk)
                return
            except Exception as exc:
                last_exc = exc
                wait = min(30.0, self.client.delay * (2 ** attempt))
                logger.warning("FTP download failed (%s/%s) %s: %s", attempt, self.client.retries, url, exc)
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url}") from last_exc


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique path for {path}")


def _retry_wait(response: requests.Response | None, base_delay: float, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        retry_after = retry_after.strip()
        if retry_after.isdigit():
            return min(300.0, max(float(retry_after), base_delay))
        try:
            retry_at = parsedate_to_datetime(retry_after)
            return min(300.0, max((retry_at.timestamp() - time.time()), base_delay))
        except (TypeError, ValueError, OverflowError):
            pass
    return min(300.0, max(base_delay, 1.0) * (2 ** (attempt - 1)))


def _has_matching_robot_group(robot_parser: RobotFileParser, user_agent: str) -> bool:
    user_agent = user_agent.split("/")[0].lower()
    entries = getattr(robot_parser, "entries", [])
    default_entry = getattr(robot_parser, "default_entry", None)
    if default_entry is not None:
        return True
    for entry in entries:
        for agent in getattr(entry, "useragents", []):
            agent = agent.lower()
            if agent == "*" or agent == user_agent or agent in user_agent:
                return True
    return False

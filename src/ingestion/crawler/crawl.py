from __future__ import annotations

import argparse
import logging
import os

from src.ingestion.crawler.spiders.registry import SPIDERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download guideline PDFs and write data/raw_pdf/<source>/metadata.jsonl.")
    parser.add_argument("sources", nargs="+", choices=sorted(SPIDERS), help="Sources to crawl")
    parser.add_argument("--raw-root", default="data/raw_pdf")
    parser.add_argument("--source-dir-name", default=None, help="Override output directory name for one source")
    parser.add_argument("--limit", type=int, default=None, help="Maximum new PDFs per source")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--ignore-robots", action="store_true")
    parser.add_argument("--ncbi-api-key", default=None, help="NCBI API key for PMC E-utilities requests")
    parser.add_argument("--ncbi-email", default=None, help="Developer email sent with PMC E-utilities requests")
    parser.add_argument("--ncbi-tool", default=None, help="Tool name sent with PMC E-utilities requests")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(name)s: %(message)s")

    from src.ingestion.crawler.common.http import HttpClient

    client = HttpClient(
        timeout=args.timeout,
        retries=args.retries,
        delay=args.delay,
        respect_robots=not args.ignore_robots,
    )
    if args.source_dir_name and len(args.sources) != 1:
        parser.error("--source-dir-name can only be used with one source at a time")
    if args.ncbi_api_key:
        os.environ["NCBI_API_KEY"] = args.ncbi_api_key
    if args.ncbi_email:
        os.environ["NCBI_EMAIL"] = args.ncbi_email
    if args.ncbi_tool:
        os.environ["NCBI_TOOL"] = args.ncbi_tool
    for source in args.sources:
        spider = SPIDERS[source](client=client)
        downloaded = spider.run(raw_root=args.raw_root, limit=args.limit, source_dir_name=args.source_dir_name)
        logging.info("%s: downloaded %s new PDFs", source, len(downloaded))


if __name__ == "__main__":
    main()

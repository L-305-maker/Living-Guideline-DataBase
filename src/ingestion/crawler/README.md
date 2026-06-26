# Medical guideline PDF crawlers

This package stops at the first-pass origin JSONL stage.

## Crawl PDFs

```bash
python -m src.ingestion.crawler.crawl who uspstf ada sign pmc --limit 20
```

Downloaded PDFs are stored under `data/raw_pdf/<source>/`.
Each source also gets `data/raw_pdf/<source>/metadata.jsonl` with:

- `url`
- `title`
- `published_year`
- `source`
- `pdf_path`
- `landing_url`
- `sha256`

### PMC Open Access practice guidelines

The `pmc` source downloads PDF files for PubMed Central Open Access Subset
records matching:

```text
"Practice Guideline"[Publication Type] AND open access[filter] AND 2016:2027[pdat]
```

It does not crawl PubMed or PMC HTML pages. Discovery uses NCBI E-utilities,
then each PMCID is checked against the PMC OA Web Service API and only PDF links
returned by that OA API are downloaded.
Legacy PMC FTP links returned by the OA API are normalized to the current
`https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/...` download path.

NCBI request metadata can be provided with command-line options or environment
variables:

```bash
python -m src.ingestion.crawler.crawl pmc --limit 20 --ncbi-email you@example.org --ncbi-tool medical_guideline_crawler
python -m src.ingestion.crawler.crawl pmc --limit 20 --ncbi-api-key "$env:NCBI_API_KEY"
```

For PMC, the crawler enforces a minimum delay of 0.34 seconds per request
without an API key and 0.11 seconds with an API key. HTTP 429 and 503 responses
are retried with exponential backoff and `Retry-After` support.

## Build origin JSONL

```bash
python -m src.ingestion.crawler.process_origin who uspstf ada sign pmc
```

Outputs are written as `data/origin/<source>_origin.jsonl`.
Each row contains exactly the initial structured fields requested:

- `content`
- `title`
- `url`
- `published_year`
- `source`
- `tables`

PDF parsing uses `pdfplumber`; install dependencies with:

```bash
pip install -r requirements.txt
```

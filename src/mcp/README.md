# PDF Markdown RAG MCP Server

This server exposes three MCP tools:

- `search`: search candidate guideline documents by query, institution, department, and time range.
- `read`: read a clean Markdown document by `doc_id` or `title`.
- `retrieve`: retrieve traceable RAG chunks with BM25 + vector RRF fusion.

## Install

```powershell
python -m pip install "mcp>=1.9"
```

## Run

```powershell
$env:RAG_BACKEND = "local"
$env:RAG_DATA_DIR = "D:\python\Data_splitting_ingestion\data\evidence"
python -m src.mcp
```

## MCP Client Config Example

```json
{
  "mcpServers": {
    "pdf-markdown-rag": {
      "command": "python",
      "args": ["-m", "src.mcp"],
      "cwd": "D:\\python\\Data_splitting_ingestion",
      "env": {
        "RAG_BACKEND": "local",
        "RAG_DATA_DIR": "D:\\python\\Data_splitting_ingestion\\data\\evidence"
      }
    }
  }
}
```

## Tool Arguments

### `search`

```json
{
  "query": "糖尿病 胰岛素",
  "source_institution": "CMA",
  "clinical_department": "内分泌",
  "time_range": "2012-2026",
  "publication_date": null,
  "recency_boost": true,
  "topk": 10
}
```

### `read`

```json
{
  "doc_id": "cma_2021_560ba98613e7",
  "title": null,
  "max_chars": null
}
```

### `retrieve`

```json
{
  "query": "糖尿病患者胰岛素治疗如何管理",
  "source_institution": "CMA",
  "clinical_department": "内分泌",
  "time_range": "2012-2026",
  "publication_date": null,
  "topk": 5
}
```

## Local Smoke Test

This verifies the underlying tool APIs even when MCP is not installed:

```powershell
python -X utf8 -m src.mcp.smoke_test --data-dir data/evidence
```

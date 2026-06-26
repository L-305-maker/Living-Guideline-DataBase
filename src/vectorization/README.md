# Vectorization Preparation

`src/vectorization/` builds a stable JSONL queue for embedding and vector-store ingestion. It does not call an embedding
provider and does not write to pgvector or another vector database.

The default embedding contract is BGE-M3 dense retrieval:

- model: `BAAI/bge-m3`
- embedding version label: `dense-v1`
- pgvector dimensions: `1024`
- default max length in the ingest CLI: `8192`

Default inputs from a processed run:

- `recommendation_versions.jsonl`: only `publishable` versions by default.
- `blocks.jsonl`: recommendation, evidence, GRADE, and PICO-like source blocks.
- `pico_questions.jsonl`: clinical question retrieval anchors.
- `evidence_items.jsonl`: evidence retrieval anchors.

Example:

```powershell
python -m src.vectorization.embedding_queue `
  --run-dir data\processed\runs\full_rule_20260625_151944 `
  --output data\processed\runs\full_rule_20260625_151944\embedding_queue.jsonl `
  --manifest-output data\processed\runs\full_rule_20260625_151944\embedding_queue_manifest.jsonl
```

Use `--include-review-versions` to route non-publishable versions into `lg_review_candidates`. Keep review vectors
separate from published recommendation vectors so low-quality candidates cannot pollute production retrieval.

After queue generation, ingest with pgvector:

```powershell
python -m src.storage.vector_ingest init
python -m src.storage.vector_ingest ingest --queue <embedding_queue.jsonl>
```

Use `--fake-embeddings` only for local database smoke tests.

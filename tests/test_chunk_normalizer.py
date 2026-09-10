from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.chunk_normalizer import iter_normalized_chunks, normalize_chunk_record
from src.storage.postgres_store import helper_chunk_rows


class ChunkNormalizerTest(unittest.TestCase):
    def test_normalizes_structural_chunk_fields(self) -> None:
        record = {
            "chunk_id": "doc1_recommendation_bundle_abc",
            "source_doc_id": "doc1",
            "chunk_type": "recommendation_bundle",
            "text": "We recommend treatment.\n\nEvidence:\nTrial benefit.",
            "section_path": ["Guideline", "Recommendations"],
            "text_for_embedding": "[GUIDELINE SECTION]: Guideline > Recommendations\n[CHUNK TYPE]: recommendation_bundle",
            "recommendation": "We recommend treatment.",
            "evidence": ["Trial benefit."],
            "metadata": {"order_index": 3},
        }
        docs = {
            "doc1": {
                "title": "Demo Guideline",
                "publication_date": "2026-01-01",
                "source_institution": "Demo",
                "clinical_department": "Cardiology",
                "source_file": "demo.pdf",
                "markdown_clean_path": "demo.md",
            }
        }

        chunk = normalize_chunk_record(record, docs, 7)

        self.assertEqual("doc1", chunk["doc_id"])
        self.assertEqual("Demo Guideline", chunk["title"])
        self.assertEqual(7, chunk["chunk_index"])
        self.assertEqual(record["text"], chunk["content"])
        self.assertEqual(record["text_for_embedding"], chunk["retrieval_text"])
        self.assertEqual("doc1#7", chunk["retrieval_key"])
        self.assertEqual(["Trial benefit."], chunk["evidence"])

    def test_postgres_reader_accepts_structural_chunks(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            data_dir = Path(tmp)
            (data_dir / "chunks").mkdir()
            document = {
                "doc_id": "doc1",
                "title": "Demo Guideline",
                "publication_date": "2026-01-01",
                "source_institution": "Demo",
                "clinical_department": "Cardiology",
                "source_file": "demo.pdf",
                "markdown_clean_path": "demo.md",
            }
            chunk = {
                "chunk_id": "doc1_clinical_detail_abc",
                "source_doc_id": "doc1",
                "chunk_type": "clinical_detail",
                "text": "Dose: 5 mg PO daily.",
                "section_path": ["Demo", "Dose"],
                "text_for_embedding": "[GUIDELINE SECTION]: Demo > Dose\n[CHUNK TYPE]: clinical_detail\n[CONTENT]:\nDose: 5 mg PO daily.",
            }
            (data_dir / "sections").mkdir()
            (data_dir / "documents.jsonl").write_text(json.dumps(document) + "\n", encoding="utf-8")
            (data_dir / "chunks" / "all_chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")

            normalized = list(iter_normalized_chunks(data_dir))
            pg_rows = helper_chunk_rows(data_dir)

        self.assertEqual(1, len(normalized))
        self.assertEqual("doc1", normalized[0]["doc_id"])
        self.assertEqual("doc1", pg_rows[0][1])
        self.assertEqual("clinical_detail", pg_rows[0][12])
        self.assertIn("[GUIDELINE SECTION]", pg_rows[0][13])


if __name__ == "__main__":
    unittest.main()

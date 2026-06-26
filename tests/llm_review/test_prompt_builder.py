import json
import unittest

from src.pipeline.llm_review.prompts.builder import build_recommendation_prompt


def _extract_input_json(prompt: str) -> dict:
    marker = "INPUT_JSON:\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n\nNow return OUTPUT_JSON only.", start)
    return json.loads(prompt[start:end])


class RecommendationPromptBuilderTests(unittest.TestCase):
    def test_recommendation_prompt_includes_document_cleaning_policy(self) -> None:
        prompt = build_recommendation_prompt(
            {
                "queue_id": "queue_1",
                "recommendation_candidate_id": "rec_candidate_1",
                "task_type": "recommendation_candidate_review",
                "source_text": "Return to recommendations",
                "current_candidate_state": {
                    "recommendation": {
                        "candidate_id": "rec_candidate_1",
                        "source_text": "Return to recommendations",
                        "recommendation_text": "Return to recommendations",
                    }
                },
            }
        )

        self.assertIn("navigation text", prompt)
        self.assertIn("methodology or wording templates", prompt)
        self.assertIn("research-only recommendations", prompt)
        self.assertIn("GRADE legends/tables", prompt)
        self.assertIn("OCR corruption", prompt)

        payload = _extract_input_json(prompt)
        self.assertIn("review_policy", payload)
        self.assertIn("navigation_or_toc_text", payload["review_policy"]["reject_text_types"])
        self.assertIn("Grade for quality of evidence", payload["review_policy"]["reject_examples"])

    def test_recommendation_prompt_preserves_source_provenance_context(self) -> None:
        prompt = build_recommendation_prompt(
            {
                "queue_id": "queue_2",
                "recommendation_candidate_id": "rec_candidate_2",
                "task_type": "recommendation_candidate_review",
                "source_section": "Recommendations > Treatment",
                "source_text": "Offer CPAP to adults with moderate or severe obstructive sleep apnoea.",
                "current_candidate_state": {
                    "recommendation": {
                        "candidate_id": "rec_candidate_2",
                        "record_id": "record_1",
                        "guideline_id": "guideline_1",
                        "source_url": "https://example.test/guideline",
                        "source_section": "Recommendations > Treatment",
                        "source_text": "Offer CPAP to adults with moderate or severe obstructive sleep apnoea.",
                        "recommendation_text": "Offer CPAP to adults with moderate or severe obstructive sleep apnoea.",
                        "normalized_payload": {
                            "block_id": "block_12",
                            "section_path": ["Recommendations", "Treatment"],
                            "source_order": 4,
                            "statement_index": 0,
                            "section_role": "recommendations",
                            "block_type": "list_item",
                            "quality_status": "trusted_pdf_text",
                            "text_quality_score": 0.96,
                            "quality_notes": [],
                            "source_metadata": {"page": 12, "bbox": [10, 20, 200, 240]},
                        },
                    }
                },
            }
        )

        payload = _extract_input_json(prompt)
        provenance = payload["source_provenance"]
        self.assertEqual(provenance["block_id"], "block_12")
        self.assertEqual(provenance["section_role"], "recommendations")
        self.assertEqual(provenance["block_type"], "list_item")
        self.assertEqual(provenance["quality_status"], "trusted_pdf_text")
        self.assertEqual(provenance["page"], 12)
        self.assertEqual(provenance["bbox"], [10, 20, 200, 240])

        compact_state = payload["current_candidate_state"]["recommendation"]
        self.assertEqual(compact_state["source_provenance"]["block_id"], "block_12")
        self.assertEqual(compact_state["source_provenance"]["text_quality_score"], 0.96)


if __name__ == "__main__":
    unittest.main()

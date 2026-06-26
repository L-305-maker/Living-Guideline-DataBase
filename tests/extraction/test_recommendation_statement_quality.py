import json
import unittest
from pathlib import Path

from src.pipeline.extraction.routing.candidate_router import routed_block
from src.pipeline.extraction.recommendation.candidate_extractor import (
    evaluate_recommendation_statement,
    extract_from_block,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class RecommendationStatementQualityTests(unittest.TestCase):
    def test_meta_recommendation_statement_is_rejected(self) -> None:
        result = evaluate_recommendation_statement(
            "The recommendations in this guideline were formulated to meet the needs of most patients."
        )
        self.assertEqual(result["decision"], "reject")
        self.assertIn("guideline_meta_statement", result["noise_reasons"])

    def test_incomplete_action_phrase_is_not_pending(self) -> None:
        block = {
            "block_id": "incomplete",
            "record_id": "r",
            "guideline_id": "g",
            "text": "We suggest that clinicians use",
            "section_path": ["Demo", "Recommendations"],
            "candidate_hints": ["recommendation"],
            "quality": {"skip_candidate_extraction": False},
        }
        candidates, _trace = extract_from_block(routed_block(block))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].status, "needs_review")
        self.assertIn("incomplete_action_phrase", candidates[0].normalized_payload["quality_notes"])

    def test_negative_recommendations_are_kept_as_action(self) -> None:
        texts = [
            "We recommend against routine antibiotics for uncomplicated viral infection.",
            "Clinicians should not use valproic acid during pregnancy unless no safer alternative is available.",
            "This intervention is not recommended for patients with known hypersensitivity.",
        ]
        for text in texts:
            result = evaluate_recommendation_statement(text)
            self.assertNotEqual(result["decision"], "reject", text)
            self.assertTrue(result["action_patterns"], text)

    def test_positive_anchor_recall(self) -> None:
        anchors = _load_jsonl(FIXTURE_DIR / "recommendation_positive_anchor_sample.jsonl")
        hits = 0
        for block in anchors:
            candidate_block = dict(block)
            candidate_block["candidate_hints"] = ["recommendation"]
            candidate_block["quality"] = {"skip_candidate_extraction": False}
            candidates, _trace = extract_from_block(routed_block(candidate_block))
            if len(candidates) >= int(block.get("expected_min_candidates", 1)):
                hits += 1
        self.assertGreaterEqual(hits / len(anchors), 0.95)


if __name__ == "__main__":
    unittest.main()

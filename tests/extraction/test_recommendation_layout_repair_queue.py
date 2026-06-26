import unittest

from src.pipeline.extraction.routing.candidate_router import route_blocks, routed_block
from src.pipeline.extraction.recommendation.candidate_extractor import extract_from_block


def _block(text, section=None, block_type="paragraph"):
    return {
        "block_id": "block",
        "record_id": "record",
        "guideline_id": "guideline",
        "text": text,
        "section_path": section or ["Demo", "Recommendations"],
        "block_type": block_type,
        "candidate_hints": ["recommendation"],
        "quality": {"skip_candidate_extraction": False},
    }


def _quality_block(text, quality):
    block = _block(text, section=["Demo", "Results"])
    block["quality"].update(quality)
    return block


class RecommendationLayoutRepairQueueTests(unittest.TestCase):
    def test_layout_glued_action_routes_to_repair_not_extraction(self) -> None:
        block = _block("ThisTextIsLayoutGluedAcrossColumns and clinicians should offer therapy for adults with disease.")
        routed = routed_block(block)
        self.assertEqual(routed["route"]["route_action"], "needs_layout_repair")
        candidates, _trace = extract_from_block(routed)
        self.assertEqual(candidates, [])

    def test_table_action_routes_to_repair(self) -> None:
        block = _block("Table 1 Class of recommendation: clinicians should offer treatment for adults.", block_type="table")
        routed = routed_block(block)
        self.assertEqual(routed["route"]["route_action"], "needs_layout_repair")

    def test_administrative_without_action_skips(self) -> None:
        block = _block("Copyright 2026. All rights reserved. Contact the corresponding author for permissions.")
        routed = routed_block(block)
        self.assertEqual(routed["route"]["route_action"], "skip")

    def test_layout_without_action_skips(self) -> None:
        block = _block("ThisTextIsLayoutGluedAcrossColumns without a clinical action statement.")
        routed = routed_block(block)
        self.assertEqual(routed["route"]["route_action"], "skip")

    def test_route_blocks_writes_repair_queue(self) -> None:
        routed = route_blocks([
            _block("ThisTextIsLayoutGluedAcrossColumns and clinicians should offer therapy for adults with disease.")
        ])
        self.assertEqual(len(routed["layout_repair"]), 1)
        self.assertEqual(len(routed["recommendation"]), 0)

    def test_paper_record_suppresses_recommendation_extraction_outside_recommendation_section(self) -> None:
        block = _quality_block(
            "The study recommends further research and clinicians should consider these findings cautiously.",
            {"record_cleaning_status": "ready", "record_type": "paper"},
        )
        routed = routed_block(block)
        self.assertEqual(routed["route"]["route_action"], "skip")
        self.assertIn("paper_record_recommendation_suppressed", routed["route"]["noise_reasons"])
        candidates, _trace = extract_from_block(routed)
        self.assertEqual(candidates, [])

    def test_non_ready_record_suppresses_candidate_extraction(self) -> None:
        block = _quality_block(
            "Clinicians should offer treatment for adults with disease.",
            {"record_cleaning_status": "needs_layout_repair", "record_type": "guideline"},
        )
        routed = routed_block(block)
        self.assertEqual(routed["route"]["route_action"], "skip")
        self.assertIn("record_cleaning_status_needs_layout_repair", routed["route"]["noise_reasons"])


if __name__ == "__main__":
    unittest.main()

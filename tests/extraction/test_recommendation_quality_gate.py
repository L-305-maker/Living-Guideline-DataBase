"""抽取阶段测试文件：验证推荐、GRADE、PICO、证据和 source span 等候选抽取行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import json
import unittest
from pathlib import Path

from src.pipeline.extraction.routing.candidate_router import routed_block
from src.pipeline.extraction.recommendation.candidate_extractor import extract_from_block
from src.pipeline.extraction.recommendation.candidate_extractor import split_recommendation_statements


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class RecommendationQualityGateTests(unittest.TestCase):
    def test_auto_rejected_noise_blocks_do_not_extract_candidates(self) -> None:
        blocks = _load_jsonl(FIXTURE_DIR / "recommendation_noise_blocks.jsonl")
        self.assertGreaterEqual(len(blocks), 20)

        for block in blocks:
            routed = routed_block(block)
            route = routed["route"]
            self.assertIn(route["route_action"], {"skip", "sentence_only", "needs_layout_repair"})
            candidates, _trace = extract_from_block(routed)
            self.assertEqual(candidates, [], block["block_id"])

    def test_clear_recommendation_blocks_are_recalled(self) -> None:
        blocks = _load_jsonl(FIXTURE_DIR / "recommendation_positive_blocks.jsonl")
        hits = 0

        for block in blocks:
            routed = routed_block(block)
            self.assertNotEqual(routed["route"]["route_action"], "skip", block["block_id"])
            candidates, _trace = extract_from_block(routed)
            if candidates:
                hits += 1

        self.assertGreaterEqual(hits / len(blocks), 0.95)

    def test_long_blocks_never_emit_overlong_statement(self) -> None:
        text = (
            "Background text. "
            + "column noise " * 90
            + "We recommend treatment for adults with disease. "
            + "more column noise " * 90
        )
        statements = split_recommendation_statements(text)
        self.assertTrue(all(len(statement) <= 700 for statement in statements))

    def test_direct_clinical_action_statement_is_retained(self) -> None:
        block = {
            "block_id": "nice_offer_1",
            "record_id": "record_1",
            "guideline_id": "guideline_1",
            "text": "Offer continuous positive airway pressure to adults with moderate or severe obstructive sleep apnoea.",
            "section_path": ["Recommendations", "Treatment"],
            "candidate_hints": ["recommendation"],
        }

        routed = routed_block(block)
        candidates, _trace = extract_from_block(routed)

        self.assertEqual(routed["route"]["route_action"], "normal_extract")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].status, "pending")


if __name__ == "__main__":
    unittest.main()


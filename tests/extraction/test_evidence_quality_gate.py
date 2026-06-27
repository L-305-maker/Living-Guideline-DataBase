"""抽取阶段测试文件：验证推荐、GRADE、PICO、证据和 source span 等候选抽取行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.pipeline.extraction.evidence.item_extractor import (
    PicoIndex,
    RecommendationIndex,
    build_evidence_item,
    extract_from_block,
)


def _block(text, quality=None):
    return {
        "block_id": "block-evidence",
        "record_id": "record-evidence",
        "guideline_id": "guideline-evidence",
        "paper_id": "paper-evidence",
        "text": text,
        "order": 1,
        "section_path": ["Demo", "Evidence"],
        "route": {"primary_task": "evidence_extraction"},
        "quality": quality or {"record_type": "paper", "record_quality_flags": []},
    }


class EvidenceQualityGateTests(unittest.TestCase):
    def test_unlinked_background_text_does_not_emit_evidence_item(self) -> None:
        block = _block("This background section describes the guideline scope and implementation context.")
        item = build_evidence_item(block, PicoIndex([]), RecommendationIndex([]), include_unlinked=True)

        self.assertIsNone(item)

    def test_unlinked_structured_evidence_goes_to_association_review(self) -> None:
        block = _block(
            "A randomized trial reported mortality outcome RR 0.72, 95% CI 0.55 to 0.94, sample size 1200."
        )
        item = build_evidence_item(block, PicoIndex([]), RecommendationIndex([]), include_unlinked=True)

        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["screening_status"], "association_review")
        self.assertTrue(item["normalized_payload"]["has_structured_evidence_signal"])

    def test_trace_reports_no_structured_evidence_signal(self) -> None:
        block = _block("This background section describes the guideline scope and implementation context.")
        item, trace = extract_from_block(block, PicoIndex([]), RecommendationIndex([]), include_unlinked=True)

        self.assertIsNone(item)
        self.assertEqual(trace.parsed_output["skipped_reason"], "no_structured_evidence_signal")


if __name__ == "__main__":
    unittest.main()


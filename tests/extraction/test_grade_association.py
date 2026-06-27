"""抽取阶段测试文件：验证推荐、GRADE、PICO、证据和 source span 等候选抽取行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.pipeline.extraction.grade.candidate_extractor import RecommendationIndex


class GradeAssociationTests(unittest.TestCase):
    def test_nearest_same_section_recommendation_can_link_beyond_five_blocks(self) -> None:
        index = RecommendationIndex(
            [
                {
                    "candidate_id": "rec-1",
                    "record_id": "record-1",
                    "source_section": "Demo Recommendations Pharmacologic treatment",
                    "normalized_payload": {"source_order": 20, "block_id": "rec-block"},
                }
            ]
        )
        block = {
            "block_id": "grade-block",
            "record_id": "record-1",
            "order": 10,
            "section_path": ["Demo", "Recommendations", "Pharmacologic treatment"],
        }

        candidate_id, reason = index.find(block)

        self.assertEqual(candidate_id, "rec-1")
        self.assertEqual(reason, "nearest_same_section_order_10")
        self.assertEqual(index.find(block).quality, "weak")

    def test_far_different_section_grade_does_not_link(self) -> None:
        index = RecommendationIndex(
            [
                {
                    "candidate_id": "rec-1",
                    "record_id": "record-1",
                    "source_section": "Demo Recommendations Screening",
                    "normalized_payload": {"source_order": 20, "block_id": "rec-block"},
                }
            ]
        )
        block = {
            "block_id": "grade-block",
            "record_id": "record-1",
            "order": 10,
            "section_path": ["Demo", "Evidence", "Pharmacologic treatment"],
        }

        candidate_id, reason = index.find(block)

        self.assertEqual(candidate_id, "")
        self.assertEqual(reason, "not_found")
        self.assertEqual(index.find(block).quality, "missing")

    def test_near_order_association_quality_is_medium(self) -> None:
        index = RecommendationIndex(
            [
                {
                    "candidate_id": "rec-1",
                    "record_id": "record-1",
                    "source_section": "Demo Recommendations",
                    "normalized_payload": {"source_order": 13, "block_id": "rec-block"},
                }
            ]
        )
        block = {
            "block_id": "grade-block",
            "record_id": "record-1",
            "order": 10,
            "section_path": ["Demo", "Evidence"],
        }

        match = index.find(block)

        self.assertEqual(match.candidate_id, "rec-1")
        self.assertEqual(match.reason, "nearest_order_3")
        self.assertEqual(match.quality, "medium")
        self.assertEqual(match.distance, 3)


if __name__ == "__main__":
    unittest.main()


"""清洗阶段测试文件：验证 source_cleaner、quality_gate、offset 和记录类型路由等清洗链路行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.pipeline.parsing.structure_parser import parse_source_record


def _record(record_type):
    return {
        "record_id": f"record-{record_type}",
        "record_type": record_type,
        "title": "Demo",
        "source": "demo",
        "content": "Recommendations\nClinicians should offer treatment for adults with disease.",
        "guideline_seed": {"guideline_id": "guideline-1"},
        "paper_seed": {"paper_id": "paper-1"},
        "direct_extraction": {"guideline_id": "guideline-1", "paper_id": "paper-1"},
    }


class RecordTypeRoutingTests(unittest.TestCase):
    def test_guideline_record_blocks_do_not_keep_paper_id(self) -> None:
        blocks = parse_source_record(_record("guideline"))

        self.assertTrue(blocks)
        self.assertEqual(blocks[0]["guideline_id"], "guideline-1")
        self.assertEqual(blocks[0]["paper_id"], "")

    def test_paper_record_blocks_do_not_keep_guideline_id(self) -> None:
        blocks = parse_source_record(_record("paper"))

        self.assertTrue(blocks)
        self.assertEqual(blocks[0]["guideline_id"], "")
        self.assertEqual(blocks[0]["paper_id"], "paper-1")

    def test_mixed_record_blocks_keep_both_ids(self) -> None:
        blocks = parse_source_record(_record("mixed"))

        self.assertTrue(blocks)
        self.assertEqual(blocks[0]["guideline_id"], "guideline-1")
        self.assertEqual(blocks[0]["paper_id"], "paper-1")


if __name__ == "__main__":
    unittest.main()


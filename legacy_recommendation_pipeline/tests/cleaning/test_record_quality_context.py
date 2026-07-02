"""清洗阶段测试文件：验证 source_cleaner、quality_gate、offset 和记录类型路由等清洗链路行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.common.record_quality import block_record_quality_context, record_quality_context


class RecordQualityContextTests(unittest.TestCase):
    def test_record_context_normalizes_dict_flags(self) -> None:
        context = record_quality_context(
            {
                "cleaning_status": "ready",
                "cleaning_quality_score": 0.82,
                "cleaning_quality_flags": [{"code": "single_line_flattened_text"}],
                "record_type": "guideline",
            }
        )

        self.assertEqual(context["record_cleaning_status"], "ready")
        self.assertEqual(context["record_quality_flags"], ["single_line_flattened_text"])
        self.assertEqual(context["record_type"], "guideline")

    def test_block_context_reads_quality_or_metadata_flags(self) -> None:
        context = block_record_quality_context(
            {
                "quality": {
                    "record_cleaning_status": "ready",
                    "record_quality_flags": ["table_heavy_without_structured_tables"],
                    "record_type": "paper",
                },
                "metadata": {
                    "record_cleaning_status": "needs_layout_repair",
                    "record_type": "guideline",
                },
            }
        )

        self.assertEqual(context["record_cleaning_status"], "ready")
        self.assertEqual(context["record_quality_flags"], ["table_heavy_without_structured_tables"])
        self.assertEqual(context["record_type"], "paper")


if __name__ == "__main__":
    unittest.main()


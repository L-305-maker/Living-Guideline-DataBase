"""抽取阶段测试文件：验证推荐、GRADE、PICO、证据和 source span 等候选抽取行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.pipeline.extraction.grade.candidate_extractor import grade_domain_fields, infer_certainty as infer_grade_certainty
from src.pipeline.extraction.recommendation.candidate_extractor import infer_certainty as infer_recommendation_certainty


class FlowCertaintyRuleTests(unittest.TestCase):
    def test_certainty_was_low_phrase_is_consistent_across_extractors(self) -> None:
        text = "The overall certainty of evidence was low due to risk of bias and imprecision."

        self.assertEqual(infer_recommendation_certainty(text), "low")
        self.assertEqual(infer_grade_certainty(text), "low")

    def test_grade_downgrade_reasons_populate_domain_fields(self) -> None:
        domains = grade_domain_fields(["risk_of_bias", "imprecision"])

        self.assertEqual(domains["risk_of_bias"], "serious_or_concern")
        self.assertEqual(domains["imprecision"], "serious_or_concern")
        self.assertEqual(domains["inconsistency"], "not_reported")


if __name__ == "__main__":
    unittest.main()


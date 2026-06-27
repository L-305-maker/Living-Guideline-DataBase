"""编排流程测试文件：验证端到端流水线入口、公开 API 和阶段衔接是否正常。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import read_jsonl, write_jsonl
from src.pipeline.orchestration import run_pipeline


class PipelineOrchestrationTests(unittest.TestCase):
    def test_run_pipeline_writes_manifest_and_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = root / "origin.jsonl"
            output_root = root / "runs"
            raw_content = (
                "Recommendations\n"
                "Recommendation 1: In adults with insomnia, we suggest cognitive behavioral therapy.\n"
                "The overall certainty of evidence was low due to risk of bias and imprecision.\n"
                "References\n1. Smith J. Lancet. 2020."
            )
            write_jsonl(
                origin,
                [
                    {
                        "content": raw_content,
                        "tables": [],
                        "title": "Demo Guideline",
                        "source": "aasm",
                        "published_year": "2026",
                    }
                ],
            )

            result = run_pipeline(origin, output_root, run_id="test_run")
            manifest = json.loads((output_root / "test_run" / "run_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(result.run_id, "test_run")
            self.assertEqual(manifest["pipeline_version"], "guideline_review_pipeline_v1")
            self.assertEqual(manifest["artifact_counts"]["ready_cleaned"], 1)
            self.assertGreaterEqual(manifest["artifact_counts"]["blocks"], 1)
            self.assertGreaterEqual(manifest["artifact_counts"]["recommendation_candidates"], 1)
            self.assertIn("pico_extraction", manifest["stage_summaries"])
            self.assertIn("evidence_extraction", manifest["stage_summaries"])
            self.assertIn("recommendation_versioning", manifest["stage_summaries"])
            self.assertIn("update_logs", manifest["stage_summaries"])
            self.assertIn("quality_reports", manifest["stage_summaries"])
            self.assertIn("recommendation_versions", manifest["artifact_counts"])
            self.assertEqual(manifest["artifact_counts"]["recommendation_versions_report"], 1)
            self.assertEqual(manifest["artifact_counts"]["update_logs_report"], 1)
            self.assertGreaterEqual(manifest["artifact_counts"]["quality_report_files"], 5)
            self.assertTrue(read_jsonl(output_root / "test_run" / "llm_review_queue.jsonl"))
            self.assertTrue((output_root / "test_run" / "quality_reports" / "candidate_overall_quality_summary.jsonl").exists())


if __name__ == "__main__":
    unittest.main()


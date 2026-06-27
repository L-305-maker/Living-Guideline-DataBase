"""质量评估测试文件：验证 goldset、数据产物 manifest 和质量报告相关逻辑。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import tempfile
import unittest
from pathlib import Path

from src.common.data_artifacts import build_manifest


class DataArtifactManifestTests(unittest.TestCase):
    def test_manifest_classifies_processed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "processed" / "tmp_test_run").mkdir(parents=True)
            (root / "processed" / "regression100_latest").mkdir(parents=True)
            (root / "processed" / "tmp_test_run" / "sample.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "origin").mkdir()
            (root / "origin" / "origin.jsonl").write_text("{}\n", encoding="utf-8")

            manifest = build_manifest(root)
            runs = {item["path"]: item for item in manifest["processed_runs"]}
            self.assertEqual(runs["processed/tmp_test_run"]["type"], "temporary_run")
            self.assertFalse(runs["processed/tmp_test_run"]["keep"])
            self.assertEqual(runs["processed/regression100_latest"]["type"], "regression_baseline")
            self.assertTrue(runs["processed/regression100_latest"]["keep"])


if __name__ == "__main__":
    unittest.main()


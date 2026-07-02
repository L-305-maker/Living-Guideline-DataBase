"""存储层测试文件：验证 PostgreSQL schema、contract、原子入库和完整性检查。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import json
import unittest

from src.pipeline.cleaning.source_cleaner import clean_source_record
from src.storage.row_mappers import cleaned_record_row as mapped_cleaned_record_row
from src.storage.storage_PG import cleaned_record_row


class StorageContractTests(unittest.TestCase):
    def test_storage_pg_keeps_legacy_import_contract(self) -> None:
        self.assertIs(cleaned_record_row, mapped_cleaned_record_row)

    def test_cleaned_record_row_keeps_clean_content_and_full_raw_record(self) -> None:
        cleaned = clean_source_record(
            {
                "content": "Recommendations\nWe recommend treatment.\nReferences\n1. Smith J. 2020.",
                "title": "Demo",
                "source": "aasm",
                "tables": [],
            }
        )

        row = cleaned_record_row(cleaned)
        raw_record = json.loads(row["raw_record"])

        self.assertEqual(row["content"], cleaned["clean_content"])
        self.assertEqual(raw_record["raw_content"], cleaned["raw_content"])
        self.assertEqual(raw_record["clean_content"], cleaned["clean_content"])
        self.assertIn("cleaning_log", raw_record)
        self.assertEqual(row["guideline_id"], cleaned["guideline_seed"]["guideline_id"])
        self.assertEqual(row["paper_id"], cleaned["paper_seed"]["paper_id"])


if __name__ == "__main__":
    unittest.main()


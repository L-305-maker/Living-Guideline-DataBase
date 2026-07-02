"""存储层测试文件：验证 PostgreSQL schema、contract、原子入库和完整性检查。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.storage.schema import SCHEMA_TABLES
from src.storage.schema_definitions import CONSTRAINT_MIGRATIONS, STORAGE_SCHEMA_VERSION, TABLE_DDL


class StorageSchemaConstraintTests(unittest.TestCase):
    def test_schema_tracks_migration_table(self) -> None:
        self.assertIn("storage_schema_migrations", SCHEMA_TABLES)
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS storage_schema_migrations" in ddl for ddl in TABLE_DDL))

    def test_constraint_migrations_cover_core_quality_guards(self) -> None:
        sql = "\n".join(CONSTRAINT_MIGRATIONS)

        self.assertIn("ck_recommendation_candidates_status", sql)
        self.assertIn("ck_recommendation_candidates_confidence_range", sql)
        self.assertIn("ck_recommendation_versions_quality_status", sql)
        self.assertIn("ck_evidence_items_extraction_confidence_range", sql)
        self.assertIn("NOT VALID", sql)
        self.assertIn(STORAGE_SCHEMA_VERSION, sql)


if __name__ == "__main__":
    unittest.main()


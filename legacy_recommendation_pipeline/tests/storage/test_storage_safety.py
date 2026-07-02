"""存储层测试文件：验证 PostgreSQL schema、contract、原子入库和完整性检查。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import unittest

from src.storage.connection import mask_database_url


class StorageSafetyTests(unittest.TestCase):
    def test_database_url_password_is_masked(self) -> None:
        masked = mask_database_url("postgresql://user:secret@localhost:5432/db")
        self.assertEqual(masked, "postgresql://user:***@localhost:5432/db")
        self.assertNotIn("secret", masked)

    def test_database_url_without_password_is_unchanged(self) -> None:
        url = "postgresql://localhost/db"
        self.assertEqual(mask_database_url(url), url)


if __name__ == "__main__":
    unittest.main()


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

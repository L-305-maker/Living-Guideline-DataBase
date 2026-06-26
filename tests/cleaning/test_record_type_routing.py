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

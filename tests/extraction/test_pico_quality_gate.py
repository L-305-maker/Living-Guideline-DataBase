import unittest

from src.pipeline.extraction.pico.question_extractor import build_pico


def _block(text, quality=None):
    return {
        "block_id": "block-pico",
        "record_id": "record-pico",
        "guideline_id": "guideline-pico",
        "text": text,
        "order": 1,
        "section_path": ["Demo", "Recommendations"],
        "route": {"primary_task": "pico_extraction"},
        "quality": quality or {"record_type": "guideline", "record_quality_flags": []},
    }


class PicoQualityGateTests(unittest.TestCase):
    def test_pico_without_outcomes_is_under_review(self) -> None:
        row = build_pico(
            _block("In adults with chronic insomnia, clinicians should use cognitive behavioral therapy."),
            "In adults with chronic insomnia, clinicians should use cognitive behavioral therapy.",
            0,
        )

        self.assertEqual(row["status"], "under_review")
        self.assertLessEqual(row["extraction_confidence"], 0.64)

    def test_complete_pico_can_remain_active(self) -> None:
        row = build_pico(
            _block("In adults with chronic insomnia, clinicians should use cognitive behavioral therapy to improve sleep quality."),
            "In adults with chronic insomnia, clinicians should use cognitive behavioral therapy to improve sleep quality.",
            0,
        )

        self.assertEqual(row["status"], "active")
        self.assertGreaterEqual(row["extraction_confidence"], 0.8)

    def test_weak_record_structure_lowers_pico_confidence(self) -> None:
        row = build_pico(
            _block(
                "In adults with chronic insomnia, clinicians should use cognitive behavioral therapy to improve sleep quality.",
                {"record_type": "guideline", "record_quality_flags": ["single_line_flattened_text"]},
            ),
            "In adults with chronic insomnia, clinicians should use cognitive behavioral therapy to improve sleep quality.",
            0,
        )

        self.assertEqual(row["status"], "under_review")
        self.assertLess(row["extraction_confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()

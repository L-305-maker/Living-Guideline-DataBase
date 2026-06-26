import unittest

from src.pipeline.cleaning.offset_mapping import map_clean_span_to_raw
from src.pipeline.extraction.source_span.first_pass import run_source_span_first_pipeline


class OffsetMappingTests(unittest.TestCase):
    def test_hyphenated_raw_text_maps_to_clean_span(self) -> None:
        raw = "Recommendation 1: We suggest cognitive behav-\nioral therapy."
        clean = "Recommendation 1: We suggest cognitive behavioral therapy."
        start = clean.index("cognitive")
        end = len(clean)

        result = map_clean_span_to_raw(raw, clean, start, end)

        self.assertIsNotNone(result.raw_start)
        self.assertIsNotNone(result.raw_end)
        self.assertIn(result.mapping_confidence, {"normalized", "fuzzy"})

    def test_source_span_first_outputs_raw_and_clean_offsets(self) -> None:
        raw = (
            "Recommendations\n"
            "Recommendation 1: In adults with insomnia, we suggest cognitive behavioral therapy.\n"
            "References\n1. Smith J. Lancet. 2020."
        )

        payload = run_source_span_first_pipeline(
            {
                "content": raw,
                "tables": [],
                "title": "Demo Guideline",
                "source": "aasm",
                "guideline_seed": {"guideline_id": "guideline_1"},
            }
        )

        recommendation = payload["entities"]["recommendation_versions"][0]
        self.assertEqual(recommendation["raw_start_char"], raw.index("Recommendation 1:"))
        self.assertIsNotNone(recommendation["raw_end_char"])
        self.assertIsNotNone(recommendation["clean_start_char"])
        self.assertEqual(recommendation["offset_mapping"]["version"], "offset_mapping_v1")


if __name__ == "__main__":
    unittest.main()

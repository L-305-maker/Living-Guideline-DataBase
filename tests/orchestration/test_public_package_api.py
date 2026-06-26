import unittest


class PublicPackageApiTests(unittest.TestCase):
    def test_pipeline_public_api_imports(self) -> None:
        from src import pipeline

        self.assertTrue(callable(pipeline.run_pipeline))

    def test_extraction_public_api_imports(self) -> None:
        from src.pipeline import extraction

        self.assertTrue(callable(extraction.route_blocks))
        self.assertTrue(callable(extraction.extract_recommendation_candidates_file))
        self.assertTrue(callable(extraction.run_source_span_first_pipeline))
        self.assertTrue(callable(extraction.build_version))

    def test_llm_review_public_api_imports(self) -> None:
        from src.pipeline import llm_review

        self.assertTrue(callable(llm_review.build_queue_file))
        self.assertTrue(callable(llm_review.call_queue_items_with_config))
        self.assertTrue(callable(llm_review.validate_llm_result_for_item))

    def test_quality_public_api_imports(self) -> None:
        from src.pipeline import quality

        self.assertTrue(callable(quality.build_block_quality_report))
        self.assertTrue(callable(quality.build_candidate_quality_report))
        self.assertTrue(callable(quality.build_pico_quality_report))


if __name__ == "__main__":
    unittest.main()

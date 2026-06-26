import unittest

from src.pipeline.cleaning.source_cleaner import clean_source_record
from src.pipeline.extraction.common.mappers import map_payload_to_business_tables
from src.pipeline.extraction.source_span.first_pass import (
    extract_evidence_items,
    extract_grade_assessments,
    extract_pico_questions,
    extract_recommendation_spans,
    extract_recommendations,
    run_source_span_first_pipeline,
)
from src.pipeline.extraction.common.validators import validate_recommendation


class SourceSpanFirstExtractionTests(unittest.TestCase):
    def test_recommendation_span_detection(self) -> None:
        text = "Recommendations\nRecommendation 1: In adults with chronic insomnia disorder, the AASM suggests CBT-I."

        spans = extract_recommendation_spans(text)

        self.assertGreaterEqual(len(spans), 1)
        self.assertTrue(spans[0]["source_span"])
        self.assertEqual(spans[0]["recommendation_code"], "Recommendation 1")

    def test_recommendation_field_extraction(self) -> None:
        spans = extract_recommendation_spans(
            "Recommendations\nRecommendation 1: We suggest treatment. Conditional recommendation, low certainty of evidence."
        )
        recs = extract_recommendations(spans)

        self.assertEqual(recs[0]["strength"], "conditional")
        self.assertEqual(recs[0]["certainty"], "low")

    def test_direction_extraction_against(self) -> None:
        spans = extract_recommendation_spans("Recommendations\nRecommendation 1: The panel suggests against the use of drug A.")
        recs = extract_recommendations(spans)

        self.assertEqual(recs[0]["direction"], "against")

    def test_pico_extraction_from_recommendation_text(self) -> None:
        spans = extract_recommendation_spans(
            "Recommendation 1: In adults with chronic insomnia disorder, "
            "the AASM suggests the use of combination treatment with CBT-I plus insomnia medication "
            "over insomnia medication alone."
        )
        recs = extract_recommendations(spans)
        picos = extract_pico_questions(recs)

        self.assertEqual(picos[0]["population"], "adults with chronic insomnia disorder")
        self.assertIn("CBT-I plus insomnia medication", picos[0]["intervention"])
        self.assertEqual(picos[0]["comparator"], "insomnia medication alone")

    def test_evidence_summary_extraction(self) -> None:
        text = "Evidence\nThe TF identified six RCTs published in seven articles."
        items = extract_evidence_items(text, recommendations=[{"recommendation_id": "rec_1"}], picos=[{"recommendation_id": "rec_1", "pico_id": "pico_1"}])

        self.assertEqual(items[0]["study_design"], "RCT")
        self.assertEqual(items[0]["study_count"], 6)
        self.assertEqual(items[0]["article_count"], 7)
        self.assertEqual(items[0]["evidence_type"], "evidence_summary")

    def test_grade_extraction_uses_not_reported_for_unmentioned_domains(self) -> None:
        text = "Evidence\nThe overall certainty of evidence was low due to risk of bias and imprecision."
        grades = extract_grade_assessments(text, recommendations=[{"recommendation_id": "rec_1"}], picos=[{"recommendation_id": "rec_1", "pico_id": "pico_1"}])

        self.assertEqual(grades[0]["final_certainty"], "low")
        self.assertNotEqual(grades[0]["risk_of_bias"], "not_reported")
        self.assertNotEqual(grades[0]["imprecision"], "not_reported")
        self.assertIn(grades[0]["inconsistency"], {"not_reported", "unclear"})
        self.assertIn(grades[0]["indirectness"], {"not_reported", "unclear"})
        self.assertIn(grades[0]["publication_bias"], {"not_reported", "unclear"})

    def test_no_source_span_no_formal_ingestion(self) -> None:
        recommendation = {
            "recommendation_id": "rec_1",
            "recommendation_version_id": "rv_1",
            "recommendation_text": "We suggest treatment.",
            "direction": "for",
            "strength": "conditional",
            "certainty": "low",
        }
        recommendation["validation"] = validate_recommendation(recommendation).to_dict()
        payload = {"entities": {"recommendation_versions": [recommendation], "model_traces": []}}

        mapped = map_payload_to_business_tables(payload)

        self.assertFalse(recommendation["validation"]["is_valid"])
        self.assertEqual(mapped["recommendation_versions"], [])

    def test_model_trace_first_payload_then_formal_mapping(self) -> None:
        payload = run_source_span_first_pipeline(
            {
                "content": "Recommendations\nRecommendation 1: In adults with insomnia, we suggest CBT-I.",
                "tables": [],
                "title": "Demo Guideline",
                "source": "aasm",
                "guideline_seed": {"guideline_id": "guideline_1"},
            }
        )

        self.assertTrue(payload["entities"]["model_traces"])
        mapped = map_payload_to_business_tables(payload)
        self.assertTrue(mapped["model_traces"])
        self.assertTrue(mapped["recommendation_versions"])
        self.assertTrue(mapped["recommendation_versions"][0]["source_span"])
        self.assertIsNotNone(mapped["recommendation_versions"][0]["start_char"])
        self.assertIsNotNone(mapped["recommendation_versions"][0]["end_char"])

    def test_references_do_not_generate_recommendations(self) -> None:
        text = "Recommendations\nRecommendation 1: We suggest CBT-I.\nReferences\n1. Authors recommend additional research."

        spans = extract_recommendation_spans(text)

        self.assertEqual(len(spans), 1)
        self.assertNotIn("Authors recommend", spans[0]["source_span"])

    def test_table_fallback_when_record_tables_empty(self) -> None:
        cleaned = clean_source_record(
            {
                "content": "Recommendations\nRecommendation 1: We suggest CBT-I.\nTable 1\nOutcome Importance\nReferences\n1. Smith J.",
                "tables": [],
                "title": "Demo Guideline",
                "source": "aasm",
            }
        )

        self.assertTrue(cleaned["tables"])
        self.assertEqual(cleaned["cleaning_warnings"][0]["warning_type"], "tables_empty_but_table_text_detected")


if __name__ == "__main__":
    unittest.main()

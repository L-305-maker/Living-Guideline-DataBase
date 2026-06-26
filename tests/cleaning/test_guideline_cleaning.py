import unittest

from src.pipeline.cleaning.source_cleaner import clean_source_record, clean_source_records, fix_hyphenation


class GuidelineCleaningTests(unittest.TestCase):
    def test_raw_content_is_preserved(self) -> None:
        record = {"content": "Recommendation 1: We recommend treatment.", "title": "Demo", "source": "aasm"}
        output = clean_source_record(record)

        self.assertEqual(output["raw_content"], record["content"])
        self.assertTrue(output["clean_content"])

    def test_pdf_hyphenation_is_fixed_without_breaking_real_hyphens(self) -> None:
        text = "recommenda- tion treat- ment pharmaco- logical evidence-based patient-centered short-term"

        fixed = fix_hyphenation(text)

        self.assertIn("recommendation treatment pharmacological", fixed)
        self.assertIn("evidence-based", fixed)
        self.assertIn("patient-centered", fixed)
        self.assertIn("short-term", fixed)

    def test_references_are_split_from_clean_content(self) -> None:
        output = clean_source_record(
            {
                "content": "Conclusion\nUse treatment.\nReferences\n1. Smith J. Lancet. 2020.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertNotIn("Smith J.", output["clean_content"])
        self.assertIn("References", output["references_text"])

    def test_inline_references_are_split_from_clean_content(self) -> None:
        output = clean_source_record(
            {
                "content": "Conclusion. Use treatment. References 1. Smith J. Lancet. 2020.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertNotIn("Smith J.", output["clean_content"])
        self.assertIn("References", output["references_text"])

    def test_early_references_heading_before_real_body_is_not_split(self) -> None:
        body = (
            "Preface text.\nReferences\n1. Smith J. Lancet. 2020.\n"
            "CHAPTER 1: DEFINITION AND OVERVIEW\nOVERALL KEY POINTS\n"
            + "Patients with disease should receive treatment according to clinical status.\n" * 120
        )
        output = clean_source_record(
            {
                "content": body,
                "title": "Demo",
                "source": "gold",
            }
        )

        self.assertIn("CHAPTER 1", output["clean_content"])
        self.assertEqual(output["references_text"], "")

    def test_empty_tables_falls_back_to_table_text_warning(self) -> None:
        output = clean_source_record(
            {
                "content": "Recommendations\nWe suggest treatment.\nTable 1\nCritical outcomes\nTable 2\nImplications",
                "tables": [],
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertTrue(output["tables"])
        self.assertEqual(output["table_count"], 2)
        self.assertGreater(output["table_row_count"], 0)
        self.assertIn("tables_empty_but_table_text_detected", {item["warning_type"] for item in output["cleaning_warnings"]})

    def test_recommendation_is_structured(self) -> None:
        output = clean_source_record(
            {
                "content": "Recommendation 1: In adults with chronic insomnia disorder, the AASM suggests treatment.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertGreaterEqual(len(output["recommendations"]), 1)
        recommendation = output["recommendations"][0]
        self.assertIn("text", recommendation)
        self.assertIn("direction", recommendation)
        self.assertIn("strength", recommendation)
        self.assertIn("certainty", recommendation)

    def test_grade_unmentioned_dimensions_are_not_reported(self) -> None:
        output = clean_source_record(
            {
                "content": "The overall certainty of evidence was low due to risk of bias and imprecision.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        grade = output["grade_assessments"][0]
        self.assertEqual(grade["certainty"], "low")
        self.assertEqual(grade["risk_of_bias"], "serious_or_concern")
        self.assertEqual(grade["imprecision"], "serious_or_concern")
        self.assertEqual(grade["inconsistency"], "not_reported")
        self.assertEqual(grade["indirectness"], "not_reported")
        self.assertEqual(grade["publication_bias"], "not_reported")

    def test_affiliations_are_split_from_clean_content(self) -> None:
        output = clean_source_record(
            {
                "content": "Recommendations\nWe recommend treatment.\nAuthors and Affiliations\nDepartment of Medicine, Demo University.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertNotIn("Department of Medicine", output["clean_content"])
        self.assertIn("Authors and Affiliations", output["affiliations_text"])

    def test_inline_affiliations_are_split_from_clean_content(self) -> None:
        output = clean_source_record(
            {
                "content": "Recommendations. We recommend treatment. Authors and Affiliations Department of Medicine, Demo University.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertNotIn("Department of Medicine", output["clean_content"])
        self.assertIn("Authors and Affiliations", output["affiliations_text"])

    def test_flattened_inline_section_headings_are_restored(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Background This guideline addresses adults with insomnia and summarizes the clinical context. "
                    "Recommendations We suggest cognitive behavioral therapy for adults with chronic insomnia. "
                    "Evidence The certainty of evidence was moderate for sleep outcomes."
                ),
                "title": "Demo",
                "source": "aasm",
            }
        )

        section_names = {section["section_name"] for section in output["sections"]}
        self.assertIn("Background", section_names)
        self.assertIn("Recommendations", section_names)
        self.assertGreaterEqual(output["cleaning_log"]["inline_section_breaks_inserted"], 2)

    def test_flattened_numbered_recommendations_are_restored(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Background This long guideline paragraph explains the population, context, eligibility, "
                    "intervention choices, outcomes, and implementation setting before recommendations. "
                    "Recommendations Recommendation 1: We suggest cognitive behavioral therapy for adults with "
                    "chronic insomnia. Recommendation 2: We suggest shared decision making before hypnotic drugs. "
                    "Evidence The certainty of evidence was moderate for sleep outcomes."
                ),
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertIn("\nRecommendation 1:", output["clean_content"])
        self.assertIn("\nRecommendation 2:", output["clean_content"])
        self.assertGreaterEqual(len(output["recommendations"]), 2)

    def test_two_column_rows_are_reordered_before_space_normalization(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Recommendations                         Evidence\n"
                    "Recommendation 1: We suggest CBT.       Trial evidence showed benefit.\n"
                    "Recommendation 2: We avoid drug A.      Harms were increased.\n"
                    "Recommendation 3: We offer follow-up.   Certainty was moderate.\n"
                ),
                "title": "Two Column Guideline",
                "source": "aasm",
            }
        )

        clean = output["clean_content"]
        self.assertLess(clean.index("Recommendation 1"), clean.index("Trial evidence"))
        self.assertTrue(output["cleaning_log"]["two_column_layout_repaired"])

    def test_repeated_page_headers_and_footers_are_removed(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Demo Guideline Page 1\nRecommendations\nWe recommend treatment.\n"
                    "Demo Guideline Page 1\nEvidence\nThe certainty of evidence was moderate.\n"
                    "Demo Guideline Page 1\n"
                ),
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertNotIn("Demo Guideline Page 1", output["clean_content"])

    def test_figure_and_table_blocks_are_isolated_from_main_content(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Recommendations\nWe suggest treatment for adults.\n"
                    "Figure 1. Evidence profile\nRisk ratio 0.80 95% CI 0.70 to 0.95\n"
                    "Evidence\nThe certainty of evidence was moderate."
                ),
                "tables": [],
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertTrue(output["tables"])
        self.assertIn("Figure 1", output["clean_content"])
        self.assertNotIn("Risk ratio 0.80", output["clean_content"])

    def test_flattened_inline_table_caption_does_not_crash_cleaning(self) -> None:
        for caption in ("Table 1. Baseline characteristics", "Table 1 Baseline characteristics", "Table 1."):
            with self.subTest(caption=caption):
                output = clean_source_record(
                    {
                        "content": (
                            "Background This long flattened PDF paragraph describes eligibility, interventions, "
                            f"outcomes, and implementation considerations. {caption} Outcome Benefit "
                            "Recommendations We suggest treatment for adults with disease."
                        ),
                        "tables": [],
                        "title": "Demo",
                        "source": "aasm",
                    }
                )

                self.assertIn("Table 1", output["clean_content"])
                self.assertIn("We suggest treatment", output["clean_content"])

    def test_recommendation_boxes_are_detected_and_bounded(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Background\nContext paragraph.\n"
                    "Box 1. Key recommendations\n"
                    "We recommend treatment for adults with disease.\n"
                    "Evidence\nThe certainty of evidence was high."
                ),
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertEqual(len(output["recommendation_boxes"]), 1)
        self.assertIn("We recommend treatment", output["recommendation_boxes"][0]["text"])
        self.assertGreater(output["cleaning_log"]["source_span_boundaries_detected"], 0)

    def test_plain_recommendations_section_is_not_treated_as_box(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Recommendations\n"
                    + "We suggest treatment for adults with disease. Evidence explains the rationale.\n" * 300
                    + "Evidence\nThe certainty of evidence was moderate."
                ),
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertEqual(output["recommendation_boxes"], [])
        self.assertIn("We suggest treatment", output["clean_content"])

    def test_methodology_boilerplate_is_isolated_from_clean_content(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Methods\n"
                    "The task force performed a literature search and documented conflicts of interest.\n"
                    "Recommendations\n"
                    "We suggest treatment for adults with disease."
                ),
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertNotIn("task force performed", output["clean_content"])
        self.assertEqual(len(output["methodology_blocks"]), 1)
        self.assertIn("We suggest treatment", output["clean_content"])

    def test_source_span_boundaries_include_sections_tables_and_boxes(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Background\nContext.\n"
                    "Recommendations\nWe recommend treatment.\n"
                    "Table 1. Summary\nOutcome Benefit\n"
                    "Box 2. Implementation\nClinicians should monitor symptoms."
                ),
                "tables": [],
                "title": "Demo",
                "source": "aasm",
            }
        )

        labels = " ".join(item["label"] for item in output["source_span_boundaries"])
        self.assertIn("Recommendations", labels)
        self.assertIn("Table 1", labels)
        self.assertIn("Box 2", labels)

    def test_column_merge_artifact_in_recommendation_word_is_repaired(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Recommendations\n"
                    "Recommendation 3: We suggest that clinicians use actigof 3 studies demonstrated a clinically "
                    "significant large mean raphy in the assessment of adult patients with circadian rhythm disorder."
                ),
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertIn("use actigraphy in the assessment", output["clean_content"])
        self.assertNotIn("actigof", output["clean_content"])
        self.assertGreaterEqual(output["cleaning_log"]["column_merge_repairs"], 1)

    def test_common_should_glue_artifact_is_repaired(self) -> None:
        output = clean_source_record(
            {
                "content": "Recommendations\nPrimarycareprovidersshouldconsidermaintaining follow-up for adults with disease.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertIn("providersshould consider maintaining", output["clean_content"])
        self.assertNotIn("shouldconsider", output["clean_content"])

    def test_medical_terms_are_not_split_by_glue_repair(self) -> None:
        output = clean_source_record(
            {
                "content": "Recommendations\nWe suggest polysomnography and actigraphy before electroencephalography.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        self.assertIn("polysomnography", output["clean_content"])
        self.assertIn("actigraphy", output["clean_content"])
        self.assertIn("electroencephalography", output["clean_content"])

    def test_cleaning_log_contains_deletion_audit(self) -> None:
        output = clean_source_record(
            {
                "content": "Recommendations\nWe recommend treatment.\nReferences\n1. Smith J. Lancet. 2020.",
                "title": "Demo",
                "source": "aasm",
            }
        )

        audit = output["cleaning_log"]["audit"]
        self.assertGreater(audit["raw_chars"], audit["clean_chars"])
        self.assertGreater(audit["references_chars"], 0)

    def test_empty_raw_content_is_not_removed_ratio_error(self) -> None:
        output = clean_source_record({"content": "", "title": "Empty", "source": "who"})

        audit = output["cleaning_log"]["audit"]
        self.assertIn("empty_raw_content", audit["audit_flags"])
        self.assertIn("empty_clean_content", audit["audit_flags"])
        self.assertNotIn("removed_ratio_high", audit["audit_flags"])

    def test_nice_glued_boilerplate_is_removed(self) -> None:
        output = clean_source_record(
            {
                "content": (
                    "Recommendations\nWe recommend treatment. "
                    "Seewww.nice.org.uk/guidance/NG106 "
                    "©NICE2018.Allrightsreserved.SubjecttoNoticeofrights."
                ),
                "title": "NICE demo",
                "source": "nice",
            }
        )

        self.assertNotIn("www.nice.org.uk/guidance", output["clean_content"])
        self.assertNotIn("Allrightsreserved", output["clean_content"])
        self.assertNotIn("SubjecttoNoticeofrights", output["clean_content"])

    def test_clean_source_records_is_lazy(self) -> None:
        def records():
            yield {"content": "Recommendations\nWe recommend treatment.", "title": "One", "source": "aasm"}
            raise AssertionError("second record should not be pulled before iteration continues")

        iterator = clean_source_records(records())
        first = next(iter(iterator))

        self.assertIn("We recommend treatment", first["clean_content"])


if __name__ == "__main__":
    unittest.main()

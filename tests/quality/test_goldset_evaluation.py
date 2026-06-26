import unittest
from pathlib import Path

from src.pipeline.quality.goldset import evaluate_goldset
from src.common.process_jsonl import iter_jsonl
from scripts.evaluate_goldset import failed_thresholds


ROOT = Path(__file__).resolve().parents[2]


class GoldsetEvaluationTests(unittest.TestCase):
    def test_goldset_evaluation_reports_entity_and_field_metrics(self) -> None:
        gold = [
            {
                "entity_id": "rec-1",
                "expected": {
                    "recommendation_text": "We recommend treatment.",
                    "strength": "strong",
                    "certainty": "moderate",
                    "raw_start_char": 10,
                    "raw_end_char": 33,
                },
            },
            {
                "entity_id": "rec-2",
                "expected": {
                    "recommendation_text": "We suggest CBT-I.",
                    "strength": "conditional",
                    "certainty": "low",
                },
            },
        ]
        predictions = [
            {
                "entity_id": "rec-1",
                "recommendation_text": "We recommend treatment.",
                "strength": "strong",
                "certainty": "low",
                "raw_start_char": 10,
                "raw_end_char": 33,
            },
            {
                "entity_id": "rec-extra",
                "recommendation_text": "Extra.",
            },
        ]

        report = evaluate_goldset(gold, predictions, fields=["recommendation_text", "strength", "certainty"])

        self.assertEqual(report["matched_entities"], 1)
        self.assertEqual(report["missing_prediction_ids"], ["rec-2"])
        self.assertEqual(report["extra_prediction_ids"], ["rec-extra"])
        self.assertEqual(report["entity_precision"], 0.5)
        self.assertEqual(report["entity_recall"], 0.5)
        self.assertEqual(report["field_accuracy"]["certainty"], 0.0)
        self.assertEqual(report["raw_offset_accuracy"], 1.0)

    def test_fixture_goldset_passes_exact_prediction_baseline(self) -> None:
        gold = list(iter_jsonl(ROOT / "tests" / "fixtures" / "goldset" / "recommendation_versions_gold.jsonl"))
        predictions = list(iter_jsonl(ROOT / "tests" / "fixtures" / "goldset" / "recommendation_versions_predictions.jsonl"))

        report = evaluate_goldset(gold, predictions)

        self.assertEqual(report["gold_entities"], 3)
        self.assertEqual(report["entity_precision"], 1.0)
        self.assertEqual(report["entity_recall"], 1.0)
        self.assertEqual(report["entity_f1"], 1.0)
        self.assertEqual(report["raw_offset_accuracy"], 1.0)

    def test_goldset_threshold_failures_are_explicit(self) -> None:
        report = {
            "entity_precision": 0.9,
            "entity_recall": 0.7,
            "entity_f1": 0.78,
            "field_accuracy": {"strength": 1.0, "certainty": 0.6},
        }

        failures = failed_thresholds(
            report,
            min_precision=0.85,
            min_recall=0.8,
            min_f1=0.8,
            min_field_accuracy=0.75,
        )

        self.assertEqual(failures, ["entity_recall", "entity_f1", "field_accuracy.certainty"])


if __name__ == "__main__":
    unittest.main()

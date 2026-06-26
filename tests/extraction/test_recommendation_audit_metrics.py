import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.quality.recommendation_audit import eval_audit, evaluate_thresholds, merged_thresholds


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class RecommendationAuditMetricsTests(unittest.TestCase):
    def test_audit_metrics_are_calculated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(
                root / "recommendation_candidate_audit_sample.jsonl",
                [
                    {
                        "annotation": {
                            "is_true_recommendation": True,
                            "is_complete_statement": True,
                            "is_clinical_action": True,
                            "is_guideline_meta_text": False,
                            "has_layout_contamination": False,
                            "needs_layout_repair": False,
                        }
                    },
                    {
                        "annotation": {
                            "is_true_recommendation": False,
                            "is_complete_statement": False,
                            "is_clinical_action": False,
                            "is_guideline_meta_text": True,
                            "has_layout_contamination": True,
                            "needs_layout_repair": True,
                        }
                    },
                ],
            )
            _write_jsonl(
                root / "recommendation_skipped_action_audit_sample.jsonl",
                [
                    {
                        "annotation": {
                            "contains_true_recommendation": True,
                            "is_recoverable_by_sentence_split": False,
                            "requires_layout_repair": True,
                            "is_pure_noise": False,
                            "should_route_to": "needs_layout_repair",
                        }
                    }
                ],
            )
            _write_jsonl(
                root / "recommendation_positive_anchor_sample.jsonl",
                [{"expected_min_candidates": 1, "observed_candidates": 1}],
            )

            report = eval_audit(root)
            self.assertEqual(report["candidate_precision"], 0.5)
            self.assertEqual(report["skipped_false_negative_rate"], 1.0)
            self.assertEqual(report["layout_repair_need_rate"], 1.0)
            self.assertEqual(report["positive_anchor_recall"], 1.0)

    def test_empty_annotation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(root / "recommendation_candidate_audit_sample.jsonl", [{"annotation": {}}])
            _write_jsonl(root / "recommendation_skipped_action_audit_sample.jsonl", [{"annotation": {}}])
            _write_jsonl(root / "recommendation_positive_anchor_sample.jsonl", [{"expected_min_candidates": 1}])
            with self.assertRaises(ValueError):
                eval_audit(root)

    def test_invalid_route_label_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(
                root / "recommendation_candidate_audit_sample.jsonl",
                [
                    {
                        "annotation": {
                            "is_true_recommendation": True,
                            "is_complete_statement": True,
                            "has_layout_contamination": False,
                        }
                    }
                ],
            )
            _write_jsonl(
                root / "recommendation_skipped_action_audit_sample.jsonl",
                [
                    {
                        "annotation": {
                            "contains_true_recommendation": False,
                            "requires_layout_repair": False,
                            "should_route_to": "bad_route",
                        }
                    }
                ],
            )
            _write_jsonl(root / "recommendation_positive_anchor_sample.jsonl", [{"expected_min_candidates": 1}])
            with self.assertRaises(ValueError):
                eval_audit(root)

    def test_threshold_check_reports_failures(self) -> None:
        metrics = {
            "candidate_precision": 0.5,
            "positive_anchor_recall": 1.0,
            "skipped_false_negative_rate": 0.2,
        }
        result = evaluate_thresholds(metrics, merged_thresholds(["candidate_precision>=0.8", "skipped_false_negative_rate<=0.1"]))
        self.assertFalse(result["passed"])
        failed = {item["metric"] for item in result["failed_metrics"]}
        self.assertIn("candidate_precision", failed)
        self.assertIn("skipped_false_negative_rate", failed)


if __name__ == "__main__":
    unittest.main()

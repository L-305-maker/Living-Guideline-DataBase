"""复核与发布测试文件：验证 review queue、backfill、release export 和正式发布闭包逻辑。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import tempfile
import unittest
from pathlib import Path

from src.common.process_jsonl import write_jsonl
from src.pipeline.publish.release_exporter import export_release_package


def _publish_gate() -> dict:
    return {
        "quality_status": "publishable",
        "publishable": True,
        "blocking_reasons": [],
        "warning_reasons": [],
    }


def _trace(trace_id: str, target_id: str) -> dict:
    return {
        "model_trace_id": trace_id,
        "task_type": "test",
        "method": "rule",
        "model_name": "rule",
        "input_entity_type": "block",
        "input_entity_id": "block-1",
        "input_text": "input",
        "raw_output": {},
        "parsed_output": {},
        "confidence": 1.0,
        "parameters": {},
        "token_usage": {},
        "success": True,
        "human_verified": False,
        "target_table": "test",
        "target_entity_id": target_id,
    }


class ReleaseExporterTests(unittest.TestCase):
    def test_export_filters_non_publish_ready_evidence_and_backlogs_unclosed_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "reviewed"
            source_root = root / "source"
            downstream = source_root / "downstream"
            gate = source_root / "gate"
            run_dir.mkdir()
            downstream.mkdir(parents=True)
            gate.mkdir()

            write_jsonl(
                run_dir / "recommendation_versions.jsonl",
                [
                    {
                        "recommendation_version_id": "version-1",
                        "recommendation_id": "recommendation-1",
                        "recommendation_candidate_id": "candidate-1",
                        "guideline_id": "guideline-1",
                        "version_number": "v1",
                        "recommendation_text": "We recommend treatment.",
                        "pico_id": "pico-1",
                        "record_id": "record-1",
                        "grade_candidate_id": "grade-1",
                        "quality_status": "publishable",
                        "direction": "for",
                        "strength": "strong",
                        "certainty": "moderate",
                        "source_span": "We recommend treatment.",
                        "source_span_ref": "block-1",
                        "start_char": 0,
                        "end_char": 23,
                        "change_type": "new",
                        "normalized_payload": {
                            "publish_gate": _publish_gate(),
                            "linked_evidence": {"linked_evidence_ids": ["evidence-1", "evidence-2"]},
                        },
                        "raw_payload": {},
                    },
                    {
                        "recommendation_version_id": "version-2",
                        "recommendation_id": "recommendation-2",
                        "recommendation_candidate_id": "candidate-2",
                        "guideline_id": "guideline-1",
                        "version_number": "v1",
                        "recommendation_text": "We suggest another treatment.",
                        "pico_id": "pico-2",
                        "record_id": "record-1",
                        "grade_candidate_id": "grade-2",
                        "quality_status": "publishable",
                        "direction": "for",
                        "strength": "conditional",
                        "certainty": "low",
                        "source_span": "We suggest another treatment.",
                        "source_span_ref": "block-2",
                        "start_char": 30,
                        "end_char": 59,
                        "change_type": "new",
                        "normalized_payload": {
                            "publish_gate": _publish_gate(),
                            "linked_evidence": {"linked_evidence_ids": ["evidence-3"]},
                        },
                        "raw_payload": {},
                    },
                ],
            )
            write_jsonl(
                run_dir / "evidence_items.jsonl",
                [
                    {
                        "evidence_id": "evidence-1",
                        "paper_id": "paper-1",
                        "pico_id": "pico-1",
                        "recommendation_candidate_id": "candidate-1",
                        "screening_status": "included",
                        "effect_direction": "benefit",
                        "source_span": "trial evidence",
                        "normalized_payload": {},
                    },
                    {
                        "evidence_id": "evidence-2",
                        "paper_id": "paper-1",
                        "pico_id": "pico-1",
                        "recommendation_candidate_id": "candidate-1",
                        "screening_status": "association_review",
                        "effect_direction": "uncertain",
                        "source_span": "uncertain evidence",
                        "normalized_payload": {},
                    },
                    {
                        "evidence_id": "evidence-3",
                        "paper_id": "paper-1",
                        "pico_id": "",
                        "recommendation_candidate_id": "candidate-2",
                        "screening_status": "included",
                        "effect_direction": "uncertain",
                        "source_span": "missing pico evidence",
                        "normalized_payload": {},
                    },
                ],
            )
            write_jsonl(
                run_dir / "pico_questions.jsonl",
                [
                    {"pico_id": "pico-1", "guideline_id": "guideline-1", "clinical_question": "Q1", "population": "P", "intervention": "I", "status": "under_review"},
                    {"pico_id": "pico-2", "guideline_id": "guideline-1", "clinical_question": "Q2", "population": "P", "intervention": "I", "status": "under_review"},
                ],
            )
            write_jsonl(
                run_dir / "recommendation_candidates.jsonl",
                [
                    {"candidate_id": "candidate-1", "model_trace_id": "trace-rec-1", "statement": "We recommend treatment.", "guideline_id": "guideline-1", "record_id": "record-1", "status": "accepted", "normalized_payload": {}},
                    {"candidate_id": "candidate-2", "model_trace_id": "trace-rec-2", "statement": "We suggest another treatment.", "guideline_id": "guideline-1", "record_id": "record-1", "status": "accepted", "normalized_payload": {}},
                ],
            )
            write_jsonl(
                run_dir / "grade_candidates.jsonl",
                [
                    {"grade_candidate_id": "grade-1", "model_trace_id": "trace-grade-1", "source_grade_raw": "strong", "recommendation_candidate_id": "candidate-1", "guideline_id": "guideline-1", "status": "accepted", "normalized_payload": {}},
                    {"grade_candidate_id": "grade-2", "model_trace_id": "trace-grade-2", "source_grade_raw": "conditional", "recommendation_candidate_id": "candidate-2", "guideline_id": "guideline-1", "status": "accepted", "normalized_payload": {}},
                ],
            )
            write_jsonl(
                source_root / "all_cleaned.jsonl",
                [
                    {
                        "record_id": "record-1",
                        "content": "clean text",
                        "guideline_seed": {"guideline_id": "guideline-1", "title": "Guideline"},
                        "paper_seed": {"paper_id": "paper-1", "title": "Paper"},
                        "direct_extraction": {"guideline_id": "guideline-1", "paper_id": "paper-1"},
                    }
                ],
            )
            write_jsonl(gate / "all_cleaned.ready.jsonl", [{"record_id": "record-1"}])
            write_jsonl(gate / "all_cleaned.needs_layout_repair.jsonl", [])
            write_jsonl(gate / "all_cleaned.parse_failed.jsonl", [])
            write_jsonl(downstream / "recommendation_traces.jsonl", [_trace("trace-rec-1", "candidate-1"), _trace("trace-rec-2", "candidate-2")])
            write_jsonl(downstream / "grade_traces.jsonl", [_trace("trace-grade-1", "grade-1"), _trace("trace-grade-2", "grade-2")])

            result = export_release_package(
                run_dir,
                source_root=source_root,
                publish_dir=run_dir / "publish_ready_v1",
                backlog_dir=run_dir / "review_backlog_v1",
                published_at="2026-01-01T00:00:00+00:00",
            )

            self.assertTrue(result["formal_gate_report"]["passed"])
            self.assertEqual(result["formal_gate_report"]["release_counts"]["recommendation_versions"], 1)
            self.assertEqual(result["formal_gate_report"]["release_counts"]["evidence_items"], 1)
            self.assertEqual(result["formal_gate_report"]["release_counts"]["recommendation_version_evidence_links"], 1)
            released_versions = list((run_dir / "publish_ready_v1" / "recommendation_versions.jsonl").read_text(encoding="utf-8").splitlines())
            self.assertEqual(len(released_versions), 1)
            self.assertIn("evidence-1", released_versions[0])
            self.assertNotIn("evidence-2", released_versions[0])
            evidence_links = (run_dir / "publish_ready_v1" / "recommendation_version_evidence_links.jsonl").read_text(encoding="utf-8")
            self.assertIn("version-1", evidence_links)
            self.assertIn("evidence-1", evidence_links)
            excluded_versions = (run_dir / "review_backlog_v1" / "excluded_recommendation_versions.jsonl").read_text(encoding="utf-8")
            self.assertIn("version-2", excluded_versions)
            excluded_evidence = (run_dir / "review_backlog_v1" / "excluded_evidence_items.jsonl").read_text(encoding="utf-8")
            self.assertIn("evidence-2", excluded_evidence)
            self.assertIn("evidence-3", excluded_evidence)


if __name__ == "__main__":
    unittest.main()


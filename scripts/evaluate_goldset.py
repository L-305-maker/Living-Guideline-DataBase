from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.quality.goldset import evaluate_goldset_file


JsonDict = Dict[str, Any]


def _write_json(path: str | Path, payload: JsonDict) -> None:
    """写入可读 JSON 报告，供人工查看 goldset 阈值门禁结果。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def failed_thresholds(
    report: JsonDict,
    *,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    min_field_accuracy: float,
) -> list[str]:
    """根据调用方设置的阈值返回失败指标名称，便于 CI 或脚本直接定位问题。"""

    failures: list[str] = []
    if float(report.get("entity_precision") or 0.0) < min_precision:
        failures.append("entity_precision")
    if float(report.get("entity_recall") or 0.0) < min_recall:
        failures.append("entity_recall")
    if float(report.get("entity_f1") or 0.0) < min_f1:
        failures.append("entity_f1")
    field_accuracy = report.get("field_accuracy")
    if isinstance(field_accuracy, dict):
        for field, value in field_accuracy.items():
            if float(value or 0.0) < min_field_accuracy:
                failures.append(f"field_accuracy.{field}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用人工 goldset 评估推荐预测结果。")
    parser.add_argument("--gold-input", required=True)
    parser.add_argument("--predictions-input", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-f1", type=float, default=0.0)
    parser.add_argument("--min-field-accuracy", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    """命令行入口：生成 goldset 指标报告，并在低于阈值时返回非零退出码。"""

    args = parse_args()
    report = evaluate_goldset_file(args.gold_input, args.predictions_input, args.report_output)
    failures = failed_thresholds(
        report,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
        min_field_accuracy=args.min_field_accuracy,
    )
    gate_report = {
        "gold_input": args.gold_input,
        "predictions_input": args.predictions_input,
        "report_output": args.report_output,
        "threshold_failures": failures,
        "passed": not failures,
        "metrics": report,
    }
    _write_json(Path(args.report_output).with_suffix(".gate.json"), gate_report)
    print(
        "gold_entities={gold_entities} predicted_entities={predicted_entities} precision={entity_precision} recall={entity_recall} f1={entity_f1} passed={passed}".format(
            passed=not failures,
            **report,
        )
    )
    if failures:
        print("goldset threshold failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

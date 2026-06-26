from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TEST_GROUPS: Dict[str, List[str]] = {
    "cleaning": [
        "tests.cleaning.test_cleaning_quality_gate",
        "tests.cleaning.test_guideline_cleaning",
        "tests.cleaning.test_offset_mapping",
        "tests.cleaning.test_record_quality_context",
        "tests.cleaning.test_record_type_routing",
    ],
    "extraction": [
        "tests.extraction.test_evidence_quality_gate",
        "tests.extraction.test_flow_certainty_rules",
        "tests.extraction.test_grade_association",
        "tests.extraction.test_pico_quality_gate",
        "tests.extraction.test_recommendation_audit_metrics",
        "tests.extraction.test_recommendation_layout_repair_queue",
        "tests.extraction.test_recommendation_quality_gate",
        "tests.extraction.test_recommendation_statement_quality",
        "tests.extraction.test_source_span_first_extraction",
    ],
    "versioning": [
        "tests.versioning.test_recommendation_current_state",
        "tests.versioning.test_recommendation_versioning",
        "tests.versioning.test_version_diff_update_logs",
    ],
    "review_publish": [
        "tests.review_publish.test_manual_review_roundtrip",
        "tests.review_publish.test_review_batch",
        "tests.review_publish.test_recommendation_publisher",
        "tests.review_publish.test_recommendation_version_review_queue",
        "tests.review_publish.test_rule_assisted_backfill",
    ],
    "storage": [
        "tests.storage.test_atomic_ingestion",
        "tests.storage.test_pre_ingest_check",
        "tests.storage.test_postgres_publish_smoke",
        "tests.storage.test_storage_contract_check",
        "tests.storage.test_storage_contract",
        "tests.storage.test_storage_safety",
        "tests.storage.test_pgvector_ingest",
    ],
    "orchestration": [
        "tests.orchestration.test_pipeline_e2e",
        "tests.orchestration.test_pipeline_orchestration",
        "tests.orchestration.test_public_package_api",
    ],
    "quality": [
        "tests.quality.test_data_artifact_manifest",
        "tests.quality.test_goldset_evaluation",
    ],
    "vectorization": [
        "tests.vectorization.test_embedding_queue",
    ],
}


def modules_for_groups(groups: Iterable[str]) -> List[str]:
    """根据测试分组名称返回测试模块列表。

    这里显式维护模块清单，而不是依赖 `unittest discover` 自动扫描。这样做的好处是：
    1. 维护者打开一个文件就能看到项目测试版图；
    2. CI 或本地调试可以按领域分组运行；
    3. live PostgreSQL smoke test 等特殊测试仍然留在 storage 分组中，由测试自身决定是否跳过。
    """

    modules: List[str] = []
    for group in groups:
        if group not in TEST_GROUPS:
            raise ValueError(f"未知测试分组: {group}")
        modules.extend(TEST_GROUPS[group])
    return modules


def load_suite(groups: Iterable[str] | None = None) -> unittest.TestSuite:
    """集中加载项目测试套件。

    默认加载全部分组；传入 `--group storage` 等参数时只加载指定分组。
    质量门脚本也调用这个入口，避免不同命令加载到不同测试集合。
    """

    selected_groups = list(groups or TEST_GROUPS)
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module_name in modules_for_groups(selected_groups):
        suite.addTests(loader.loadTestsFromName(module_name))
    return suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一测试入口：按领域分组运行项目测试。")
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted(TEST_GROUPS),
        help="只运行指定测试分组；可重复传入。不传时运行全部分组。",
    )
    parser.add_argument("-v", "--verbosity", type=int, default=1, help="unittest 输出详细程度。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_suite(args.group)
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

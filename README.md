# Data Splitting Ingestion

面向医学指南和生物医学文献的结构化推荐抽取、证据关联、复核发布与 PostgreSQL 入库项目。

项目的长期研究目标见 [docs/project_memory.md](docs/project_memory.md)：构建 **Evidence-Grounded Adaptive Retrieval-Reasoning System for Biomedical Recommendation Extraction**。当前代码库是这个目标的工程底座，重点解决从噪声 PDF/文本中抽取可追溯的 `RecommendationVersion`、`PICOQuestion`、`EvidenceItem`、GRADE 信息，并通过发布闸门进入正式知识库。

## 当前正式发布状态

最近一次正式发布包：

```text
data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\publish_ready_v1
```

对应 review backlog：

```text
data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\review_backlog_v1
```

当前 `publish_ready_v1` 已通过 formal gate 和 storage contract。正式包规模：

| Entity | Rows |
| --- | ---: |
| cleaned_records | 77 |
| guidelines | 77 |
| papers | 77 |
| model_traces | 278 |
| recommendation_candidates | 139 |
| grade_candidates | 139 |
| pico_questions | 123 |
| recommendation_versions | 139 |
| recommendations | 139 |
| evidence_items | 409 |
| recommendation_version_evidence_links | 424 |
| update_logs | 139 |

这不是把 170 条候选 `RecommendationVersion` 全量发布，而是经过 manifest-scoped closure gate 后的正式闭包：

- 只发布 `quality_status = publishable` 的版本。
- 只保留 `screening_status = included` 等 publish-ready evidence。
- 每条正式 `RecommendationVersion` 必须有 grade、PICO、source span 和 evidence link。
- `association_review`、`uncertain`、`pending` evidence 留在 backlog，不进入正式入库和默认 RAG。

## 快速开始

### 1. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

如使用 `pyproject.toml` 的开发安装：

```powershell
python -m pip install -e .
```

### 2. 配置 PostgreSQL

PowerShell 示例：

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
```

### 3. 重新生成正式发布包

```powershell
python -B -X utf8 -m src.pipeline.publish.release_exporter `
  --run-dir data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1 `
  --source-root data\processed\cleaning_repair_eval_20260626 `
  --publish-dir data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\publish_ready_v1 `
  --backlog-dir data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\review_backlog_v1
```

### 4. 入库前检查

```powershell
python -B -X utf8 -m src.storage.storage_PG validate-run `
  --run-dir data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\publish_ready_v1 `
  --output data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\publish_ready_v1\storage_contract_report.jsonl
```

### 5. 重建项目表并入库正式包

```powershell
python -B -X utf8 -m src.storage.storage_PG ingest-release `
  --release-dir data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\publish_ready_v1 `
  --recreate
```

`--recreate` 会删除并重建本项目管理的 Living-Guideline 表。不要在生产库或混合业务库上随意使用。

### 6. 复核数据库状态

```powershell
python -B -X utf8 -m src.storage.storage_PG counts
python -B -X utf8 -m src.storage.storage_PG integrity-report
```

当前健康目标：

```text
orphan_counts = 0
versions_without_grade = 0
versions_without_pico = 0
versions_without_source_span = 0
versions_without_evidence_link = 0
backlog_status_in_evidence = 0
```

## 项目结构

```text
src/
  common/          JSONL、文本标准化、通用工具
  domain/          领域实体定义，如 RecommendationVersion、EvidenceItem
  ingestion/       爬虫、PDF 下载、原始数据处理
  pipeline/
    cleaning/      PDF/文本清洗、质量 gate、offset 映射
    parsing/       文档结构解析、block 生成
    extraction/    Recommendation、GRADE、PICO、Evidence 抽取
    review/        人工/规则辅助复核与关联增强
    publish/       发布包导出、正式发布 bundle 构建
    quality/       质量报告、goldset 评估
    update/        版本差异与 update log
  storage/         PostgreSQL schema、contract check、入库 CLI
  vectorization/   后续向量化队列入口

scripts/           本地运行、审计、质量检查脚本
tests/             单元测试、流程测试、storage/publish 测试
docs/              项目记忆、研究目标和设计说明
data/              本地数据与 pipeline 产物，默认不纳入 Git
```

## 核心数据边界

项目里容易混淆的是“候选数据”和“正式发布数据”的边界：

| Layer | Meaning | Default Use |
| --- | --- | --- |
| `recommendation_candidates` | 抽取出来的推荐语句候选 | 复核、构建版本 |
| `grade_candidates` | 证据等级/推荐强度候选 | 复核、构建版本 |
| `pico_questions` | PICO 问题结构 | 正式包只保留 active 闭包 |
| `evidence_items` | 证据条目 | 正式包只保留 publish-ready evidence |
| `recommendation_versions` | 不可变推荐版本 | 只有 publishable 进入正式知识库 |
| `recommendations` | 当前正式推荐状态 | 由发布流程生成 |
| `recommendation_version_evidence_links` | RV 与 evidence 的显式多对多关系 | 检索、审计、RAG grounding |
| `review_backlog_v1` | 待复核或不闭合数据 | 不进入默认 RAG |

正式入库必须使用 `publish_ready_v1`，不要直接把上游全量 `evidence_items.jsonl` 或 `recommendation_versions.jsonl` 入库。

## 常用命令

### 全量测试

```powershell
python -B -X utf8 -m unittest discover -s tests
```

### Ruff 检查

```powershell
ruff check src tests
```

### 项目质量脚本

```powershell
python scripts\check_quality.py --with-ruff
```

### 查看当前正式发布 gate

```powershell
Get-Content data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\publish_ready_v1\formal_gate_report.json
```

### 查看 backlog 摘要

```powershell
Get-Content data\processed\cleaning_repair_eval_20260626\reviewed_association_processed_v1\review_backlog_v1\backlog_summary.json
```

## 当前 backlog 现状

`review_backlog_v1` 保存正式发布外的数据。最近一次报告中：

- 排除的 `RecommendationVersion`：31
- 从正式版本证据链接中排除的 `EvidenceItem`：84
- 主要原因：
  - `source_record_failed_cleaning_gate`: 30
  - `source_record_not_in_ready_gate`: 30
  - `version_without_publish_ready_evidence`: 1
  - evidence 侧还有 `association_review`、`uncertain`、missing PICO、PICO mismatch 等问题

这部分数据不是废数据，而是后续人工/LLM 复核和清洗修复的工作队列。

## 数据清理原则

可以安全删除：

- `__pycache__/`
- `.ruff_cache/`
- `.vs/`
- 已确认和基础文件完全一致的重复 `.reviewed.jsonl`

不要随意删除：

- `publish_ready_v1/`
- `review_backlog_v1/`
- `formal_gate_report.json`
- `release_manifest.json`
- `storage_contract_report.jsonl`
- `docs/project_memory.md`
- `tests/`

## 质量与发布原则

1. 清洗层负责尽量恢复可读文本，但不负责“发明”结构化事实。
2. 抽取层生成候选，候选不等于正式知识。
3. 复核层处理弱关联、冲突、缺失 PICO/证据等问题。
4. 发布层只接收闭合、可追溯、publishable 的版本。
5. 入库层必须保留外键闭包和 RV-Evidence 多对多关系。
6. 默认检索/RAG 只能读取 `publish_ready_v1` 或正式库中的 publish-ready 表。

## 后续优先级

1. 继续处理 `review_backlog_v1` 中的 31 条版本和 evidence association 问题。
2. 为 NICE/WHO/USPSTF 等来源建立更强的 source-specific 清洗与推荐框识别规则。
3. 扩充人工 goldset，单独评估 Recommendation、PICO、GRADE、Evidence linking。
4. 将 `RecommendationVersion`、`PICOQuestion`、`EvidenceItem` 作为第一批向量化对象，排除 backlog。
5. 中期引入更正式的数据库迁移管理，替代长期手写 DDL/migration list。

## 最近验证结果

最近一次本地验证：

```text
python -B -X utf8 -m unittest discover -s tests
Ran 161 tests, OK, skipped=2

ruff check src tests
All checks passed
```

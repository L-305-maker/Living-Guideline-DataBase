# Pipeline 数据流说明

`src/pipeline/` 负责把原始 JSONL 数据处理成可审核、可发布的 Living-Guideline 结构化产物。这里的核心原则是：

- `orchestration/run_pipeline.py` 只生成候选产物、质量报告和审核队列。
- `publish/recommendation_publisher.py` 才负责把审核通过的版本发布为正式当前推荐。
- 普通抽取流程不要直接写 `recommendations` 当前态，避免未审核候选污染正式知识库。

## 主数据流图

```mermaid
flowchart TD
    A["origin JSONL<br/>原始指南/论文记录"] --> B["cleaning/source_cleaner.py<br/>清洗文本、表格、引用、清洗日志"]
    B --> C["cleaning/quality_gate.py<br/>清洗质量门：ready / repair / failed / skipped"]
    C --> D["parsing/structure_parser.py<br/>切分 SourceBlock，保留章节和 offset 线索"]
    D --> E["extraction/routing/candidate_router.py<br/>按块内容路由到推荐、GRADE、PICO、证据任务"]

    E --> F["recommendation/candidate_extractor.py<br/>生成 RecommendationCandidate + ModelTrace"]
    E --> G["grade/candidate_extractor.py<br/>生成 GradeCandidate + ModelTrace"]
    E --> H["pico/question_extractor.py<br/>生成 PicoQuestion + ModelTrace"]
    E --> I["evidence/item_extractor.py<br/>生成 EvidenceItem + ModelTrace"]

    F --> J["versioning/recommendation_version_builder.py<br/>生成 RecommendationVersion 候选版本和 publish_gate"]
    G --> J
    H --> J
    I --> J

    J --> K["update/version_diff.py<br/>生成版本差异候选报告"]
    J --> L["llm_review/queue/builder.py<br/>构建 LLM/人工复核队列"]
    J --> M["quality/reports/*<br/>候选质量报告和失败样本"]

    L --> N["review/manual_review_gate.py<br/>通用候选复核和回写"]
    J --> O["publish/recommendation_publisher.py<br/>仅发布 publishable 或 approved review item"]
    O --> P["Recommendation 当前态<br/>正式可查询推荐"]
    O --> Q["UpdateLog<br/>正式发布更新事件"]
```

## 候选生成与正式发布的边界

```mermaid
flowchart TD
    A["run_pipeline.py<br/>候选生成"] --> B["recommendation_versions.jsonl<br/>可能包含 blocked / needs_review / publishable"]
    B --> C["generic_ingest.py<br/>发布门禁分流"]
    C -->|publishable| D["recommendation_versions<br/>正式版本表"]
    C -->|blocked / needs_review| E["recommendation_version_review_queue<br/>发布审核队列"]
    E --> F["人工审核 approved"]
    F --> G["recommendation_publisher.py --ingest<br/>原子发布"]
    G --> H["recommendations.current_version_id<br/>当前正式推荐"]
    G --> I["update_logs<br/>正式发布事件"]
    G --> J["review_status = published"]
```

## 包结构

- `cleaning/`：来源清洗、PDF 噪声处理、清洗质量门。
- `parsing/`：指南 profile 识别、结构块解析。
- `extraction/`：规则抽取主包。
  - `common/`：抽取层共享枚举、校验和映射。
  - `routing/`：SourceBlock 到任务输入的分流。
  - `recommendation/`：推荐语句候选抽取。
  - `pico/`：PICO 问题抽取。
  - `grade/`：GRADE 候选和推荐关联。
  - `evidence/`：证据条目和关联筛选。
  - `enhancement/`：LLM 后处理增强。
  - `versioning/`：不可变推荐版本和当前态构建。
  - `source_span/`：以源文本 span 为优先的实验抽取路径。
- `llm_review/`：LLM 复核队列、prompt、响应解析、auto QC。
- `review/`：通用人工复核队列构建和回写。
- `publish/`：正式发布编排，把 approved 版本写入当前态和更新日志。
- `quality/`：质量报告、goldset 评估和失败样本输出。
- `orchestration/`：端到端候选生成流水线入口。

## 维护规则

1. 修改抽取规则时，必须补充或更新对应测试 fixture。
2. 修改 `RecommendationVersion` 生成逻辑时，必须检查 `publish_gate` 是否仍能拦截低质量版本。
3. 修改发布流程时，必须保证 version、recommendation、update_log、review queue 状态同事务。
4. 不要把正式发布逻辑塞进 `run_pipeline.py`。
5. 移动模块时，同步更新 Python import、PowerShell 脚本和本 README 的数据流图。

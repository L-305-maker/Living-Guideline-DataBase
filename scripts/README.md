# scripts/

`scripts/` 是数据生产、OCR 调度、语料修复、远程部署等**离线运维入口**的集合。它不参与运行时的检索 / MCP 服务；调用的是 `src/` 下的库代码。

按数据生命周期分为 5 组：PDF 分类 → OCR 候选与修复 → 语料修复 → 结构化抽取（IDSA）→ 远程部署。

---

## 1. PDF 分类

### `classify_guideline_pdfs.py`

扫描 `data/raw_pdf` 下的所有 PDF，用关键词判定是否医学指南，识别后**移动**非指南文件到 `data/excluded/`。

- 文本提取：`pymupdf`（PyMuPDF）
- 关键词匹配：英文（guideline / recommendation / consensus / clinical practice …）+ 中文（指南 / 共识 / 推荐意见 / 中华医学会 …）；非指南关键词（annual report / 招聘 / 捐赠 …）
- 判决：`guideline` / `non_guideline` / `unknown`
- 输出：`reports/guideline_classify.jsonl`

```bash
python -B scripts/classify_guideline_pdfs.py --root data/raw_pdf            # dry-run
python -B scripts/classify_guideline_pdfs.py --root data/raw_pdf --apply    # 移动 non_guideline
```

---

## 2. OCR 候选构建、调度与质检

这组脚本围绕 MinerU + 百度 OCR + DeepSeek OCR 三个引擎，构成「扫描版 PDF」的修复流水线。默认数据目录是 `data/evidence_candidate/`，主库 `data/evidence/` 不会被污染。

### `rebuild_evidence_candidates.py`

从两份审计报告挑出需要 OCR 修复的 PDF，构建**隔离候选库**。

- 输入：`data/reports/missing_full_content_candidates_20260720.csv`（缺正文）+ `outputs/markdown_clean_mojibake_files_20260719_complete.csv`（乱码）+ `data/evidence/source_pdf_inventory.jsonl`
- 按 sha256 去重；按 cache_doc_id 判断已 OCR 页数
- 决策 action：`rebuild_from_ocr_cache` / `pending_baidu_ocr` / `pending_deepseek_ocr` / `reuse_canonical_skip_reextract` …
- `--ocr-provider baidu|deepseek`：`--execute-ocr --max-api-pages N --ocr-workers N` 真的去调 OCR API
- `--apply` 才写 `markdown_raw` 并重建 chunks
- 错误处理：百度配额耗尽 / DeepSeek 失败 → 自动转 pending，记录 `quota_exhausted` / `provider_halted`

```bash
python -B scripts/rebuild_evidence_candidates.py                          # 仅生成候选清单
python -B scripts/rebuild_evidence_candidates.py --apply --execute-ocr \
    --ocr-provider baidu --max-api-pages 5000 --ocr-workers 2
```

### `run_mineru_inventory_batch.py`

本地 GPU 上批量跑 **MinerU OCR**，按页数贪心分片，自动跳过已完成的 doc_id。

- 输入：`data/evidence_candidate/candidate_inventory.csv`
- 已用 DeepSeek 或之前 MinerU 批次完成的 doc_id 自动跳过
- `--shards N`（默认 20）按总页数贪心均分
- 通过 `os.link` 硬链接源 PDF 到 `run_inputs/<run_id>/shard_NN/`
- GPU 保护环境变量：`CUDA_VISIBLE_DEVICES=0`、`MINERU_PROCESSING_WINDOW_SIZE=1`、`MINERU_API_MAX_CONCURRENT_REQUESTS=1` …
- 调 `mineru.exe -p ... -o ... -b pipeline -m ocr -l ch`
- 输出 `manifest.csv` + `ocr_quality.csv` + per-shard 日志
- 不带 `--run` 只生成 manifest（可中断续跑）

```bash
python -B scripts/run_mineru_inventory_batch.py                          # 仅 manifest
python -B scripts/run_mineru_inventory_batch.py --run --shards 20
```

### `run_mineru_local.ps1`

PowerShell 版本，单文件或单目录跑 MinerU，无分片/manifest 逻辑。适合快速测试单个 PDF。

- 必传 `-InputPath`（PDF 或目录），可选 `-OutputPath`
- 强校验 `.venv-mineru-gpu/Scripts/mineru.exe` 与 `data/mineru/mineru.json` 存在
- 同设置 GPU 保护环境变量

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mineru_local.ps1 `
    -InputPath data\raw_pdf\scan\foo.pdf -OutputPath data\evidence_candidate\mineru_local
```

### `audit_mineru_quality.py`

审核 MinerU 批次输出，决定哪些需要重新 OCR。

- 读 `candidate_inventory.csv` + MinerU `output/<doc_id>/ocr/` 下的 `.md`、`_content_list.json`、`_content_list_v2.json`、`_model.json`
- 用 `pypdfium2` 读源 PDF 页数做交叉校验
- 统计指标：页数对齐、可见字符、每页字符（median / p10）、空 body 页、短 body 页（<30 字符）、图片/表格页、标题数、坏字符 `\ufffd` / 控制字符、重复行比例
- 复用 `assess_cleaned_body` 复用清洗器质量评分
- 三档判决：`reocr_recommended` / `manual_review` / `pass`
- 输出：`quality_review_20260721.csv` + `quality_review_reocr.csv` + `quality_review_manual.csv` + `quality_review_summary_*.json`

```bash
python -B scripts/audit_mineru_quality.py \
    --inventory data/evidence_candidate/candidate_inventory.csv \
    --batch-dir data/evidence_candidate/mineru_inventory_batch_001
```

### `promote_ocr_candidates.py`

OCR 候选 Markdown 通过质检后，**晋升** 到正式 `data/evidence_candidate/`。带 staging → backup → 原子切换。

- 用 **fasttext**（`data/language_models/lid.176.ftz`）语言识别；CJK 字符 ≥ 20% 直接判 zh
- 仅主语言 `en` / `zh` 的文档进入 `candidate_processing_inventory.csv`；其他进 `candidate_language_exclusions.csv`
- `link_assets()` 把 MinerU `images/` 软链到 `markdown_raw/assets/<doc_id>/`
- `--apply` 时：
  1. staging 目录 `.ocr_promotion_stage_<run_id>/` 内重跑 clean + document_views + chunks
  2. 校验 documents / cards / chunks 数量一致
  3. 把 `data/evidence_candidate/` 下 7 个产物**移动**到 `backups/ocr_promotion_<run_id>/`
  4. staging 产物移过来
- 路径保护：拒绝在 `data/evidence_candidate/` 外 promote

```bash
python -B scripts/promote_ocr_candidates.py                                   # dry-run
python -B scripts/promote_ocr_candidates.py --apply
```

---

## 3. 语料修复（evidence 主库）

### `repair_evidence_corpus.py`

审查 `data/evidence/` 中每个源 PDF 的处置是否明确，并补齐 / 隔离异常项。

- 读 `data/reports/raw_pdf_source_department_20260714/final/all_pdf_source_department.csv` 作为 prior review
- 按 sha256 + review 状态判定：
  - `active`（已有 doc_id）
  - `duplicate_pdf_content`（hash 已在 active 集合）
  - `confirmed_unprocessed`（review 标 confirmed_signal 但还没入）
  - `excluded_clear_non_guideline` / `review_required` / `review_required_no_prior_review`
- `--apply` 时：
  - 多进程跑 `helper_convert_pdf_worker` 转换 confirmed_unprocessed
  - 重跑 clean + encode + document_repr + chunks
  - 把 `documents_excluded.jsonl` 里记录的原始 MD 移到 `quarantine/corpus_repair_<run_id>/markdown_raw/`
  - 同步 `documents_raw.jsonl`
- 失败抛 RuntimeError，防止带未处理 PDF 进下游

```bash
python -B scripts/repair_evidence_corpus.py                              # 仅审计
python -B scripts/repair_evidence_corpus.py --apply --workers 4 --ocr-mode never
```

### `build_evidence_last.py`

从审核过的 guideline + consensus PDF 构建 **最终数据集** `data/evidence_last/`，三阶段流水线。

| 阶段 | 命令 | 内容 |
|---|---|---|
| `prepare` | `python -B scripts/build_evidence_last.py prepare --workers 4` | 多线程 PDF→MD；`route=direct_text` 立即产出；`route=mineru` 写 `mineru_inventory.csv` 待后续 OCR |
| `finalize` | `python -B scripts/build_evidence_last.py finalize` | 包装 MinerU 产物 → clean → encode → document_repr → chunks → validate |
| `validate` | `python -B scripts/build_evidence_last.py validate` | 校验 documents / cards / sections / chunks 文件数一致；`document_kind` 仅 `guideline` / `consensus` |

- 复用 `selected_rows()` 选择 `audit_decision=medical_guideline` 或 hardcoded consensus 文件
- 按 sha256 排除已 active 的重复 PDF

---

## 4. 结构化抽取（IDSA 子集专用）

### `run_idsa_full_information.py`

对 **IDSA（美国感染病学会）指南** 做端到端的「指南信息抽取」，输出 `information/IDSA/`。

- 流程：PDF → markdown_raw → markdown_clean → documents → 过滤 → sections → chunks → candidates → 调 LLM 抽取 → 验证
- **过滤规则**（`exclusion_reason()`）：
  - `JAPANESE_GUIDELINE_EXCLUDED`（含日文假名）
  - `POOR_TEXT_QUALITY_EXCLUDED`
  - `OCR_REQUIRED_EXCLUDED`
- 调 `CandidateBuilder().build(sections, run_id)` 构建抽取候选
- 调 `run_pilot(pilot_id="idsa_full_v1", client_name="openai-compatible", ...)` 跑 LLM 抽取 + 验证（依赖 `.env` 里的 `GUIDELINE_LLM_*` 配置）
- `--model-only`：跳过 PDF→chunks，只对已存在的 `candidates.jsonl` 跑模型
- `--force`：备份 `evidence/` 为 `evidence.backup/` 再重建
- `--resume` 支持 LLM 阶段断点续跑
- 产物：`extraction_results.jsonl` / `verification_results.jsonl` / `validation_results.jsonl` / `route_results.jsonl` / `model_responses.jsonl` + failure 文件

```bash
python -B scripts/run_idsa_full_information.py --workers 4 --max-items 200
python -B scripts/run_idsa_full_information.py --model-only --resume
```

### `export_idsa_simplified_results.py`

把 IDSA 集成抽取文件简化成两个便于分析的 JSONL。

- 输入：`information/IDSA/integrated/idsa_information_integrated_v1.jsonl`
- 输出（在 `information/IDSA/final/`）：
  - `idsa_recommendations_minimal.jsonl`：仅 9 个核心字段（`recommendation_text` / `direction` / `strength` / `certainty` / `population` / `interventions` / `dosage` / `duration` / `conditions`）+ 路由 / 验证状态
  - `idsa_recommendations_review.jsonl`：minimal + `candidate_text` / `context_before` / `context_after` + `is_hard_negative`
- 简化 status：extraction 存在 → `EXTRACTED`，否则取 `integration_status`
- 写 `idsa_recommendations_simplified_manifest.json` 记录计数与说明

---

## 5. 远程部署

### `remote_pg_build.py`

通过 SSH 在 **远端服务器** 上跑 PostgreSQL 入库 + 向量化 + 索引 + 验证。密码不传本地，全部从远端 `/etc/pdf-markdown-rag/postgres.env` 读取。

- 通过 `ssh <alias> bash -s -- <args...>` 把内嵌的 REMOTE_SCRIPT 推到远端执行
- 远端依次：
  1. `init --with-vector`（建表 + pgvector 扩展）
  2. `ingest --data-dir ... --batch-size 500`（JSONL → PostgreSQL，除非 `--skip-ingest`）
  3. 三个 `bge_m3_vectorize`（document_cards / document_views / chunks，分页）
  4. `index-vectors`（创建 IVFFlat）
  5. `verify --model Qwen/Qwen3-Embedding-8B`
- 默认：项目目录 `/home/lhj/project/evidence_generation`、模型 `Qwen/Qwen3-Embedding-8B`、HF cache `/data/lhj/huggingface`、PG env `/etc/pdf-markdown-rag/postgres.env`
- `--dry-run` 只打印 ssh 命令不连接
- `--skip-ingest`：复用现有 PG 数据，仅补缺失向量（**仅在数据快照未变时使用**）

```bash
python -B scripts/remote_pg_build.py your-ssh-alias --dry-run
python -B scripts/remote_pg_build.py your-ssh-alias
python -B scripts/remote_pg_build.py your-ssh-alias --skip-ingest     # 补齐缺失向量
```

---

## 一张图看脚本在数据流中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│ ① PDF 分类                                                     │
│    classify_guideline_pdfs.py                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ ② OCR 候选与修复（不影响主库）                                  │
│    rebuild_evidence_candidates.py   → candidate_inventory.csv   │
│    run_mineru_inventory_batch.py / run_mineru_local.ps1         │
│    audit_mineru_quality.py         → 质检报告                    │
│    promote_ocr_candidates.py       → staging → evidence_candidate│
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ ③ evidence 主库语料修复                                        │
│    repair_evidence_corpus.py      → quarantine + 重跑 clean      │
│    build_evidence_last.py         → evidence_last/ 最终数据集   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ ④ PostgreSQL + 向量 + 远程部署                                │
│    remote_pg_build.py             → ssh + ingest + vectorize    │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ ⑤ 结构化抽取（仅 IDSA 子集）                                  │
│    run_idsa_full_information.py   → information/IDSA/           │
│    export_idsa_simplified_results.py → minimal + review         │
└─────────────────────────────────────────────────────────────────┘
```

## 通用约定

- 所有脚本默认从仓库根目录运行（`ROOT = Path(__file__).resolve().parents[1]`），并将项目根加入 `sys.path`
- 数据契约、ID、`text_for_embedding` 规则与 `src/` 共享——修改 `src/` 字段后需同步重建对应下游 JSONL
- OCR 凭据、API key、HF cache 路径只通过环境变量或 `deploy/postgres.env` 注入，不写入仓库
- 涉及大批量数据写入 / 替换前，先用 dry-run 看报告再 `--apply`
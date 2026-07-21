# 核心运行脚本

该目录只保留当前数据生产、OCR、语料修复和 PostgreSQL 部署入口。

| 文件 | 作用 |
| --- | --- |
| `rebuild_evidence_candidates.py` | 从缺失正文或乱码报告重建隔离候选库，可选 OCR |
| `run_mineru_inventory_batch.py` | 批量准备和运行 MinerU OCR |
| `run_mineru_local.ps1` | 在本地 GPU 环境运行单个或目录级 MinerU OCR |
| `audit_mineru_quality.py` | 审核 MinerU 批次输出质量 |
| `promote_ocr_candidates.py` | 筛选 OCR 结果，并通过 staging/backup 更新候选数据 |
| `build_evidence_last.py` | 根据最终审核结果构建最终 evidence 数据集 |
| `repair_evidence_corpus.py` | 审核并修复 evidence 语料，排除项进入 quarantine |
| `remote_pg_build.py` | 远程执行 PostgreSQL 入库、向量化、索引和验证 |


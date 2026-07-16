# quality 质量审计与修复

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `audit_pdf_markdown.py` | 对账 PDF 与 Markdown，检查缺失、完整度、页码和 OCR 状态 |
| `backfill_evidence_artifacts.py` | 从 clean Markdown 重建文档、卡片、视图并回填元数据 |
| `repair_quality.py` | 修复已有元数据和派生记录，重新汇总 chunk 与向量元数据 |

## 推荐流程

1. 运行审计并保留 JSONL 报告。
2. 查看低完整度、页码不匹配和 OCR 未解决样本。
3. 大范围回填前先使用 `dry_run`。
4. 确认后执行写入，并同步 section、chunk、card 和 view。
5. 内容或检索文本改变后，重新入库并向量化。

## 指标解释

- `markdown_to_pdf_text_ratio` 用于发现抽取严重缺失，不等同于质量评分。
- `page_mismatch` 允许封面和空白页造成的小幅差异。
- `ocr_unresolved` 表示 OCR 不可用、失败或仍需人工复核。
- `missing_required_views` 检查范围人群、推荐摘要和 PICO 等视图。

## 安全边界

回填以 `markdown_clean` 为权威集合，旧 `documents.jsonl` 仅补充缺失字段。实际执行会重写多个 JSONL 主产物，应先备份并检查 dry-run 报告。
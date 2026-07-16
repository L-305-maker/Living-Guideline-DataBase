# cleaning 清洗与派生产物

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `pdf_to_md.py` | 批量 PDF 转换、内容哈希去重、OCR 决策、并行执行和 manifest |
| `pdf_to_markdown.py` | 单文档转换的兼容入口 |
| `cleaner.py`、`markdown_cleaner.py` | 清理页眉页脚、异常字符和版式噪声，并评估质量 |
| `chunker.py`、`semantic_chunker.py` | 按标题和语义边界生成分块 |
| `block_chunker.py`、`block_encoder.py` | 块级切分与编码辅助逻辑 |
| `encoder.py` | 文本向量编码的通用封装 |
| `evidence_pipeline.py` | 串联转换、清洗、派生产物和索引前准备 |
| `add_clinical_departments.py` | 为文档和派生记录补充分科标签 |

## 输入与输出

典型输入为 `data/raw_pdf/**/*.pdf`。主要输出：

- `data/markdown_raw/*.md`：原始 Markdown。
- `data/markdown_clean/*.md`：带 front matter 的权威清洗文本。
- `data/documents.jsonl`：文档级元数据和正文路径。
- `data/sections/*.jsonl`、`data/chunks/*.jsonl`：章节与分块数据。
- `data/document_cards.jsonl`、`data/document_views.jsonl`：文档召回文本。

## 关键逻辑

- PDF 去重基于内容哈希；同内容不同名称只保留一份。
- 多进程转换只传可序列化参数，结果使用 `as_completed` 收集。
- 相同标题可能产生相同 `doc_id`，写 manifest 前会追加稳定后缀。
- 切分优先保留标题、列表、表格和推荐语句边界；超长单元才拆分。
- `cleaning_quality` 与 `cleaning_flags` 会影响重排分数。

## 变更与验证

修改清洗规则后重建 clean Markdown 和派生产物；修改切分规则后重建词法及向量索引。

```powershell
python -B -m pytest tests/test_markdown_cleaner.py tests/test_pdf_ocr_ingestion.py -q -p no:cacheprovider
python -B -m compileall -q src/pipeline/cleaning
```
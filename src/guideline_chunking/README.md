# guideline_chunking 结构化指南切分

## 处理链路

`Markdown 解析 → 块分类 → 父章节 → 原子块 → 表格块 → 关系链接 → BM25/向量索引 → 检索与评估`

| 文件 | 作用 |
| --- | --- |
| `markdown_parser.py` | 解析 front matter、标题栈、页码标记、段落、列表和表格 |
| `block_classifier.py` | 判断推荐、证据、背景、表格等块类型 |
| `section_chunker.py` | 按 `heading_path` 聚合父章节并限制章节长度 |
| `atomic_chunker.py` | 生成可独立检索的原子块，保留推荐与证据结构 |
| `table_chunker.py` | 生成表格父块、行块和备注块 |
| `chunk_linker.py` | 建立父子、表格和相邻块双向关系 |
| `bm25_index.py`、`vector_index.py` | 本地词法及向量索引 |
| `chunk_retrieve_service.py` | 查询、过滤和结果组装 |
| `structural_rag.py` | 运行完整结构化处理流程 |
| `evaluation.py` | 检查召回和结构保持效果 |
| `models.py` | 块、章节和链接数据结构 |
| `text_templates.py` | 不同块类型的检索文本模板 |

## 关键约束

- 块顺序以 `order_index` 为准，标题变化通过 `heading_path` 表达。
- 父章节只在块边界拆分，避免截断表格、列表或推荐语句。
- 表格父块用于整体语义，行块用于精确命中；缺失单元格必须保持列对齐。
- 所有块保留 `doc_id`、标题路径、页码范围和来源信息。
- 链接应在全部块生成后统一建立，避免引用尚不存在的 ID。

## 修改影响

修改解析、分类、切分模板或 ID 规则后，应重新执行整个链路并重建索引。只更新某一种 chunk 会导致 `all_chunks.jsonl`、链接和索引之间不一致。

## 验证

```powershell
python -B -m pytest tests/test_guideline_markdown_parser.py tests/test_guideline_section_chunker.py tests/test_guideline_atomic_chunker.py tests/test_guideline_table_chunker.py tests/test_guideline_chunk_linker.py tests/test_guideline_structural_rag.py -q -p no:cacheprovider
```
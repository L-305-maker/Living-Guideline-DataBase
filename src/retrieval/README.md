# retrieval 检索、融合与重排

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `sqlite_store.py` | 本地 SQLite 主表、中文/通用 FTS5 索引和 BM25 检索 |
| `vector_store.py` | FAISS 文档卡片、文档视图、分块索引的构建与搜索 |
| `document_multiview_search.py` | 多视图文档召回、按文档聚合和科室路由 |
| `hybrid.py` | 文档召回、分块召回、RRF 融合、过滤和上下文组装 |
| `reranker.py` | 规则重排与 BGE-M3 CrossEncoder 重排 |
| `rrf.py` | 用名次而非原始分数融合不同召回通道 |
| `chunk_normalizer.py` | 统一不同切分产物的字段形状 |
| `chunk_doc_aggregator.py` | 把 chunk 命中聚合为文档级信号 |
| `bm25_store.py` | 轻量 BM25 数据持久化辅助逻辑 |
| `common.py` | 统一时间范围解析、旧 chunk 字段补齐、文本截断和来源上下文拼接 |
| `document_repr/` | 构建文档卡片和多视图表示 |

## 混合检索流程

1. 文档卡片和文档视图分别进行词法及向量召回。
2. 使用 RRF 融合通道名次，避免直接比较 BM25、余弦相似度和规则分。
3. 科室路由和元数据过滤缩小文档候选。
4. 在候选文档内执行分块词法与向量召回。
5. 规则或 CrossEncoder 重排后，补充相邻块形成可追溯上下文。

## 向量检索约束

- FAISS 索引和 mapping JSONL 必须逐行对应，数量或顺序不一致时应重建。
- 建库与查询必须使用相同模型和 L2 归一化设置。
- `VECTOR_RETRIEVAL_REQUIRED=1` 时缺少索引、映射或模型依赖会直接报错；关闭时允许返回空向量通道。
- 分片索引由 manifest 记录顺序和路径；存在分片清单时优先使用分片布局。

## 重排配置

可通过 `DOCUMENT_RERANKER`、`CHUNK_RERANKER` 选择实现；BGE 参数包括 `BGE_RERANKER_MODEL`、`BGE_RERANKER_DEVICE`、`BGE_RERANKER_BATCH_SIZE`、`BGE_RERANKER_MAX_LENGTH` 和 `BGE_RERANKER_LOCAL_ONLY`。

## 维护与验证

调整召回池大小或融合权重后，检查 `retrieval_scores`、通道名次和 `match_reason`。不要删除回退路径，它用于模型未缓存或推理失败时维持可用性。

```powershell
python -B -m pytest tests/test_document_multiview_search.py tests/test_new_retrieval_scheme_guardrails.py -q -p no:cacheprovider
python -B -m compileall -q src/retrieval
```
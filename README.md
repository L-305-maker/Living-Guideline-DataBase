# Data Splitting Ingestion

这是指南 PDF 清洗、结构化切分、向量化和混合检索项目。

## 流程

PDF 文本质量评估 -> Markdown 转换 -> 噪声清洗 -> 章节和 chunk 切分 -> 数据库入库 -> BGE 向量化 -> 混合检索。

## 目录

- src/pipeline：PDF、OCR、清洗、质量审计和流水线编排。
- src/guideline_chunking：结构化 Markdown 解析和指南切分。
- src/retrieval：SQLite、BM25、向量、RRF 和重排。
- src/storage：PostgreSQL、pgvector 和向量化。
- src/mcp：MCP 服务和检索 API。
- deploy：远程服务器部署模板。

## 运行

~~~powershell
python -m pytest -q
python -m src.mcp
~~~

默认数据目录是 data/evidence。禁止提交密码、Token、远程 DSN 和模型缓存。

## PostgreSQL

应用调用 BGE 生成 embedding，PostgreSQL 加 pgvector 负责存储和近邻查询。入库和查询必须使用同一个模型、维度和文本字段。

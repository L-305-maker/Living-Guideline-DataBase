# models 共享数据模型

## 文件

`schemas.py` 定义数据处理、存储和检索阶段共享的结构化模型，`__init__.py` 提供包入口。

## 维护要求

- 字段新增应尽量提供合理默认值，以兼容已有 JSONL 数据。
- 字段重命名属于跨模块变更，需要同步清洗生产者、结构化切分、数据库入库、检索输出和测试。
- `doc_id`、`section_id`、`chunk_id` 等标识字段不能随展示文本变化而随意改变。
- 页码、字符范围、标题路径、来源文件和 Markdown 路径属于证据追踪字段，不应为了简化模型而删除。
- 布尔字段写入 JSONL、front matter 和数据库时要保持统一语义，避免字符串与布尔值混用。

## 兼容性检查

修改模型后，至少检查以下路径：

1. 旧 JSONL 能否被读取。
2. 新产物能否写入 SQLite 和 PostgreSQL。
3. MCP 输出能否序列化为 JSON。
4. 分块与文档关联是否仍能通过稳定 ID 恢复。

```powershell
python -B -m py_compile src/models/schemas.py
python -B -m pytest tests -q -p no:cacheprovider
```
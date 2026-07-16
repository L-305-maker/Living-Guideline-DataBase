# utils 公共工具

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `io.py` | 数据目录、父目录创建、Markdown/JSONL 迭代与写入 |
| `ids.py` | 文本和文件哈希、稳定 ID 辅助逻辑 |
| `front_matter.py` | 解析和输出 Markdown front matter |
| `metadata.py` | 标题、摘要、日期和机构等元数据提取与清洗 |
| `clinical_department.py` | 基于标题、摘要和正文信号进行临床科室分类 |

## 使用约束

- JSONL 写入统一使用 UTF-8，并保证一行一个完整 JSON 对象。
- 稳定 ID 的输入字段和拼接顺序属于数据协议；修改后必须重建所有派生产物和索引。
- front matter 输出应保持确定性，减少无意义的全文件差异。
- 元数据提取优先使用可靠的路径或 front matter 提示，再扫描正文前部，避免参考文献中的日期和机构污染结果。
- 科室分类同时使用绝对阈值和相对优势，低置信度结果应保留为未分类而不是强行猜测。

## 修改边界

公共工具被多个阶段共享。修改前先搜索所有调用点；不要在工具函数内加入隐式网络请求、遥测或环境状态写入。错误处理应保留源路径或记录 ID，便于批量任务定位单篇问题。

## 验证

```powershell
python -B -m compileall -q src/utils
python -B -m pytest tests/test_chunk_normalizer.py tests/test_markdown_cleaner.py -q -p no:cacheprovider
```
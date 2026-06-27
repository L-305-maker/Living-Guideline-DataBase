# common 模块

`common/` 放跨阶段复用的小工具。它不应该包含具体业务决策，比如“某条推荐是否可发布”，而是提供读写、格式化、质量上下文等基础能力。

## 数据如何流动

```text
pipeline / storage / scripts
  -> 调用 common 工具
  -> 读写 JSONL、整理质量字段、处理通用结构
  -> 返回给调用方继续业务处理
```

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `process_jsonl.py` | JSONL 读写基础函数，是流水线文件传递的底座 |
| `extraction_common.py` | 抽取相关共享类型 |
| `record_quality.py` | 把 cleaned record 的质量信息传递到 block 和候选中 |
| `data_artifacts.py` | 数据产物清单、分类和治理辅助 |
| `mcp_server.py` | 面向外部工具/服务的辅助入口 |

## 维护原则

- 这里的函数应该小而稳定。
- 不要让 `common/` 反向依赖 `pipeline/` 或 `storage/` 的业务规则。
- 如果一个工具只被某个阶段使用，优先放在那个阶段目录里。

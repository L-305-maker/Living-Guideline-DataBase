# domain/common

这里放领域模型共享的基础能力：枚举、稳定 ID、序列化基类和通用类型。

## 数据流动

```text
domain 实体
  -> 使用 common 枚举和 stable_id
  -> to_dict()
  -> JSONL / storage
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `base.py` | 序列化基础 mixin |
| `ids.py` | 稳定 ID 生成 |
| `types.py` | 枚举和通用类型 |

这里的改动会影响很多实体，修改前应跑全量测试。

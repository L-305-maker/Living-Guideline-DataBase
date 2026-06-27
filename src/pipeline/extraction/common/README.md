# extraction/common

这里存放抽取阶段共享的枚举、映射、校验和通用构造函数。

## 数据流动

```text
recommendation / grade / pico / evidence extractors
  -> common validators / mappers / enums
  -> normalized candidate rows
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `enums.py` | 抽取相关枚举 |
| `extractor_common.py` | trace、source metadata 等共享构造 |
| `mappers.py` | 字段映射 |
| `validators.py` | 候选输出校验 |

这里适合放“多个抽取器都会用”的逻辑；单一抽取器专属规则应留在对应目录。

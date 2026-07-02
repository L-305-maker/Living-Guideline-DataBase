# extraction/enhancement

这里负责对初步候选进行增强、归一化或补充字段。

## 数据流动

```text
raw candidates
  -> enhancement
  -> enriched candidates
  -> review / versioning
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `candidate_enhancer.py` | 推荐候选增强 |
| `grade_candidate_enhancer.py` | GRADE 候选增强 |
| `common.py` | 增强阶段共享逻辑 |

增强阶段不应绕过 review 和 publish gate。

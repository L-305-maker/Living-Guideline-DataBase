# quality 阶段

`quality/` 负责生成质量报告、评估 goldset、抽样审计候选结果。

## 数据如何流动

```text
blocks / candidates / picos / evidence / traces
  -> quality reports
  -> summary jsonl
  -> sample jsonl
  -> 人工排查或回归评估
```

## 子目录

| 子目录 | 作用 |
| --- | --- |
| `reports/` | 各类质量报告生成器 |
| `common/` | 报告共享工具 |

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `goldset.py` | 人工 goldset 对比评估 |
| `recommendation_audit.py` | 推荐候选审计指标 |

质量报告不改变业务数据，只帮助判断哪一阶段需要修规则或补复核。

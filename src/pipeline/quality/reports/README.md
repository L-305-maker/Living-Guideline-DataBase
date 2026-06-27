# quality/reports

这里包含面向不同产物的质量报告生成器。

## 数据流动

```text
blocks / candidates / picos / evidence / traces
  -> report builder
  -> summary JSONL
  -> sample JSONL
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `block_quality_report.py` | SourceBlock 质量 |
| `candidate_quality_report.py` | 推荐候选质量 |
| `candidate_overall_quality_report.py` | 推荐和 GRADE 综合质量 |
| `pico_quality_report.py` | PICO 质量 |
| `evidence_quality_report.py` | EvidenceItem 质量 |
| `llm_output_quality_report.py` | LLM 输出质量 |

报告文件只观察和汇总，不直接修改流水线产物。

# extraction 测试组

## 覆盖范围

本组测试候选抽取阶段：

- 推荐语句抽取、噪声过滤和语句质量门。
- GRADE 与 recommendation 的关联。
- PICO 和 evidence 的基础质量门。
- source span first 抽取路径。
- 推荐候选审计指标和 layout repair 队列。

## 什么时候必须运行

修改以下模块时必须运行：

```powershell
python -m tests.test_main --group extraction
```

- `src/pipeline/extraction/recommendation/`
- `src/pipeline/extraction/grade/`
- `src/pipeline/extraction/pico/`
- `src/pipeline/extraction/evidence/`
- `src/pipeline/extraction/source_span/`

## 常见失败含义

- 推荐质量测试失败：检查推荐动词、否定句、表格噪声和句子切分。
- GRADE 关联失败：检查 block id、source order、section overlap。
- source span 失败：检查 raw/clean offset 是否被破坏。

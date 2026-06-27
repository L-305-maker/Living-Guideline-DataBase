# quality/common

这里放质量报告生成器共享的小工具。

## 数据流动

```text
quality reports/*
  -> common/report_common.py
  -> summary rows / sample rows
```

报告公共逻辑应保持轻量，避免引入具体抽取器的业务判断。

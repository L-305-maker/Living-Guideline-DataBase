# ocr OCR 与 PDF 质量处理

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `pdf_quality.py` | 根据页数、可抽取字符和页面密度判断是否需要 OCR |
| `ocrmypdf_runner.py` | 调用本地 OCR 工具并管理输出 |
| `baidu_ocr.py` | 页面渲染、百度 OCR、版面重建、缓存、配额和批处理 |
| `deepseek_ocr.py` | 调用 DeepSeek OCR 服务识别页面并按页缓存 Markdown；仅用于显式选择该后端的任务 |

## 百度 OCR 配置

凭据只能通过环境变量提供：

- `BAIDU_OCR_ACCESS_TOKEN`：已有访问令牌时直接使用。
- `BAIDU_OCR_API_KEY` 与 `BAIDU_OCR_SECRET_KEY`：用于换取访问令牌。

默认页缓存位于 `data/evidence/baidu_ocr_pages`。更换 DPI 或识别策略后，应确认旧缓存是否仍适用。

DeepSeek OCR 的地址、模型与凭据必须通过运行环境提供，禁止写入源码或日志。切换 OCR 后端后需要使用独立缓存目录，避免不同模型的页面结果互相覆盖。

## 执行语义

- `dry_run` 只规划候选、页数和输出路径，不请求 API。
- `max_pages` 限制单篇页数，`max_total_pages` 控制整批预算。
- 配额耗尽后，后续文档标记为 `deferred_quota`。
- `replace_markdown` 只替换 clean Markdown 正文，front matter 仍需保留。
- 版面重建识别左右栏和跨栏块；坐标不足时退化为单栏顺序。

## 产物与排错

每次批处理生成独立 JSONL manifest。优先检查 `missing_source_pdf`、`skipped_page_budget`、`deferred_quota` 与 `failed`。

```powershell
python -B -m pytest tests/test_baidu_ocr.py tests/test_pdf_ocr_ingestion.py -q -p no:cacheprovider
```
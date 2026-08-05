# OCR 辅助模块：扫描版 PDF 的多引擎调度入口。
#
# 主要子模块：
# - mineru_runner：本地 GPU MinerU 调用，含跨进程文件锁
# - baidu_ocr / deepseek_ocr：云端 OCR 备选（配额敏感，需节流）
# - ocrmypdf_runner：本地无 GPU 时的兜底
# - pdf_quality：PDF 文本层评估，决定是否需要 OCR
"""OCR helpers for PDF-to-Markdown evidence ingestion."""
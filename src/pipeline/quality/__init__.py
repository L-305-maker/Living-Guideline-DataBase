# 证据语料的质检与修复入口。
#
# 主要子模块：
# - audit_pdf_markdown：PDF 与转换后 Markdown 的保真度审计
# - repair_quality：常见质量问题的自动修复（标题、日期、mojibake 等）
# - backfill_evidence_artifacts：补齐缺失的 JSONL 字段，保持下游契约一致
"""Quality audit and repair helpers for evidence artifacts."""
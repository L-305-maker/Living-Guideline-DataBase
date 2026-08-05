# 公共工具模块：文件系统、JSONL I/O、ID 生成、文本规范化等。
#
# 模块职责：作为跨 pipeline / storage / retrieval / mcp 各层共享的辅助函数集合；
# 设计原则：保持无状态（除模块常量）、依赖轻量、可被任意层 import。
"""Utility helpers."""
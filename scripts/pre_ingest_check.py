"""本地脚本入口：把 src 中的项目能力包装成命令行工具，方便运行、审计或质量检查。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.pre_ingest_check import main


if __name__ == "__main__":
    main()


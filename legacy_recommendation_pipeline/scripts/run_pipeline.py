"""本地脚本入口：把 src 中的项目能力包装成命令行工具，方便运行、审计或质量检查。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

import sys
from pathlib import Path


# 这个脚本是给“人在命令行直接运行”的薄入口。
# 它不承载业务逻辑，只负责把项目根目录加入 sys.path，然后转交给真正的流水线 main()。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestration.run_pipeline import main


if __name__ == "__main__":
    # 真正的参数解析和流水线执行都在 src.pipeline.orchestration.run_pipeline.main 中。
    main()


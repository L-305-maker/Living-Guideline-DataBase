"""原始数据接入文件：负责从外部来源采集、整理或转换医学指南和文献原始输入。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from __future__ import annotations

from src.ingestion.crawler.processors.origin import main


if __name__ == "__main__":
    main()



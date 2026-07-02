"""包初始化文件：声明当前目录是 Python 包，并集中暴露本包对外可用的入口。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from src.domain.origin.guideline import Guideline
from src.domain.origin.paper import Paper
from src.domain.origin.source_block import SourceBlock, Sourceblock
from src.domain.origin.source_record import SourceRecord

__all__ = ["Guideline", "Paper", "SourceBlock", "Sourceblock", "SourceRecord"]


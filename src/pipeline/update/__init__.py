"""版本更新检测工具：比较 RecommendationVersion 并生成更新事件候选。"""

from src.pipeline.update.version_diff import build_update_logs, build_update_logs_file

__all__ = ["build_update_logs", "build_update_logs_file"]

"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional

from src.domain.common import SerializableMixin, UpdateType


@dataclass
class UpdateLog(SerializableMixin):
    update_log_id: str  # 更新日志唯一ID，用于记录一次知识更新事件
    guideline_id: str  # 所属指南ID，关联 origin.guideline.guideline_id
    new_recommendation_version_id: str  # 更新后推荐意见版本ID，关联 knowledge.recommendation_version.recommendation_version_id
    recommendation_version_id: Optional[str] = None  # 本次更新直接作用的推荐意见版本ID，通常等于 new_recommendation_version_id
    old_recommendation_version_id: Optional[str] = None  # 更新前推荐意见版本ID，用于追踪版本演变
    update_type: UpdateType = "evidence_updated"  # 更新类型，例如新增证据、推荐意见变化、GRADE变化
    change_summary: str = ""  # 更新内容摘要，说明本次变化的核心结论
    change_reason: Optional[str] = None  # 更新原因，例如新证据出现、指南修订、人工审核修正
    triggering_evidence_ids: list[str] = field(default_factory=list)  # 触发本次更新的证据ID列表
    updated_by: Optional[str] = None  # 更新执行者，可以是模型名、流程名或人工审核者
    updated_at: str = ""  # 系统内记录更新时间，建议使用 ISO 8601 字符串
    published_at: Optional[str] = None  # 若更新已对外发布，记录发布时间


"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from src.domain.candidate import GradeCandidate, ModelTrace, RecommendationCandidate
from src.domain.common import (
    ArticleType,
    CandidateStatus,
    Certainty,
    Direction,
    EffectDirection,
    ExtractionMethod,
    GradeDomainJudgement,
    GradeSystem,
    GuidelineStatus,
    GuidelineType,
    InputEntityType,
    JsonDict,
    ModelMethod,
    ModelTaskType,
    PicoStatus,
    Priority,
    PublicationBiasJudgement,
    RecommendationChangeType,
    RecommendationStrength,
    ScreeningStatus,
    SerializableMixin,
    StudyDesign,
    UpdateType,
    stable_id,
)
from src.domain.knowledge import EvidenceItem, PicoQuestion, RecommendationVersion
from src.domain.origin import Guideline, Paper, SourceBlock, SourceRecord, Sourceblock
from src.domain.profile import GuidelineProfile
from src.domain.update import UpdateLog

__all__ = [
    "SerializableMixin",
    "stable_id",
    "JsonDict",
    "Direction",
    "RecommendationStrength",
    "Certainty",
    "ExtractionMethod",
    "CandidateStatus",
    "GuidelineType",
    "GuidelineStatus",
    "PicoStatus",
    "Priority",
    "GradeSystem",
    "GradeDomainJudgement",
    "PublicationBiasJudgement",
    "ModelTaskType",
    "ModelMethod",
    "InputEntityType",
    "RecommendationChangeType",
    "ArticleType",
    "ScreeningStatus",
    "StudyDesign",
    "EffectDirection",
    "UpdateType",
    "SourceRecord",
    "SourceBlock",
    "Guideline",
    "Paper",
    "ModelTrace",
    "RecommendationCandidate",
    "GradeCandidate",
    "PicoQuestion",
    "RecommendationVersion",
    "EvidenceItem",
    "GuidelineProfile",
    "UpdateLog",
    "Sourceblock",
]


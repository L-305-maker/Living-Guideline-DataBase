"""领域模型文件：定义 Living-Guideline 项目中的核心数据结构，只描述数据形状，不负责文件读写或数据库操作。

阅读本文件时，先看模块入口函数和被谁调用，再看具体规则或数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.domain.common import InputEntityType, JsonDict, ModelMethod, ModelTaskType, SerializableMixin


@dataclass
class ModelTrace(SerializableMixin):
    model_trace_id: str
    task_type: ModelTaskType
    method: ModelMethod
    model_name: str
    input_entity_type: InputEntityType
    input_entity_id: str
    input_text: str
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    raw_output: Any = None
    parsed_output: Any = None
    confidence: Optional[float] = None
    parameters: JsonDict = field(default_factory=dict)
    token_usage: JsonDict = field(default_factory=dict)
    runtime_ms: Optional[int] = None
    code_version: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    human_verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    target_table: Optional[str] = None
    target_entity_id: Optional[str] = None
    created_at: str = ""


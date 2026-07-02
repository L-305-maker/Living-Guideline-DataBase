"""在清洗、结构解析和抽取之间传递质量上下文的辅助函数。"""

from __future__ import annotations

from src.common.extraction_common import JsonDict


def _flag_codes(flags: object) -> list[str]:
    """把清洗阶段产生的 flag 统一整理成字符串列表，方便下游报告和路由使用。"""

    if not isinstance(flags, list):
        return []
    codes: list[str] = []
    for flag in flags:
        if isinstance(flag, dict) and flag.get("code"):
            codes.append(str(flag.get("code")))
        elif isinstance(flag, str) and flag:
            codes.append(flag)
    return codes


def record_quality_context(record: JsonDict) -> JsonDict:
    """从清洗记录中提取需要随 SourceBlock 一起传递的质量字段。"""

    return {
        "record_cleaning_status": record.get("cleaning_status", ""),
        "record_cleaning_quality_score": record.get("cleaning_quality_score", None),
        "record_quality_flags": _flag_codes(record.get("cleaning_quality_flags")),
        "record_type": record.get("record_type", ""),
        "record_type_confidence": record.get("record_type_confidence", None),
    }


def block_record_quality_context(block: JsonDict) -> JsonDict:
    """从 SourceBlock 的 quality 或 metadata 中读取记录级质量上下文。"""

    quality = block.get("quality") if isinstance(block.get("quality"), dict) else {}
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    flags = quality.get("record_quality_flags") or metadata.get("record_quality_flags") or []
    return {
        "record_cleaning_status": quality.get("record_cleaning_status") or metadata.get("record_cleaning_status") or "",
        "record_cleaning_quality_score": quality.get("record_cleaning_quality_score", metadata.get("record_cleaning_quality_score")),
        "record_quality_flags": _flag_codes(flags),
        "record_type": quality.get("record_type") or metadata.get("record_type") or "",
        "record_type_confidence": quality.get("record_type_confidence", metadata.get("record_type_confidence")),
    }

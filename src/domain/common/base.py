from __future__ import annotations

from dataclasses import asdict, dataclass

from src.domain.common.types import JsonDict


@dataclass
class SerializableMixin:
    def to_dict(self) -> JsonDict:
        return asdict(self)

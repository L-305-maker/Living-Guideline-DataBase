"""Build local source context around Section records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateContext:
    before: str = ""
    after: str = ""


class ContextBuilder:
    def __init__(self, window: int = 1) -> None:
        self.window = max(0, window)

    def build(self, sections: list[dict[str, Any]], index: int) -> CandidateContext:
        before = sections[max(0, index - self.window) : index]
        after = sections[index + 1 : index + 1 + self.window]
        return CandidateContext(
            before="\n\n".join(str(item.get("content") or "") for item in before),
            after="\n\n".join(str(item.get("content") or "") for item in after),
        )

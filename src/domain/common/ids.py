from __future__ import annotations

import hashlib
from typing import Any


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"

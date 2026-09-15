"""Per-request retrieval stage timing for MCP call logs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from threading import Lock
from typing import Any, Iterator


class RetrievalTiming:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stages: dict[str, dict[str, float | int]] = {}

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            item = self._stages.setdefault(stage, {"duration_ms": 0.0, "calls": 0})
            item["duration_ms"] = float(item["duration_ms"]) + seconds * 1000
            item["calls"] = int(item["calls"]) + 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                stage: {"duration_ms": round(float(item["duration_ms"]), 3), "calls": int(item["calls"])}
                for stage, item in self._stages.items()
            }


_CURRENT: ContextVar[RetrievalTiming | None] = ContextVar("retrieval_timing", default=None)


def start_timing() -> tuple[RetrievalTiming, Token[RetrievalTiming | None]]:
    timing = RetrievalTiming()
    return timing, _CURRENT.set(timing)


def stop_timing(token: Token[RetrievalTiming | None]) -> None:
    _CURRENT.reset(token)


def current_timing() -> RetrievalTiming | None:
    return _CURRENT.get()


@contextmanager
def timed(stage: str, timing: RetrievalTiming | None = None) -> Iterator[None]:
    timing = timing or current_timing()
    if timing is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        timing.record(stage, time.perf_counter() - started)


def timed_call(stage: str, call: Any, *args: Any, timing: RetrievalTiming | None = None, **kwargs: Any) -> Any:
    with timed(stage, timing):
        return call(*args, **kwargs)

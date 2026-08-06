# src/mcp/concurrency.py
import os, threading, contextvars
from contextlib import contextmanager

ACTIVE = threading.BoundedSemaphore(
    int(os.getenv("MCP_MAX_CONCURRENT", "16"))
)
total = 0
lock = threading.Lock()
total_lock_seen = threading.Lock()

@contextmanager
def acquire_slot(tool: str, call_id: str):
    global total
    with total_lock_seen:
        total += 1
    acquired = ACTIVE.acquire(timeout=float(os.getenv("MCP_ACQUIRE_TIMEOUT", "30")))
    if not acquired:
        # 让上游 FastMCP 把这个当成 ResourceExhausted
        raise RuntimeError(f"MCP busy, exceeded MAX_CONCURRENT for tool={tool}")
    try:
        yield
    finally:
        ACTIVE.release()
        # 用 logging 或 atomic counters, 别在热路径上做 I/O
        
# MinerU OCR 集成（用于扫描版 PDF 处理）。
#
# MinerU 是独立的 GPU OCR 流水线，部署在 .venv-mineru-gpu 虚拟环境。
# 本模块把 mineru CLI 包装成单 PDF 调用，返回 Markdown，并通过跨进程文件锁
# 串行化并发调用，避免多 worker 抢占同一块 GPU。
#
# 配置（环境变量）：
#   MINERU_OCR_MODE        never / 其它（含未设）：是否启用 MinerU 路由
#   MINERU_EXE              mineru 可执行路径（默认 .venv-mineru-gpu/Scripts/mineru.exe，否则 PATH）
#   MINERU_CONFIG           MinerU tools config JSON（默认 data/mineru/mineru.json）
#   MINERU_MODELSCOPE_CACHE 模型缓存（默认 data/mineru/modelscope；回退 MODELSCOPE_CACHE）
#   MINERU_LANGUAGE         OCR 语言（默认 ch）
#   MINERU_TIMEOUT_SECONDS  单 PDF 超时（默认 3600）
#   CUDA_VISIBLE_DEVICES    透传给 MinerU 子进程（默认 0）
#
# 注意：MinerU 输出不含 <!-- page: N --> 标记（与 PyMuPDF 路径不同），
# 下游按页处理的逻辑（页眉/页脚去重等）会自动退化为整篇处理。
"""MinerU OCR integration for scanned PDF ingestion.

MinerU is an external GPU-backed OCR pipeline installed in its own virtual
environment (``.venv-mineru-gpu``). This module wraps the ``mineru`` CLI for a
single PDF, returns the produced Markdown, and serializes concurrent
invocations with a cross-process file lock so multiple pipeline workers do not
contend for GPU memory.

Configuration is via environment variables:

- ``MINERU_OCR_MODE``: set to ``never`` to disable MinerU routing entirely;
  any other value (including unset) enables it.
- ``MINERU_EXE``: path to the ``mineru`` executable. Defaults to
  ``<workspace>/.venv-mineru-gpu/Scripts/mineru.exe``, then ``mineru`` on PATH.
- ``MINERU_CONFIG``: MinerU tools config JSON. Defaults to
  ``<workspace>/data/mineru/mineru.json``.
- ``MINERU_MODELSCOPE_CACHE``: model cache directory. Defaults to
  ``<workspace>/data/mineru/modelscope`` (falls back to ``MODELSCOPE_CACHE``).
- ``MINERU_LANGUAGE``: OCR language passed via ``-l``. Defaults to ``ch``.
- ``MINERU_TIMEOUT_SECONDS``: per-PDF timeout. Defaults to 3600.
- ``CUDA_VISIBLE_DEVICES``: passed through to the MinerU subprocess (default
  ``0``).

Note: MinerU output has no ``<!-- page: N -->`` markers (unlike the PyMuPDF
path), so downstream page-scoped logic (per-page header/footer dedup, page
counts) degrades gracefully to whole-document processing for MinerU-routed
documents.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import WORKSPACE_ROOT

# 平台相关锁：msvcrt 仅 Windows，fcntl 仅 POSIX；缺包时锁退化为单进程内协作。
try:  # pragma: no cover - Windows only
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None

try:  # pragma: no cover - POSIX only
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_LANGUAGE = "ch"
# 视为"已禁用 MinerU"的环境变量值集合（大小写不敏感）。
DISABLED_MODES = {"never", "0", "false", "off", "no"}


@dataclass(frozen=True)
class MineruRun:
    """单次 MinerU 调用的结构化结果。

    applied / error：是否成功（失败时 error 含原因摘要）
    markdown / output_path：成功时填入 Markdown 文本与产物路径
    output_dir：MinerU scratch 目录（通常由调用方用 TemporaryDirectory 管理）
    """
    applied: bool = False
    markdown: str = ""
    output_path: str = ""
    output_dir: str = ""
    error: str = ""


def mineru_enabled() -> bool:
    """判定 MinerU 路由是否启用（默认启用）。"""
    value = os.environ.get("MINERU_OCR_MODE", "").strip().lower()
    return value not in DISABLED_MODES


def resolve_mineru_exe() -> Path | None:
    """定位 mineru 可执行文件：MINERU_EXE → 默认 exe → PATH mineru。"""
    explicit = os.environ.get("MINERU_EXE")
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    default = WORKSPACE_ROOT / ".venv-mineru-gpu" / "Scripts" / "mineru.exe"
    if default.is_file():
        return default
    found = shutil.which("mineru")
    return Path(found) if found else None


def resolve_mineru_config() -> Path | None:
    """定位 MinerU tools config JSON。"""
    explicit = os.environ.get("MINERU_CONFIG")
    path = Path(explicit) if explicit else WORKSPACE_ROOT / "data" / "mineru" / "mineru.json"
    return path if path.is_file() else None


def resolve_model_cache() -> Path | None:
    """定位 MinerU 模型缓存目录（modelscope）。"""
    explicit = os.environ.get("MINERU_MODELSCOPE_CACHE") or os.environ.get("MODELSCOPE_CACHE")
    path = Path(explicit) if explicit else WORKSPACE_ROOT / "data" / "mineru" / "modelscope"
    return path if path.is_dir() else None


def _lock_path() -> Path:
    """返回跨进程锁文件路径，默认在系统临时目录。"""
    lock_dir = Path(os.environ.get("MINERU_LOCK_DIR") or _default_lock_dir())
    return lock_dir / "mineru_ocr.lock"


def _default_lock_dir() -> str:
    import tempfile

    return tempfile.gettempdir()


def _acquire_msvcrt(handle: Any) -> None:
    """Windows msvcrt 锁获取循环。

    注意：msvcrt.locking 仅内部重试 ~10s，但 MinerU 任务可能跑几分钟，
    因此外层用 while True + 1s 间隔持续尝试。
    同时 msvcrt.locking 失败时会移动文件 offset，每次失败必须 seek(0) 回退，
    否则两个 worker 可能锁定不同字节，都进入临界区。
    """
    assert msvcrt is not None
    while True:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        except OSError:
            time.sleep(1.0)


class _MineruProcessLock:
    """跨进程互斥锁，确保并行 worker 不同时占用 GPU。

    实现细节：
    - 锁文件固定为 1 字节（多次 open 也不会膨胀）；
    - Windows 用 msvcrt.locking 字节锁；POSIX 用 fcntl.flock 文件锁；
    - enter/exit 必须配对使用，建议通过 with 语法调用。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any = None

    def __enter__(self) -> "_MineruProcessLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+b")
        # seek(0) + truncate(1) + write b"\0"：确保锁文件恒为 2 字节，
        # 避免多次 open 时文件无界增长。
        handle.seek(0)
        handle.truncate(1)
        handle.write(b"\0")
        handle.flush()
        handle.seek(0)
        if msvcrt is not None:
            _acquire_msvcrt(handle)
        elif fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        self._handle = handle
        return self

    def __exit__(self, *exc_info: object) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if msvcrt is not None:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            elif fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _find_output_markdown(output_dir: Path, stem: str) -> Path | None:
    """在 MinerU scratch 目录中查找输出 markdown。

    优先匹配 stem 相等的文件；否则若有且仅有一个 .md，返回它（兜底）；
    都没有则返回 None（视为 OCR 失败）。
    """
    candidates = sorted(output_dir.rglob("*.md"))
    for candidate in candidates:
        if candidate.stem == stem:
            return candidate
    if len(candidates) == 1:
        return candidates[0]
    return None


def run_mineru(
    pdf_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    language: str | None = None,
    timeout_seconds: int | None = None,
) -> MineruRun:
    """对单 PDF 调用 MinerU 并返回 Markdown。

    output_dir 是 MinerU 的 scratch 目录，由调用方负责生命周期（建议用 TemporaryDirectory）；
    返回的 MineruRun.markdown 是 read 完的字符串，调用方用完 scratch 后可丢弃。
    """
    exe = resolve_mineru_exe()
    if exe is None:
        return MineruRun(applied=False, error="mineru command was not found")
    config = resolve_mineru_config()
    if config is None:
        return MineruRun(applied=False, error="mineru tools config not found; set MINERU_CONFIG")
    pdf = Path(pdf_path)
    if not pdf.is_file():
        return MineruRun(applied=False, error=f"source pdf not found: {pdf}")
    if output_dir is not None and not Path(output_dir).is_dir():
        return MineruRun(applied=False, error=f"output dir not found: {output_dir}")

    lang = language or os.environ.get("MINERU_LANGUAGE", DEFAULT_LANGUAGE)
    timeout = timeout_seconds if timeout_seconds is not None else int(
        os.environ.get("MINERU_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    )
    out = Path(output_dir) / "output" if output_dir else Path(pdf.parent) / "mineru_scratch" / "output"
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # MINERU_* 限制并发与算力，与 docs/mineru.json 中的工具设置保持一致。
    env.update(
        {
            "MINERU_TOOLS_CONFIG_JSON": str(config),
            "MINERU_MODEL_SOURCE": "local",
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "MINERU_PROCESSING_WINDOW_SIZE": "1",
            "MINERU_PDF_RENDER_THREADS": "1",
            "MINERU_API_MAX_CONCURRENT_REQUESTS": "1",
            "MINERU_INTRA_OP_NUM_THREADS": "8",
            "MINERU_INTER_OP_NUM_THREADS": "2",
        }
    )
    model_cache = resolve_model_cache()
    if model_cache is not None:
        # 仅当解析到真实 cache 目录才覆盖；不抹掉父环境的 MODELSCOPE_CACHE。
        env["MODELSCOPE_CACHE"] = str(model_cache)
    command = [str(exe), "-p", str(pdf), "-o", str(out), "-b", "pipeline", "-m", "ocr", "-l", lang]

    try:
        # with 上下文确保锁一定释放，避免 MinerU 崩溃导致锁泄漏。
        with _MineruProcessLock(_lock_path()):
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=str(WORKSPACE_ROOT),
                env=env,
            )
    except subprocess.TimeoutExpired:
        return MineruRun(applied=False, error=f"mineru timed out after {timeout}s", output_dir=str(out))
    except Exception as exc:  # noqa: BLE001
        return MineruRun(applied=False, error=f"mineru failed to start: {exc}", output_dir=str(out))

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        return MineruRun(
            applied=False,
            error=(message[:800] if message else "mineru exited with non-zero code"),
            output_dir=str(out),
        )

    markdown_path = _find_output_markdown(out, pdf.stem)
    if markdown_path is None:
        return MineruRun(applied=False, error="mineru produced no markdown output", output_dir=str(out))
    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return MineruRun(applied=False, error="mineru produced empty markdown", output_dir=str(out))
    return MineruRun(applied=True, markdown=text, output_path=str(markdown_path), output_dir=str(out))
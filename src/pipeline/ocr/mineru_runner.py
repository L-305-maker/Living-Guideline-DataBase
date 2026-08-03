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
DISABLED_MODES = {"never", "0", "false", "off", "no"}


@dataclass(frozen=True)
class MineruRun:
    """Structured result of a single MinerU invocation."""

    applied: bool = False
    markdown: str = ""
    output_path: str = ""
    output_dir: str = ""
    error: str = ""


def mineru_enabled() -> bool:
    """Whether MinerU routing is enabled (default: enabled)."""
    value = os.environ.get("MINERU_OCR_MODE", "").strip().lower()
    return value not in DISABLED_MODES


def resolve_mineru_exe() -> Path | None:
    """Locate the ``mineru`` executable, or None if unavailable."""
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
    """Locate the MinerU tools config JSON, or None if unavailable."""
    explicit = os.environ.get("MINERU_CONFIG")
    path = Path(explicit) if explicit else WORKSPACE_ROOT / "data" / "mineru" / "mineru.json"
    return path if path.is_file() else None


def resolve_model_cache() -> Path | None:
    """Locate the MinerU model cache directory, or None if unavailable."""
    explicit = os.environ.get("MINERU_MODELSCOPE_CACHE") or os.environ.get("MODELSCOPE_CACHE")
    path = Path(explicit) if explicit else WORKSPACE_ROOT / "data" / "mineru" / "modelscope"
    return path if path.is_dir() else None


def _lock_path() -> Path:
    lock_dir = Path(os.environ.get("MINERU_LOCK_DIR") or _default_lock_dir())
    return lock_dir / "mineru_ocr.lock"


def _default_lock_dir() -> str:
    import tempfile

    return tempfile.gettempdir()


def _acquire_msvcrt(handle: Any) -> None:
    assert msvcrt is not None
    # LK_LOCK only retries for ~10s; MinerU jobs can run for minutes, so loop.
    # msvcrt.locking moves the file offset past the locked byte on failure, so
    # seek back to 0 on every attempt or two workers could lock different bytes
    # and both enter the critical section.
    while True:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        except OSError:
            time.sleep(1.0)


class _MineruProcessLock:
    """Cross-process exclusive lock so parallel workers do not share the GPU."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any = None

    def __enter__(self) -> "_MineruProcessLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+b")
        # Keep the lock file at a fixed tiny size: seek(0) + truncate(1) + write
        # in append mode leaves the file at 2 bytes regardless of how many
        # processes open it, so it cannot grow unbounded.
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
    """Run MinerU on a single PDF and return the produced Markdown.

    ``output_dir`` is used as scratch space for MinerU's raw output; the caller
    owns its lifecycle (e.g. a ``tempfile.TemporaryDirectory``). The produced
    Markdown is returned inline so the scratch space can be discarded.
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
        # Only override when a real cache dir is resolved; never blank out an
        # existing MODELSCOPE_CACHE from the parent environment.
        env["MODELSCOPE_CACHE"] = str(model_cache)
    command = [str(exe), "-p", str(pdf), "-o", str(out), "-b", "pipeline", "-m", "ocr", "-l", lang]

    try:
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

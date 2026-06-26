from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit
from typing import Any

try:
    import psycopg
except ModuleNotFoundError:
    # psycopg 是数据库功能的可选运行时依赖；没有安装时允许非数据库测试继续运行。
    psycopg = None  # type: ignore[assignment]


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
connection = Any


def mask_database_url(url: str) -> str:
    """隐藏 DATABASE_URL 中的密码，避免连接失败信息泄漏凭据。"""

    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid DATABASE_URL>"
    if not parts.password:
        return url
    username = parts.username or ""
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    auth = f"{username}:***@" if username else ""
    netloc = f"{auth}{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def get_connection() -> connection:
    """创建 PostgreSQL 连接，并在失败时输出脱敏后的连接信息。"""

    if psycopg is None:
        raise RuntimeError("PostgreSQL storage 需要安装 psycopg。请运行: pip install -r requirements.txt")
    try:
        return psycopg.connect(DATABASE_URL)
    except Exception as exc:  # noqa: BLE001
        safe_url = mask_database_url(DATABASE_URL)
        raise RuntimeError(f"无法连接 PostgreSQL。请检查 DATABASE_URL={safe_url!r}。原始错误: {exc}") from exc

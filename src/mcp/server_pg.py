"""PostgreSQL-only compatibility entry point for the MCP server."""

from __future__ import annotations

import os

os.environ["RAG_BACKEND"] = "postgres"

from src.mcp.server import main, mcp


if __name__ == "__main__":
    main()

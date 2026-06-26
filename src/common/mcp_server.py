from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


WORKSPACE = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")

mcp = FastMCP("medical-guideline-rag", host=MCP_HOST, port=MCP_PORT, json_response=True)


def resolve_workspace_path(path: str) -> Path:
    """解析用户传入路径，并拒绝访问工作区之外的文件。

    MCP 调用可能来自外部客户端，因此这里必须先做路径归一化和工作区边界检查，
    避免工具被用来读取或写入项目目录之外的内容。
    """

    
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKSPACE / candidate
    resolved = candidate.resolve()

    workspace = str(WORKSPACE).lower()
    if not str(resolved).lower().startswith(workspace):
        raise ValueError(f"Path is outside workspace: {path}")
    return resolved


def run_cli(args: list[str], timeout_seconds: int) -> dict[str, Any]:
    """运行项目 CLI 命令，并返回截断后的 stdout/stderr。

    返回内容限制在末尾 8000 字符，避免长日志通过 MCP 响应一次性塞满上下文。
    """

    result = subprocess.run(
        args,
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


def add_optional_arg(args: list[str], flag: str, value: Any) -> None:
    """只有可选参数有实际值时才追加 CLI flag，避免传入空字符串破坏命令语义。"""

    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        return
    args.extend([flag, str(value)])


@mcp.tool()
def ingestion(
    input_path: str = "outputs/chunk.jsonl",
    batch_size: int = 64,
    recreate: bool = False,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """MCP 工具：对指定 JSONL 文件运行 storage 入库命令。"""

    

    input_file = resolve_workspace_path(input_path)
    args = [
        PYTHON,
        "-m",
        "src.storage",
        "ingest",
        "--input",
        str(input_file),
        "--batch-size",
        str(batch_size),
    ]
    if recreate:
        args.append("--recreate")

    result = run_cli(args, timeout_seconds=timeout_seconds)
    result["input_path"] = str(input_file)
    return result


@mcp.tool()
def search(
    query: str,
    top_k: int = 5,
    year: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    issuer: str | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """MCP 工具：调用项目搜索命令，并按年份、机构等元数据做可选过滤。"""

    
    args = [
        PYTHON,
        "-m",
        "src.storage",
        "search",
        query,
        "--top-k",
        str(top_k),
    ]
    add_optional_arg(args, "--year", year)
    add_optional_arg(args, "--year-from", year_from)
    add_optional_arg(args, "--year-to", year_to)
    add_optional_arg(args, "--issuer", issuer)

    return run_cli(args, timeout_seconds=timeout_seconds)


if __name__ == "__main__":
    if MCP_TRANSPORT not in {"stdio", "streamable-http"}:
        raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable-http'")
    mcp.run(transport=MCP_TRANSPORT)

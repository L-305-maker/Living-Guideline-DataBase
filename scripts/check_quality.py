from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK_PATHS = ["src", "scripts"]


def run_command(args: list[str], *, optional: bool = False) -> int:
    """执行一个质量检查命令；可选工具缺失时只提示跳过，不中断主质量门。"""

    try:
        completed = subprocess.run(args, cwd=ROOT, check=False)
    except FileNotFoundError:
        if optional:
            print(f"skip optional tool: {args[0]}")
            return 0
        raise
    return completed.returncode


def run_compileall() -> int:
    """编译核心源码目录，提前发现语法错误和导入期的基础问题。"""

    ok = True
    for path in CHECK_PATHS:
        ok = compileall.compile_dir(str(ROOT / path), quiet=1) and ok
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="运行本地工程质量门。")
    parser.add_argument("--with-ruff", action="store_true", help="安装 ruff 时额外运行静态检查。")
    parser.add_argument("--with-mypy", action="store_true", help="安装 mypy 时额外运行类型检查。")
    args = parser.parse_args()

    checks = [
        ("compileall", run_compileall()),
        ("unittest", run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"])),
    ]
    if args.with_ruff:
        checks.append(("ruff", run_command(["ruff", "check", "src", "tests", "scripts"], optional=True)))
    if args.with_mypy:
        checks.append(("mypy", run_command(["mypy", "src"], optional=True)))

    failed = [name for name, code in checks if code != 0]
    if failed:
        print("quality gate failed: " + ", ".join(failed))
        return 1
    print("quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

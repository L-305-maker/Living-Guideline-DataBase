# MCP server 与 JSON API 入口。
#
# 子模块：
# - server.py：MCP 工具（search / read / retrieve）的 FastMCP 入口；
# - server_common.py：MCP 共享配置（env_bool / env_int）、请求载荷构造器、run_server；
# - api_pg.py：PostgreSQL 后端的 search / read / retrieve JSON API；
# - call_logger.py：JSONL 调用审计日志（含敏感字段脱敏与截断）；
# - smoke_test.py：冒烟测试脚本，验证 search / read / retrieve 链路通畅。
"""MCP server and JSON API entry points."""
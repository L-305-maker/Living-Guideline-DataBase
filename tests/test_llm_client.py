"""项目测试文件：通过小样本验证 Living-Guideline 数据处理链路的关键行为。

阅读测试时，优先看测试名称、输入样例和断言，它们通常说明对应模块的业务边界。
"""

import os
import unittest

from src.pipeline.llm_review.clients.llm_client import LLMRequest, post_llm_request


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "", headers: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}
        self.content = text.encode("utf-8") if text else b"{}"
        self.ok = 200 <= status_code < 300

    def json(self) -> dict:
        return self._payload


class LLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["OPENAI_API_KEY"] = "test-key"

    def test_post_llm_request_retries_retryable_status(self) -> None:
        calls = []
        responses = [
            FakeResponse(429, text="rate limited"),
            FakeResponse(200, payload={"output_text": "{}"}, headers={"x-request-id": "req_123"}),
        ]

        def fake_post(*args, **kwargs):
            calls.append((args, kwargs))
            return responses.pop(0)

        result = post_llm_request(
            LLMRequest(
                prompt="Review this.",
                model="test-model",
                api_url="https://example.invalid/v1/responses",
                api_key_env="OPENAI_API_KEY",
                env_file=".env.missing",
                api_format="responses",
                temperature=0.0,
                timeout_seconds=5,
                max_retries=1,
                retry_base_seconds=0.0,
            ),
            post=fake_post,
            sleep=lambda _: None,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["_request_id"], "req_123")
        self.assertEqual(result["_retry_attempts"], 1)

    def test_post_llm_request_rejects_oversized_response(self) -> None:
        def fake_post(*args, **kwargs):
            return FakeResponse(200, payload={"ok": True}, text="x" * 20)

        with self.assertRaisesRegex(RuntimeError, "response exceeded"):
            post_llm_request(
                LLMRequest(
                    prompt="Review this.",
                    model="test-model",
                    api_url="https://example.invalid/v1/responses",
                    api_key_env="OPENAI_API_KEY",
                    env_file=".env.missing",
                    api_format="responses",
                    temperature=0.0,
                    timeout_seconds=5,
                    max_response_bytes=10,
                ),
                post=fake_post,
                sleep=lambda _: None,
            )


if __name__ == "__main__":
    unittest.main()


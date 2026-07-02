# llm_review/clients

这里封装 LLM 调用客户端。

## 数据流动

```text
prompt
  -> llm_client.py
  -> raw model response
  -> parsing/
```

客户端层只负责调用和返回响应，不应承担业务判断。业务判断放在 parsing、validators 或 auto_qc。

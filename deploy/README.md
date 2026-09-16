# 远程部署维护文档

服务使用 PostgreSQL、pgvector 和 MCP，不依赖本地 FAISS 文件。

## 配置

将 DSN 写入服务器的 /etc/pdf-markdown-rag/postgres.env，不要写入仓库或 systemd 单元。

~~~bash
sudo install -d -m 750 -o lhj -g lhj /etc/pdf-markdown-rag
sudo cp deploy/postgres.env.example /etc/pdf-markdown-rag/postgres.env
sudo chown lhj:lhj /etc/pdf-markdown-rag/postgres.env
sudo chmod 600 /etc/pdf-markdown-rag/postgres.env
~~~

## vLLM 模型服务

MCP 进程不再加载模型，需在同一台 48GB GPU 服务器上分别启动 embedding 和 reranker 服务：

~~~bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-8B \
  --runner pooling \
  --served-model-name qwen3-embedding \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.50 \
  --port 8001

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Reranker-4B \
  --runner pooling \
  --served-model-name qwen3-reranker \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.30 \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --port 8002
~~~

生产环境应将二者配置为独立守护服务。MCP 使用 `VLLM_EMBEDDING_BASE_URL` 和
`VLLM_RERANKER_BASE_URL` 连接它们；若启用了 vLLM API key，只通过环境变量提供，
不要写入仓库。两个服务可共用 `VLLM_API_KEY`，也可分别设置
`VLLM_EMBEDDING_API_KEY` 和 `VLLM_RERANKER_API_KEY`。

文档检索会并行执行两条 PostgreSQL 召回 SQL，因此 `MCP_MAX_CONCURRENT` 不应超过
`PG_POOL_MAX / 2`。示例服务使用 `PG_POOL_MAX=16` 与 `MCP_MAX_CONCURRENT=8`；程序也会
在启动时自动将更大的 MCP 并发配置限制到该安全上限，避免请求在连接池内排队。

## 建库顺序

先创建表，再导入数据，然后向量化 cards、views、chunks，最后执行 index-vectors 创建 IVFFlat 索引。服务设置 RAG_BACKEND=postgres 和 PG_VECTOR_RETRIEVAL_REQUIRED=1。
## 本地通过 SSH 发起建库

先通过 VS Code SSH-Remote、`scp` 或 `rsync` 分别上传代码、`data/evidence` 和模型缓存。下面的脚本不上传文件，也不接收数据库密码；它复用本机 SSH config/密钥代理，并在远端读取 `/etc/pdf-markdown-rag/postgres.env`。

~~~powershell
python -B scripts/remote_pg_build.py your-ssh-alias --dry-run
python -B scripts/remote_pg_build.py your-ssh-alias
~~~

默认远端项目目录为 `/home/lhj/project/evidence_generation`，数据目录为项目下的 `data/evidence`，模型缓存为 `/data/lhj/huggingface`。路径不同时使用 `--project-dir`、`--data-dir` 和 `--hf-home` 覆盖。

脚本依次执行建表、流式入库、cards/views/chunks 的分页 Qwen3-Embedding-8B 向量化、向量索引创建和 `verify`。中断后若 PostgreSQL 数据快照未变化，可用 `--skip-ingest` 只补缺失向量；不要在上传了新数据后使用该参数。

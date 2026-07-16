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

## 建库顺序

先创建表，再导入数据，然后向量化 cards、views、chunks，最后执行 index-vectors 创建 IVFFlat 索引。服务设置 RAG_BACKEND=postgres 和 PG_VECTOR_RETRIEVAL_REQUIRED=1。
## 本地通过 SSH 发起建库

先通过 VS Code SSH-Remote、`scp` 或 `rsync` 分别上传代码、`data/evidence` 和模型缓存。下面的脚本不上传文件，也不接收数据库密码；它复用本机 SSH config/密钥代理，并在远端读取 `/etc/pdf-markdown-rag/postgres.env`。

~~~powershell
python -B scripts/remote_pg_build.py your-ssh-alias --dry-run
python -B scripts/remote_pg_build.py your-ssh-alias
~~~

默认远端项目目录为 `/home/lhj/project/evidence_generation`，数据目录为项目下的 `data/evidence`，模型缓存为 `/data/lhj/huggingface`。路径不同时使用 `--project-dir`、`--data-dir` 和 `--hf-home` 覆盖。

脚本依次执行建表、流式入库、cards/views/chunks 的分页 BGE-M3 向量化、向量索引创建和 `verify`。中断后若 PostgreSQL 数据快照未变化，可用 `--skip-ingest` 只补缺失向量；不要在上传了新数据后使用该参数。

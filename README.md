# Medical Guideline RAG 操作指南

本项目流程：

```text
原始指南数据 -> 清洗切分 -> 补充 quality/recommendation -> MedCPT 向量化 -> Qdrant 检索
```

## 1. 安装依赖

```powershell
pip install qdrant-client transformers torch numpy
```

检查 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. 生成 Chunk

```powershell
python .\src\segment.py --input .\data\raw\data.jsonl --output .\outputs\chunk.jsonl --min-chars 50
```

当前 chunk 文件：

```text
outputs/chunk.jsonl
```

每条 chunk 包含：

```text
page_content
metadata.title
metadata.url
metadata.published_date
metadata.medical_topic
metadata.chunk_index
metadata.char_len
metadata.quality
metadata.recommendation
```

## 3. 启动 Qdrant

保持该窗口运行：

```powershell
docker run -p 6333:6333 -p 6334:6334 `
  -v ${PWD}\qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

Dashboard：

```text
http://localhost:6333/dashboard
```

## 4. 入库

新开 PowerShell：

```powershell
cd D:\python\Data_splitting_ingestion
conda activate bert_env
python -m src.storage ingest --input .\outputs\chunk.jsonl --recreate
```

说明：

```text
文档入库使用 ncbi/MedCPT-Article-Encoder
查询检索使用 ncbi/MedCPT-Query-Encoder
Qdrant collection 名称为 medical_guidelines
```

## 5. 检索

普通检索：

```powershell
python -m src.storage search "diabetes treatment recommendation" --top-k 5
```

按指定年份：

```powershell
python -m src.storage search "heart failure recommendation" --top-k 5 --year 2023
```

按年份范围：

```powershell
python -m src.storage search "hypertension treatment" --top-k 5 --year-from 2020 --year-to 2024
```

按发布机构：

```powershell
python -m src.storage search "cancer screening" --issuer "WHO"
```

## 6. 多电脑使用

如果 Qdrant 跑在另一台机器，只需设置：

```powershell
$env:QDRANT_URL="http://服务器IP:6333"
python -m src.storage search "diabetes treatment recommendation"
```

其他电脑只检索时，不需要重新切分和重新入库。

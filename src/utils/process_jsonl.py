import json
from typing import List,Any,Dict
from pathlib import Path

# 读取文件 ###############################################
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Line {line_no} JSON parse error: {exc}")
    return records
###########################################################


# 保存文件 #################################################
def save_jsonl(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for doc in result:
            item = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
############################################################
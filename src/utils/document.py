from typing import List,Any,Dict
from dataclasses import dataclass,field
import re

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# 将数据初步处理为Document形式 ##############################
def keep_year(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"(19|20)\d{2}", str(value).strip())
    return match.group(0) if match else ""


def process_data(data: List[Dict[str, Any]]) -> List[Document]:
    result = []
    for sample in data:
        result.append(
            Document(
                page_content=sample.get("content") or "",
                metadata={
                    "published_date": keep_year(sample.get("published_date")),
                    "title": sample.get("title") or "",
                    "medical_topic": sample.get("medical_topics") or "",
                    "url": sample.get("url") or "",
                    "source": sample.get("source") or ""
                },
            )
        )
    return result
###########################################################
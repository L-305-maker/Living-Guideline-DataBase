import hashlib
from typing import List

from src.utils.document import Document


# 指南去重 #################################################
def hash_text(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicator(data: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for document in data:
        content_hash = hash_text(document.page_content)
        if content_hash in seen:
            continue
        seen.add(content_hash)
        unique.append(document)
    return unique
###########################################################
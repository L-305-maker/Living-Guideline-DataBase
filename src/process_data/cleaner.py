import re
from typing import List

from src.utils.document import Document


# 数据清洗 #################################################
class DocumentCleaner:
    MOJIBAKE_REPLACEMENTS = {
        "\u920d?": ">=",
        "\u920d\ufffd": ">=",
        "\u920d\u6a9a": "'s",
        "\u920d\u6a9b": "'t",
        "\u920d\u6a99": "'r",
        "\u920d\u6a9d": "'v",
        "\u920d\u6a91": "'l",
        "\u920d\uff1f": "'",
        "\u76f2": "a",
        "\ufffd": "",
    }

    def clean_text(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        for bad, good in self.MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)

        text = re.sub(r"(?im)^\s*#+\s*", "", text)
        text = re.sub(r"(?im)^\s*[-*•]\s+", "", text)
        text = re.sub(r"(?im)^\s*(Path|Citation|Footnotes?|References?)\s*$", "", text)
        text = re.sub(r"(?im)^\s*(\[\d+\]|\(\w\)|[a-z]|\d+)\s*$", "", text)
        text = re.sub(r"(?i)\bCitation\s+", "", text)
        text = re.sub(r"\s+\[\d+\]\s+", " ", text)
        text = re.sub(r"\s+\([a-z]\)\s+", " ", text)
        text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]])", r"\1", text)
        return text.strip()

    def clean(self, documents: List[Document]) -> List[Document]:
        cleaned_docs = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if len(cleaned_text) < 20:
                continue
            doc.page_content = cleaned_text
            cleaned_docs.append(doc)
        return cleaned_docs
###########################################################
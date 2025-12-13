"""
封装vectorStore检索到的结果和重排序后的结果
"""
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class SearchResult:
    """搜索结果"""
    document: Document
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.document.page_content,
            "metadata": self.document.metadata,
            "score": self.score,
            "rank": self.rank
        }

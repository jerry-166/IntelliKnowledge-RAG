"""
封装 vectorStore、bm25关键词、混合检索 检索到的结果和重排序后的结果
"""
from dataclasses import dataclass, field
from typing import Any, Optional, List
from langchain_core.documents import Document


@dataclass
class SearchResult:
    """搜索结果"""
    document: Document
    score: float
    rank: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.document.page_content,
            "metadata": self.document.metadata,
            "score": self.score,
            "rank": self.rank
        }


@dataclass
class KeywordSearchResult:
    """
    检索结果
    """
    document: Document
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)  # 存储匹配到的词
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.document.page_content,
            "metadata": self.document.metadata,
            "score": self.score,
            "rank": self.rank,
            "matched_terms": self.matched_terms
        }


@dataclass
class HybridSearchResult:
    """混合检索结果"""
    document: Document
    score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    matched_terms: List[str] = None
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.document.page_content,
            "metadata": self.document.metadata,
            "score": self.score,
            "rank": self.rank,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "matched_terms": self.matched_terms
        }

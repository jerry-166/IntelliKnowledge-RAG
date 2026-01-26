from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

# 创建路由实例
router = APIRouter()


# 搜索请求模型
class SearchRequest(BaseModel):
    query: str
    collection_id: Optional[int] = None
    top_k: int = 5
    rerank: bool = True
    search_type: str = "hybrid"


# 搜索结果模型
class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    document_id: str
    document_name: str
    chunk_id: str
    metadata: Optional[dict] = None


# 搜索响应模型
class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    search_type: str
    execution_time: float


# 模拟搜索结果
mock_search_results = [
    SearchResult(
        id="1",
        content="这是测试文档的第一段内容，包含了搜索关键词",
        score=0.95,
        document_id="1",
        document_name="test_document.pdf",
        chunk_id="chunk_1",
        metadata={"page": 1, "section": "introduction"}
    ),
    SearchResult(
        id="2",
        content="这是测试文档的第二段内容，也包含了相关信息",
        score=0.88,
        document_id="1",
        document_name="test_document.pdf",
        chunk_id="chunk_2",
        metadata={"page": 2, "section": "methodology"}
    )
]


# 向量检索
@router.post("/vector", response_model=SearchResponse)
def vector_search(request: SearchRequest):
    """向量检索"""
    return SearchResponse(
        results=mock_search_results,
        total=len(mock_search_results),
        search_type="vector",
        execution_time=0.123
    )


# 混合检索
@router.post("/hybrid", response_model=SearchResponse)
def hybrid_search(request: SearchRequest):
    """混合检索"""
    return SearchResponse(
        results=mock_search_results,
        total=len(mock_search_results),
        search_type="hybrid",
        execution_time=0.156
    )

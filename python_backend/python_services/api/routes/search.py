import time
from typing import List, Optional, Literal
from fastapi import APIRouter
from pydantic import BaseModel

# 创建路由实例
router = APIRouter()

# 延迟初始化
rag_config = None
vector_store = None
hybrid_retriever = None


def get_rag_config():
    global rag_config
    if rag_config is None:
        from python_services.core.settings import get_config
        rag_config = get_config()
    return rag_config


def get_vector_store():
    global vector_store
    if vector_store is None:
        from python_services.vector_store.multimodal_store import MultimodalVectorStore
        vector_store = MultimodalVectorStore(get_rag_config())
    return vector_store


def get_hybrid_retriever():
    global hybrid_retriever
    if hybrid_retriever is None:
        from python_services.retriever.hybrid_retriever import HybridRetriever
        hybrid_retriever = HybridRetriever(get_vector_store(), get_rag_config().keyword_search)
    return hybrid_retriever


# 搜索请求模型
class SearchRequest(BaseModel):
    query: str
    collection_id: Optional[int] = None
    top_k: int = 5
    rerank: bool = True
    cache: Optional[bool] = False
    search_type: Literal["text", "image", "hybrid"] = "hybrid"
    filter_dict: Optional[dict] = None


# 搜索结果模型
# class SearchResult(BaseModel):
#     id: str
#     content: str
#     score: float
#     document_id: str
#     document_name: str
#     chunk_id: str
#     metadata: Optional[dict] = None


# 搜索响应模型
class SearchResponse(BaseModel):
    results: List
    total: int
    search_type: str
    execution_time: float


# 模拟搜索结果
mock_search_results = [
    {"id": "1", "content": "这是测试文档的第一段内容，包含了搜索关键词", "score": 0.95, "document_id": "1",
     "document_name": "test_document.pdf", "chunk_id": "chunk_1", "metadata": {"page": 1, "section": "introduction"}},
    {"id": "2", "content": "这是测试文档的第二段内容，也包含了相关信息", "score": 0.88, "document_id": "1",
     "document_name": "test_document.pdf", "chunk_id": "chunk_2", "metadata": {"page": 2, "section": "methodology"}}
]


# 向量检索
@router.post("/vector", response_model=SearchResponse)
def vector_search(request: SearchRequest):
    """向量检索"""
    # 计时
    start = time.perf_counter()

    # 使用向量检索
    search_results = get_vector_store().search(
        query=request.query, top_k=request.top_k, filter_dict=request.filter_dict,
        search_type=request.search_type, use_reranker=request.rerank,
        use_cache=request.cache
    )

    cost = time.perf_counter() - start

    return SearchResponse(
        results=search_results,
        total=len(search_results),
        search_type="vector",
        execution_time=cost
    )


# 混合检索
@router.post("/hybrid", response_model=SearchResponse)
def hybrid_search(request: SearchRequest):
    """混合检索"""
    # 计时
    start = time.perf_counter()

    # 使用混合检索
    search_results = get_hybrid_retriever().search(
        request.query, request.top_k, request.filter_dict,
    )

    cost = time.perf_counter() - start

    return SearchResponse(
        results=search_results,
        total=len(search_results),
        search_type="hybrid",
        execution_time=cost
    )

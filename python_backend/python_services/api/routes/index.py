import os.path
from typing import Optional, Literal, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from python_services.api.routes.common import SuccessResponse
from python_services.core.search_results import HybridSearchResult, SearchResult
from python_services.core.settings import get_config

router = APIRouter()

# 延迟初始化pipeline，避免在模块导入时执行
pipeline = None


def get_pipeline():
    """获取RAGPipeline实例，延迟初始化"""
    global pipeline
    if pipeline is None:
        from python_services.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(config=get_config())
    return pipeline


class Query(BaseModel):
    conversation_id: int
    content: str
    top_k: int
    use_cache: bool
    search_type: Literal["text", "image", "hybrid"]
    metadata: Optional[dict] = None  # 存入用户id，用来区分知识库所属


class Response(BaseModel):
    text: str
    score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    results: List[Response]
    total: int


@router.post("/build", response_model=SuccessResponse)
def build_index(file_path: str):
    """构建索引（切分+向量化+存盘）"""
    # 这里的file_path是加强文档，需前端注意
    if not os.path.exists(file_path):
        return {
            "success": False,
            "message": f"不存在文件：{file_path}"
        }

    data = get_pipeline().split_and_store(file_path=file_path, show_progress=True)
    chunks = data.get('chunks', [])
    doc_ids = data.get('doc_ids', [])

    print(f"添加{len(chunks)}个片段成功")
    return SuccessResponse(
        message=f"构建索引成功",
        data=data,
    )


@router.post("/search", response_model=SearchResponse)
def search_topk(query: Query):
    """搜索topk个结果，并返回list[Response]"""
    results = get_pipeline().search(
        query=query.content,
        top_k=query.top_k,
        filter_dict={"user_id": query.metadata.get("user_id", None)},
        search_type=query.search_type,
        use_cache=query.use_cache,
    )
    response_items = []
    if isinstance(results[0], HybridSearchResult):
        response_items = [
            Response(
                text=res.document.content,
                score=res.score,
                metadata={
                    "vector_score": res.vector_score,
                    "keyword_score": res.keyword_score,
                    "vector_rank": res.vector_rank,
                    "keyword_rank": res.keyword_rank,
                    "matched_terms": res.matched_terms,
                    "rank": res.rank
                }
            )
            for res in results
        ]
    elif isinstance(results[0], SearchResult):
        response_items = [
            Response(
                text=res.document.content,
                score=res.score,
                metadata={
                    "rank": res.rank,
                }
            )
            for res in results
        ]
    else:
        raise HTTPException(status_code=500, detail="响应格式错误")

    return SearchResponse(results=response_items, total=len(response_items))

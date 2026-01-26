import os.path
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from python_services.core.search_results import HybridSearchResult, SearchResult
from python_services.core.settings import get_config
from python_services.rag_pipeline import RAGPipeline

router = APIRouter()

pipeline = RAGPipeline(config=get_config())


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


@router.post("/build")
def build_index(file_path):
    """构建索引（切分+向量化+存盘）"""
    if not os.path.exists(file_path):
        return {
            "success": False,
            "message": f"不存在文件：{file_path}"
        }

    chunks = pipeline.ingest(file_path, True)
    print(f"添加{len(chunks)}个片段")
    return JSONResponse(content=chunks)


@router.post("/search", response_model=Response)
def search_topk(query: Query):
    """搜索topk个结果，并返回list[Response]"""
    results = pipeline.search(
        query=query.content,
        top_k=query.top_k,
        filter_dict={"user_id": query.metadata.get("user_id")},
        search_type=query.search_type,
        use_cache=query.use_cache,
    )
    if isinstance(results[0], HybridSearchResult):
        response = [
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
        response = [
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

    return response

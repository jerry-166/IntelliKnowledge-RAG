from fastapi import APIRouter
from pydantic import BaseModel

# 创建路由实例
router = APIRouter()


# 统计响应模型
class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_conversations: int
    total_messages: int
    avg_embedding_time: float
    avg_search_time: float
    storage_usage: dict


# 集合统计模型
class CollectionStatsResponse(BaseModel):
    collection_id: int
    collection_name: str
    total_documents: int
    total_chunks: int
    avg_chunk_size: int
    storage_usage: dict
    document_types: dict


# 获取系统统计
@router.get("/overview", response_model=StatsResponse)
def get_overview_stats():
    """获取系统统计概览"""
    return StatsResponse(
        total_documents=42,
        total_chunks=1500,
        total_conversations=120,
        total_messages=580,
        avg_embedding_time=0.123,
        avg_search_time=0.056,
        storage_usage={
            "documents": "12.5 MB",
            "vectors": "8.2 MB",
            "conversations": "1.3 MB"
        }
    )


# 获取知识库统计
@router.get("/collections/{collection_id}", response_model=CollectionStatsResponse)
def get_collection_stats(collection_id: str):
    """获取知识库统计"""
    return CollectionStatsResponse(
        collection_id=collection_id,
        collection_name=f"知识库 {collection_id}",
        total_documents=20,
        total_chunks=750,
        avg_chunk_size=512,
        storage_usage={
            "documents": "6.2 MB",
            "vectors": "4.1 MB"
        },
        document_types={
            "pdf": 12,
            "txt": 5,
            "md": 3
        }
    )

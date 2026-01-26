import datetime
import os.path
import uuid

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from typing import List, Optional, Any, Literal

from fastapi.params import Path
from langchain_core.documents import Document
from pydantic import BaseModel

from basic_core.llm_factory import qwen_vision
from python_services.core.settings import get_config
from python_services.rag_pipeline import RAGPipeline

# 创建路由实例
router = APIRouter()


# 文档模型
class MyDocument(BaseModel):
    id: str
    file_name: str
    file_type: str
    file_size: int
    status: str  # 文档状态？？？
    created_at: str
    collection_id: Optional[int] = None
    metadata: Optional[dict] = None


# 上传文档请求模型
class UploadDocumentRequest(BaseModel):
    file: UploadFile
    collection_id: int
    metadata: Optional[dict] = None


# 上传文档响应模型
class UploadDocumentResponse(BaseModel):
    document_id: str
    file_name: str
    message: Any


# 文档数据
documents = []


# 获取文档列表
@router.get("", response_model=List[MyDocument])
def get_documents(
        collection_id: Optional[int] = Query(None, description="知识库ID"),
        status: Optional[str] = Query(None, description="文档状态")
):
    """获取文档列表"""
    # 根据用户id过滤属于用户自己的docs(存储于我端数据库的)
    # 存储到内存的documents
    filtered_docs = documents

    if collection_id:
        filtered_docs = [doc for doc in filtered_docs if doc.collection_id == collection_id]

    if status:
        filtered_docs = [doc for doc in filtered_docs if doc.status == status]

    return filtered_docs


def upload_doc(file: str) -> list[Document]:
    pipeline = RAGPipeline(config=get_config(), vision_llm=qwen_vision)
    fdocuments = pipeline.parse_file(file)  # todo:后台处理
    Mydocument = MyDocument(
        id=uuid.uuid4(),
        file_name=file.split('/')[-1].split('.')[0],
        file_type=file.split('.')[-1],
        created_at=datetime.datetime.now(),
        collection_id=None
    )
    documents.append(Mydocument)
    return fdocuments


# 上传文档
@router.get("/page")
def get_png(
    type: Literal['original', 'argument'],
    file_id: str = Query(..., description="文件id"),
    page: int = Query(..., ),
) -> Optional[bytes]:
    """获取文档的png图片"""
    file_name = f"{file_id}/{type}/{page}"
    if not os.path.exists(file_name):
        return None
    with open(file_name, "rb") as f:
        data = f.read()
    os.path
    return None





@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
        file: UploadFile = File(...),
):
    """上传文档"""
    fdocuments = upload_doc(file.filename)

    return UploadDocumentResponse(
        document_id=uuid.uuid4(),
        file_name=file.filename,
        message=fdocuments[0].page_content,
    )
# 批量上传文档


@router.post("/batch-upload")
async def batch_upload_documents(
        files: List[UploadFile] = File(...),
        collection_id: int = Form(...)
):
    """批量上传文档"""
    # 批量文件处理逻辑(多线程？)
    for file in files:
        upload_doc(file.filename)

    # 返回数据
    return {
        "success": True,
        "message": f"成功上传 {len(files)} 个文件",
        "uploaded_files": [file.filename for file in files]
    }
# 获取文档详情
@router.get("/{doc_id}", response_model=MyDocument)
def get_document(doc_id: str):
    """获取文档详情"""
    doc = next((d for d in documents if d.id == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


# 获取文档处理状态
@router.get("/{doc_id}/status")
def get_document_status(doc_id: str):
    """获取文档处理状态"""
    doc = next((d for d in documents if d.id == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "document_id": doc_id,
        "status": doc.status,
        "progress": 100 if doc.status == "completed" else 0,
        "message": "文档处理完成" if doc.status == "completed" else "文档处理中"
    }


# 删除文档
@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """删除文档"""
    global documents
    documents = [d for d in documents if d.id != doc_id]
    return {
        "success": True,
        "message": "文档删除成功"
    }

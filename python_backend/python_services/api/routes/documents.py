import datetime
import os.path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Literal, Any
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from pydantic import BaseModel
from python_services.api.routes.common import SuccessResponse, ErrorResponse
from python_services.core.settings import get_config


# 延迟导入，避免在模块导入时初始化

def get_qwen_vision():
    from basic_core.llm_factory import qwen_vision
    return qwen_vision


def get_rag_pipeline():
    from python_services.rag_pipeline import RAGPipeline
    return RAGPipeline


# 创建路由实例
router = APIRouter()


# 文档模型
class MyDocument(BaseModel):
    id: str
    file_name: str
    file_type: str
    file_size: int
    file_id: str
    status: str
    created_at: str
    collection_id: Optional[str] = None
    metadata: Optional[dict] = None


# 上传文档响应模型
class UploadFileResponse(BaseModel):
    file_id: str
    file_name: str


# 文档数据
documents = []


# 获取知识库下所有文档列表
@router.get("", response_model=List[MyDocument])
def get_collection_documents(
        collection_id: Optional[str] = Query(None, description="知识库ID"),
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


def upload_file(file: str, collection_id: Optional[str] = None) -> dict[str, Any]:
    RAGPipeline = get_rag_pipeline()
    qwen_vision = get_qwen_vision()
    pipeline = RAGPipeline(config=get_config(), vision_llm=qwen_vision)
    # 生成file_id
    file_name = file.split('/')[-1].split('.')[0]
    file_id = str(uuid.uuid4())
    fdocuments = pipeline.parse_file(file)  # todo:后台处理
    Mydocument = MyDocument(
        id=str(uuid.uuid4()),  # 文档id
        file_name=file_name,
        file_type=file.split('.')[-1],
        file_id=file_id,  # 文件id
        file_size=0,
        status="uploaded",
        created_at=datetime.datetime.now().isoformat(),
        collection_id=collection_id
    )
    documents.append(Mydocument)
    return {
        "documents": fdocuments,
        "file_id": file_id
    }


# 获取文件页（分为加强页和普通页）
@router.get("/page")
def get_file_png(
        type: Literal['original', 'argument'],
        file_id: str = Query(..., description="文件id"),
        page: int = Query(..., description="文件页数"),
        collection_id: Optional[str] = Query(None, description="知识库id"),
) -> Optional[bytes]:
    """获取文档的png图片"""
    # 根据file_id找到file_name
    file_name = next(
        (doc.file_name for doc in documents if doc.file_id == file_id and doc.collection_id == collection_id), None)
    file_name = f"./output/{file_name}/image/{type}/{page}.png"
    if not os.path.exists(file_name):
        return None
    with open(file_name, "rb") as f:
        data = f.read()
    return data


# 上传文件（不做切分...）
@router.post("/upload", response_model=SuccessResponse)
async def upload_file_documents(
        file: UploadFile = File(...),
        collection_id: str = Form(...)
):
    """上传文档"""
    try:
        # 保存上传的文件到临时目录
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 处理文档
        data = upload_file(file_path, collection_id)
        fdocuments = data.get('documents', [])
        file_id = data.get('file_id', str(uuid.uuid4()))

        # 清理临时文件
        os.remove(file_path)
        message = fdocuments[0].page_content[:200] + "..." if fdocuments else "No content"
        return SuccessResponse(
            message="文档上传成功",
            data=UploadFileResponse(
                file_id=file_id, file_name=file.filename
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(ErrorResponse(
                error_code="DOCUMENT_UPLOAD_FAILED",
                message="文档上传失败",
                details=str(e),
                solution="请检查文件格式和大小，或联系管理员"
            ))
        )


# 批量上传文档
@router.post("/batch-upload")
async def batch_upload_files_documents(
        files: List[UploadFile] = File(...),
        collection_id: str = Form(...)
):
    """批量上传文档"""
    uploaded_files = []
    # 批量文件处理逻辑(多线程)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(upload_file_documents, file, collection_id): files
            for file in files
        }
        for future in as_completed(futures):
            try:
                filename = future.result().data.file_name
                uploaded_files.append(filename)
            except Exception as e:
                print(f"处理文件失败: {e}")

    # 返回数据
    return SuccessResponse(
        message=f"成功上传 {len(files)} 个文件",
        data={
            "uploaded_files": uploaded_files
        }
    )


# 获取文档详情
@router.get("/{doc_id}", response_model=SuccessResponse)
def get_document(doc_id: str, collection_id: Optional[str] = None):
    """获取文档详情"""
    doc = next((d for d in documents if d.id == doc_id and d.collection_id == collection_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return SuccessResponse(
        message="获取文档详情成功",
        data=doc,
    )


# 获取文件处理状态
@router.get("/{file_id}/status")
def get_file_documents_status(file_id: str, collection_id: Optional[str] = None):
    """获取文档处理状态"""
    docs = [d for d in documents if d.file_id == file_id and collection_id == d.collection_id]
    if not docs:
        raise HTTPException(status_code=404, detail="文档不存在")

    status = 'completed'
    for doc in docs:
        if doc.status != 'completed':
            status = 'uncompleted'
    return SuccessResponse(
        message="文档处理完成" if status == "completed" else "文档处理中",
        data={
            "file_id": file_id,
            "status": status,
        }
    )


# 删除文件
@router.delete("/{file_id}")
def delete_file_documents(file_id: str, collection_id: Optional[str] = None):
    """删除文档"""
    global documents
    documents = [d for d in documents if d.file_id != file_id and d.collection_id == collection_id]  # 文档id == file_id
    return SuccessResponse(
        message="删除文档成功"
    )

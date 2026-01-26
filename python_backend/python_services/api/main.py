from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from python_services.core import settings
from python_services.api.routes import documents, search, chat, settings as settings_router, statistics, index

# 创建FastAPI应用实例
app = FastAPI(
    title="RAG智能问答系统API",
    description="基于FastAPI的RAG系统后端API",
    version="1.0.0",
    docs_url="/docs",  # ??? --- swagger文档的url
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(index.router, prefix="/api/v1/index", tags=["index"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(settings_router.router, prefix="/api/v1/settings", tags=["settings"])
app.include_router(statistics.router, prefix="/api/v1/statistics", tags=["statistics"])


# 健康检查端点
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "RAG API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "python_services.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

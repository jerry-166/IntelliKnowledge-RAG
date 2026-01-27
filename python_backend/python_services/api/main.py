import sys
import os

# 添加python_backend目录到Python路径，解决模块导入问题
# 计算main.py所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算python_backend目录（包含python_services的目录）
python_backend_dir = os.path.abspath(os.path.join(current_dir, '../..'))
# 添加到Python路径
sys.path.append(python_backend_dir)
print(f"已添加python_backend目录到Python路径: {python_backend_dir}")
print(f"Python路径中是否包含python_backend: {python_backend_dir in sys.path}")
print(f"Python路径长度: {len(sys.path)}")

# 测试基本导入
try:
    print("\n=== 开始导入必要模块 ===")
    print("1. 导入fastapi模块...")
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    print("✅ 成功导入fastapi模块")
    
    print("2. 导入python_services模块...")
    import python_services
    print("✅ 成功导入python_services模块")
    
    print("3. 导入common模块...")
    from python_services.api.routes.common import ErrorResponse
    print("✅ 成功导入common模块")
    
    print("4. 导入路由模块...")
    # 逐个导入路由模块，找出具体问题
    print("   4.1 导入documents模块...")
    from python_services.api.routes import documents
    print("   ✅ 成功导入documents模块")
    
    print("   4.2 导入search模块...")
    from python_services.api.routes import search
    print("   ✅ 成功导入search模块")
    
    print("   4.3 导入chat模块...")
    from python_services.api.routes import chat
    print("   ✅ 成功导入chat模块")
    
    print("   4.4 导入settings模块...")
    from python_services.api.routes import settings as settings_router
    print("   ✅ 成功导入settings模块")
    
    print("   4.5 导入statistics模块...")
    from python_services.api.routes import statistics
    print("   ✅ 成功导入statistics模块")
    
    print("   4.6 导入index模块...")
    from python_services.api.routes import index
    print("   ✅ 成功导入index模块")
    
    print("✅ 成功导入所有路由模块")
    
    print("\n=== 所有模块导入成功 ===")
except Exception as e:
    print(f"❌ 导入错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from python_services.api.routes import documents, search, chat, settings as settings_router, statistics, index
from python_services.api.routes.common import ErrorResponse

# 创建FastAPI应用实例
app = FastAPI(
    title="RAG智能问答系统API",
    description="基于FastAPI的RAG系统后端API",
    version="1.0.0",
    docs_url="/docs",
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


# 全局异常处理中间件
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    error_response = ErrorResponse(
        error_code="INTERNAL_SERVER_ERROR",
        message="服务器内部错误",
        details=str(exc),
        solution="请稍后重试，或联系管理员"
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
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

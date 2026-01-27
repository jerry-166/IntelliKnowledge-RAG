from fastapi import APIRouter, HTTPException, BackgroundTasks

from python_services.api.routes.common import SuccessResponse, ErrorResponse
from python_services.core.settings import get_config, RAGConfig, refresh_config

# 创建路由实例
router = APIRouter()

from pathlib import Path

# 获取配置文件的绝对路径
RAG_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
rag_config = None


# 获取配置
@router.get("", response_model=RAGConfig)
def get_settings():
    """获取RAG配置"""
    global rag_config
    rag_config = get_config(RAG_CONFIG_PATH)
    return rag_config


# 更新配置
@router.put("/update", response_model=SuccessResponse)
def update_settings(config: RAGConfig, background_tasks: BackgroundTasks):
    """更新RAG配置"""
    try:
        # 更新settings属性
        global rag_config
        rag_config = config
        # 保存本地(后台任务,返回响应后执行)
        background_tasks.add_task(RAGConfig.save_to_yaml, config, str(RAG_CONFIG_PATH))
        # 刷新配置缓存，使下次调用get_config时返回新配置
        refresh_config()
        return SuccessResponse(
            message="配置更新成功",
            data=config.model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(ErrorResponse(
                error_code="CONFIG_UPDATE_FAILED",
                message="配置更新失败",
                details=str(e),
                solution="请检查配置格式是否正确，或联系管理员"
            ))
        )

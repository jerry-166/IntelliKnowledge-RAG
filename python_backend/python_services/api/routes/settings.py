from fastapi import APIRouter, HTTPException

from python_services.core.settings import get_config, RAGConfig, refresh_config

# 创建路由实例
router = APIRouter()

RAG_CONFIG_PATH = "../config.yaml"
rag_config = None


# 获取配置
@router.get("", response_model=RAGConfig)
def get_settings():
    """获取RAG配置"""
    global rag_config
    rag_config = get_config(RAG_CONFIG_PATH)
    return rag_config


# 更新配置
@router.put("/update", response_model=RAGConfig)
def update_settings(config: RAGConfig):
    """更新RAG配置"""
    try:
        # 更新settings属性
        global rag_config
        rag_config = config
        # 保存本地(放在其他线程上，并在FASTAPI结束后保存)
        RAGConfig.save_to_yaml(config, RAG_CONFIG_PATH)
        # 刷新配置缓存，使下次调用get_config时返回新配置
        refresh_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return config

"""
向量存储基类
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.documents import Document
from basic_core.search_results import SearchResult

"""
1. 为什么要**kwargs？
    扩展性: 允许子类实现时添加额外的参数，而不需要修改基类接口
    灵活性: 不同的向量存储实现可能需要不同的参数配置
    向后兼容: 添加新参数时不会破坏现有代码
2. ID列表吗？
    ID 映射: Faiss 在检索时会返回与向量关联的 ID 列表
    向量标识: 每个存储的向量都有唯一的标识符（通常是整数 ID）
"""


class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def add_documents(self, documents: list[Document], **kwargs) -> list[str]:
        """添加文档，返回ids"""
        pass

    @abstractmethod
    def update(self, id: str, document: Document) -> bool:
        """根据ID更新Document"""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> bool:
        """根据ID列表删除"""
        pass

    @abstractmethod
    def search(
            self,
            query: str,
            top_k: int = 10,
            filter_dict: Optional[dict[str, Any]] = None,
            **kwargs
    ) -> list[SearchResult]:
        """搜索"""
        pass

    @abstractmethod
    def persist(self, persist_directory: str):
        """持久化到对应的目录中"""
        pass

    @abstractmethod
    def load(self, persist_directory: str):
        """加载向量库"""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """获取统计结果"""
        pass

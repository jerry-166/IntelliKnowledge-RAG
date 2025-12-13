"""
缓存基类
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any

logger = logging.getLogger(__name__)


class BaseCacheBackend(ABC):
    """
    缓存基类
    增（改）
    删
    清空
    查
    统计
    """
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """查"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """增/改"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """清空"""
        pass

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """统计"""
        pass

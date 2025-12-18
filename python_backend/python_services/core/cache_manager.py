"""
缓存管理器 - 支持多种后端
"""
import logging
from typing import Any, Optional, Dict
from basic_core.llm_factory import init_embedding_basic
from python_services.caches.disk_cache import DiskCacheBackend
from python_services.caches.memory_cache import MemoryCacheBackend
from python_services.caches.redis_cache import RedisCacheBackend
from python_services.core.settings import CacheConfig

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器"""
    def __init__(self, config: CacheConfig):
        self.config = config
        self._init_backend()

    def _init_backend(self):
        """初始化缓存后端"""
        if self.config.backend == "memory":
            self.backend = MemoryCacheBackend(
                max_size=self.config.max_size,
                ttl=self.config.ttl,
                embedding=init_embedding_basic(model_name=self.config.embedding),
                score_threshold=self.config.semantic_threshold,
            )
        elif self.config.backend == "redis":
            if not self.config.redis_url:
                raise ValueError("Redis URL is required")
            self.backend = RedisCacheBackend(
                url=self.config.redis_url,
                ttl=self.config.ttl,
                prefix=self.config.redis_prefix,
                embedding=init_embedding_basic(model_name=self.config.embedding),
                semantic_threshold=self.config.semantic_threshold,
                faiss_path=self.config.faiss_persist_directory,
            )
        elif self.config.backend == "disk":
            self.backend = DiskCacheBackend(
                cache_dir=self.config.cache_dir,
                ttl=self.config.ttl,
                max_size=self.config.max_size,
            )
        else:
            raise ValueError(f"Unknown cache backend: {self.config.backend}")

        logger.info(f"✅️Cache 初始化成功 - backend: {self.config.backend}")

    def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return self.backend.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        return self.backend.delete(key)

    def clear(self) -> bool:
        return self.backend.clear()

    def stats(self) -> Dict[str, Any]:
        return self.backend.stats()

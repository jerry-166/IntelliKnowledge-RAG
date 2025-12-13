"""
Redis 缓存
"""
import logging
import pickle
from typing import Any, Optional

from python_services.caches.base_cache import BaseCacheBackend

logger = logging.getLogger(__name__)


"""
5. pickle.load, dumps(value) 加码、解码？

6. return self.client.delete(self._key(key)) > 0 为啥大于0？

"""


class RedisCacheBackend(BaseCacheBackend):
    """redis缓存"""

    def __init__(self, url: str, ttl: int = 3600, prefix: str = "rag:"):
        """初始化redis缓存"""
        self.default_ttl = ttl
        self.prefix = prefix
        try:
            import redis
            self.client = redis.from_url(url)
            self.client.ping()
            logger.info("✅️Redis cache connected(连接成功)")
        except Exception as e:
            logger.error(f"❌️Redis connection failed(连接失败): {e}")

    def _key(self, key: str) -> str:
        """生成key"""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """查"""
        try:
            data = self.client.get(self._key(key))
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌️Redis get failed(获取失败): {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置"""
        ttl = ttl or self.default_ttl
        try:
            data = pickle.dumps(value)
            self.client.setex(self._key(key), ttl, data)
            logger.info(f"✅️Redis set success(设置成功): {self._key(key)}")
            return True
        except Exception as e:
            logger.error(f"❌️Redis set failed(设置失败): {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除"""
        try:
            return self.client.delete(self._key(key)) > 0
        except Exception as e:
            logger.error(f"❌️Redis delete failed(删除失败): {e}")
            return False

    def clear(self) -> bool:
        try:
            pattern = f"{self.prefix}*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            logger.info(f"✅️Redis clear success(清空成功): {pattern}")
            return True
        except Exception as e:
            logger.error(f"❌️Redis clear failed(清空失败): {e}")
            return False

    def stats(self) -> dict[str, Any]:
        """统计信息"""
        try:
            stats = self.client.info("stats")
            return {
                "hits": stats.get("keyspace_hits", 0),
                "misses": stats.get("keyspace_misses", 0),
                "keys": len(self.client.keys(f"{self.prefix}*"))
            }
        except Exception as e:
            return {}

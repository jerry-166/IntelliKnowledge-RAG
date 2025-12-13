"""
Redis 缓存
"""
import logging
import pickle
from typing import Any, Optional

from python_services.caches.base_cache import BaseCacheBackend

logger = logging.getLogger(__name__)


"""
1. pickle.load, dumps(value) 加码、解码？
    1.1 pickle.dumps(value): 将 Python 对象序列化为二进制数据（编码过程）
    将内存中的 Python 对象转换为可以存储或传输的字节流
    用于将缓存值转换为 Redis 可以存储的格式
    1.2 pickle.loads(data): 将二进制数据反序列化为 Python 对象（解码过程）
    将从 Redis 获取的字节流还原为原始的 Python 对象
    用于从缓存中读取数据时恢复原始值
    Redis 只能存储字符串或二进制数据，因此需要使用 pickle 来处理复杂的 Python 对象。

    1.3 pickle
    pickle 是 Python 标准库中的一个模块，用于对象序列化和反序列化。它可以将几乎任何 Python 对象转换为字节流，并能够将字节流还原为原始对象。
    1.3.1 序列化（Serialization）
    pickle.dumps(obj): 将对象转换为字节流
    pickle.dump(obj, file): 将对象序列化并写入文件
    1.3.2 反序列化（Deserialization）
    pickle.loads(data): 从字节流还原对象
    pickle.load(file): 从文件读取并还原对象
    
    Pickle 格式是 Python 特有的

2. return self.client.delete(self._key(key)) > 0 为啥大于0？
    Redis.delete() 方法返回被成功删除的键的数量（整数）
    因此通过判断返回值是否大于0来确定删除操作是否真正删除了数据
    这样可以区分"删除成功"（键存在并被删除）和"键不存在"的情况
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
            logger.error(f"❌️Redis stats failed(统计失败): {e}")
            return {}

"""
内存缓存后端
"""
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional
from python_services.caches.base_cache import BaseCacheBackend

"""
1. self._cache.pop(key, None)
    移除时，如果成功则移除，如果失败，有None则返回None，没有则raise（这里是防止raise）

2. next(iter(self._cache))

3. 阻塞式的请求吗？（没有return false）难道是默认false？/ 只许成功？

"""


class MemoryCacheBackend(BaseCacheBackend):
    """内存缓存后端(LRU)"""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._expiry: dict[str, float] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        # 拿锁
        with self._lock:
            # 检查缓存中是否存在该key
            if key not in self._cache:
                self._misses += 1
                return None

            # 存在key，继续检查是否过期
            if self._expiry and key in self._expiry and time.time() > self._expiry[key]:
                # 过期了，删除对应的键值对
                self._remove(key)
                self._misses += 1
                return None

            # 缓存中存在该key，且没有过期，返回对应的值
            self._hits += 1
            # LRU：移动到末尾
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # 拿锁
        with self._lock:
            # 如果已有，则删除旧数据，进行覆盖
            if key in self._cache:
                self._remove(key)
            # 检查容量
            while len(self._cache) >= self.max_size:
                oldest = next(iter(self._cache))
                self._remove(oldest)

            # 容量够了，开始写入新数据
            self._cache[key] = value

            # 设置过期时间
            ttl = ttl or self.ttl
            if ttl > 0:
                self._expiry[key] = time.time() + ttl

            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    def clear(self) -> bool:
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            return True

    def stats(self) -> dict[str, Any]:
        """获取缓存状态"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hits_rate": self._hits / max(1, self._hits + self._misses)
        }

    def _remove(self, key):
        """删除缓存中的数据"""
        self._cache.pop(key, None)
        self._expiry.pop(key, None)
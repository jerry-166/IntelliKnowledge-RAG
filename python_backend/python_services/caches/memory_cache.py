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

2. next(iter(self._cache)) 是什么用？next(),iter()?
    self._cache 是一个 OrderedDict（有序字典）
    iter(self._cache) 创建一个迭代器，指向字典的第一个元素
    next() 函数获取迭代器的下一个元素（即第一个元素）
常用场景：
    避免创建不必要的中间列表
        相比 list(collection)[0]，next(iter()) 更高效：
        不需要创建完整的列表副本
        只获取需要的第一个元素就停止迭代
    这种模式在性能敏感的场景中特别有用，比如缓存系统的淘汰机制、队列操作等
    
3. with实现了阻塞式的请求吗？（没有return false）难道是默认false？/ 只许成功？
    设计原则是尽力而为（best-effort）
    对于 set 方法，除非发生严重异常，否则总是返回 True，表示操作已完成
    对于 delete 方法，无论键是否存在都会正确处理，存在则删除并返回 True，不存在则返回 False
    设计上倾向于简化调用者逻辑，不需要处理复杂的返回状态
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

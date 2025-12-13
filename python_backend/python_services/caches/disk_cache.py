"""
磁盘缓存
"""
import json
import logging
import pickle
import time
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from python_services.caches.base_cache import BaseCacheBackend

logger = logging.getLogger(__name__)

"""
1.
oldest = min(
    self._metadata["items"].items(),
    key=lambda x: x[1]["created"]
)[0]
    _metadata = {"items": {"key0": {"created": t, "expiry": t, ...}, "key1": {}, ...}, "hits": 10, "missed": 0}
    items()获得键值对
    x 表示每一个键值对 - x[1] 表示键值对中的值
    最后返回的是键值对 - min()[0] 表示键值对中的键

2.# 什么时候保存什么元数据 self._save_metadata()
    添加新缓存项时：更新 items 记录
    删除缓存项时：从 items 中移除记录
    清空缓存时：重置整个元数据结构 目的是确保内存中的元数据变化能持久化到 _metadata.json 文件中
todo: 优化项：
    添加一个数组，数组满则去写文件，可以减少每一次都要io的时间消耗 ✅️

3. self.cache_dir.glob("*.pkl")，glob操作？
    使用通配符模式匹配所有以 .pkl 结尾的文件
    返回所有缓存数据文件的迭代器
todo: 文件还有什么常用的操作?
     Path.stat()：获取文件状态信息
     Path.read_text()/Path.write_text()：读写文本文件
     ...
     
4. 磁盘缓存不需要用锁吗？
    当前实现没有使用锁，存在线程安全问题：
    多线程同时访问可能造成元数据不一致
    并发写入可能导致文件损坏 建议添加 threading.Lock() 来保护并发访问
"""


class DiskCacheBackend(BaseCacheBackend):
    """磁盘缓存"""

    def __init__(self, cache_dir: str, ttl: int = 3600, max_size: int = 1000):
        """初始化磁盘缓存"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = ttl
        self.max_size = max_size
        self._lock = Lock()
        self._metadata = None
        self._metadata_file = self.cache_dir / "_metadata.json"
        # 加载元数据（键信息，命中信息...）
        self._load_metadata()

    def _load_metadata(self):
        """加载元数据"""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {"items": {}, "hits": 0, "misses": 0}

    def _save_metadata(self):
        """保存元数据"""
        if self._metadata:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f)

    def _file_path(self, key: str) -> Path:
        """获取文件路径"""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            # metadata中查询
            if key not in self._metadata["items"]:
                self._metadata["misses"] += 1
                return None

            # 检查过期
            item = self._metadata["items"][key]
            if time.time() > item["expiry"]:
                # 过期，删除键信息
                self.delete(key)
                self._metadata["misses"] += 1
                return None

            # 没有过期，检查文件是否存在
            file = self._file_path(key)
            if not file.exists():
                self.delete(key)
                self._metadata["misses"] += 1
                return None

            # 读取文件
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    self._metadata["hits"] += 1
                    return data
            except Exception as e:
                self.delete(key)
                logger.error(f"❌️Disk cache get failed: {e}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        # # 检查metadata中是否已存在(重复的逻辑，下方也会做)
        # if key in self._metadata["items"]:
        #     # 已存在，则覆盖原数据
        #     self._metadata["items"][key] = {
        #         "created": time.time(),
        #         "expiry": time.time() + ttl or self.default_ttl
        #     }

        # 不存在，检查容量
        with self._lock:
            while len(self._metadata["items"]) >= self.max_size:
                oldest = min(
                    self._metadata["items"].items(),
                    key=lambda x: x[1]["created"]
                )[0]
                self.delete(oldest)

            # 容量足够，写入文件
            self._metadata["items"][key] = {
                "created": time.time(),
                "expiry": time.time() + ttl or self.default_ttl
            }
            try:
                with open(self._file_path(key), 'wb') as f:
                    pickle.dump(value, f)
                    self._save_metadata()
                    return True
            except Exception as e:
                logger.error(f"❌️Disk cache set failed: {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._metadata["items"]:
                del self._metadata["items"][key]
                self._save_metadata()

            file = self._file_path(key)
            if file.exists():
                # unlink---删除文件/链接，如果是目录，使用rmdir()
                file.unlink()
                return True
            return False

    def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            self._metadata = {"items": {}, "hits": 0, "misses": 0}
            self._save_metadata()
            return True

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._metadata["items"]),
            "max_size": self.max_size,
            "hits": self._metadata["hits"],
            "misses": self._metadata["misses"],
        }

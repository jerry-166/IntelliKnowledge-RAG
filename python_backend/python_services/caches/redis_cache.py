"""
Redis 缓存
"""
import logging
import os.path
import pickle
import time
from typing import Any, Optional

from langchain_core.embeddings import embeddings

from python_services.caches.base_cache import BaseCacheBackend
from python_services.utils.index_util import IndexUtil

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
    """Redis缓存（支持语义查询）"""

    def __init__(
            self,
            url: str,
            ttl: int = 3600,
            prefix: str = "rag:",
            embedding: Optional[embeddings] = None,
            semantic_threshold: float = 0.95,  # 语义匹配阈值
            faiss_path: str = "faiss_index.pkl",
    ):
        """初始化redis缓存"""
        self.default_ttl = ttl
        self.prefix = prefix
        self.embedding = embedding
        self.semantic_threshold = semantic_threshold
        self.faiss_path = faiss_path

        try:
            import redis
            self.client = redis.from_url(url)
            self.client.ping()
            logger.info("✅ Redis连接成功")
        except Exception as e:
            logger.error(f"❌ Redis连接失败: {e}")
            raise

        # 初始化FAISS向量存储
        self._init_faiss()

    def _init_faiss(self):
        """初始化FAISS向量存储"""
        try:
            from langchain_community.vectorstores import FAISS, DistanceStrategy

            # 先尝试从本地文件加载FAISS向量存储
            if self.faiss_path and self.embedding:
                if os.path.exists(self.faiss_path):
                    try:
                        self.store = FAISS.load_local(
                            self.faiss_path,
                            self.embedding,
                            allow_dangerous_deserialization=True,
                        )
                        logger.info("✅ FAISS向量local存储加载成功")
                        return
                    except Exception as e:
                        logger.error(f"❌ FAISS向量local存储加载失败: {e}")

            if self.embedding:
                from langchain_community.docstore.in_memory import InMemoryDocstore

                index = IndexUtil.get_faiss_index(self.embedding, "flat", 100)
                self.store = FAISS(
                    embedding_function=self.embedding,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                    distance_strategy=DistanceStrategy.COSINE,
                )
                logger.info("✅ FAISS向量存储初始化成功")
        except Exception as e:
            logger.error(f"❌ FAISS初始化失败: {e}")
            self.store = None

    def _key(self, key: str) -> str:
        """生成redis key"""
        return f"{self.prefix}{key}"

    def _id_key(self, key: str) -> str:
        """生成ID映射的key"""
        return f"{self.prefix}id:{key}"

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值（支持精确匹配和语义匹配）"""
        try:
            # 1. 精确匹配
            data = self.client.get(self._key(key))
            if data:
                rebuild_value = pickle.loads(data)
                # 检查过期
                if rebuild_value["expiry"] >= time.time():
                    logger.info(f"精确匹配命中: {key}")
                    return rebuild_value["documents"]
                else:
                    self.delete(key)  # 过期删除
                    return None

            # 2. 语义匹配
            if self.embedding and self.store:
                return self._semantic_get(key)

            return None
        except Exception as e:
            logger.error(f"❌️获取redis缓存失败: {e}")
            raise e

    def _semantic_get(self, query: str) -> Optional[Any]:
        """语义查询"""
        try:
            # 搜索最相似的问题（根据query去与each_query_embed匹配，返回similar_query）
            results = self.store.similarity_search_with_score(query, k=3)
            if not results:
                return None

            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)

            for doc, score in results:
                if score >= self.semantic_threshold:
                    # 通过文档内容获取原始key
                    similar_query = doc.page_content
                    # 获取缓存值
                    data = self.client.get(self._key(similar_query))
                    if data:
                        rebuild_value = pickle.loads(data)

                        # 检查过期
                        if rebuild_value["expiry"] >= time.time():
                            logger.info(f"语义匹配命中: {query} -> {similar_query} (相似度: {score:.3f})")
                            return rebuild_value["documents"]
                        else:
                            self.delete(similar_query)  # 过期删除
            return None

        except Exception as e:
            logger.warning(f"语义查询失败: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        ttl = ttl or self.default_ttl
        rebuild_value = {
            "documents": value,
            "expiry": time.time() + ttl,
        }

        try:
            # 1. 存储到Redis
            data = pickle.dumps(rebuild_value)
            redis_key = self._key(key)
            self.client.set(redis_key, data)

            # 2. 添加到FAISS索引
            if self.embedding and self.store:
                try:
                    self.store.add_texts([key], ids=[key])
                    # 保存FAISS到磁盘
                    if self.faiss_path:
                        self.store.save_local(self.faiss_path)
                        logger.info(f"✅ FAISS索引已保存到{self.faiss_path}")
                except Exception as faiss_e:
                    logger.warning(f"添加到FAISS索引失败: {faiss_e}")

            logger.info(f"设置缓存成功: {key}")
            return True

        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            if key:
                # 1. 从Redis删除值
                redis_deleted = self.client.delete(self._key(key))

                # 2. 从FAISS删除
                faiss_deleted = True
                try:
                    self.store.delete([key])
                    logger.info(f"从FAISS删除成功: {key}")
                except Exception as e:
                    logger.warning(f"从FAISS删除失败: {e}")
                    faiss_deleted = False

                return redis_deleted > 0 and faiss_deleted
            return True
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False

    def clear(self) -> bool:
        """清空缓存"""
        try:
            # 1. 清空Redis（使用scan避免阻塞）
            pattern = f"{self.prefix}*"
            cursor = 0
            total_deleted = 0

            while True:
                cursor, keys = self.client.scan(cursor, pattern, count=100)
                if keys:
                    deleted = self.client.delete(*keys)
                    total_deleted += deleted
                if cursor == 0:
                    break

            # 2. 重新初始化FAISS
            self._init_faiss()

            logger.info(f"清空缓存成功，删除了 {total_deleted} 个键")
            return True

        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def stats(self) -> dict[str, Any]:
        """获取缓存统计信息"""
        try:
            # Redis统计
            redis_stats = self.client.info("stats")
            memory_stats = self.client.info("memory")

            # 获取键数量（使用scan计数）
            pattern = f"{self.prefix}*"
            key_count = 0
            cursor = 0

            while True:
                cursor, keys = self.client.scan(cursor, pattern, count=100)
                key_count += len(keys)
                if cursor == 0:
                    break

            # FAISS统计
            faiss_count = 0
            if self.store and hasattr(self.store.index, 'ntotal'):
                faiss_count = self.store.index.ntotal

            return {
                "redis_keys": key_count,
                "faiss_vectors": faiss_count,
                "hits": redis_stats.get("keyspace_hits", 0),
                "misses": redis_stats.get("keyspace_misses", 0),
                "hit_rate": (
                        redis_stats.get("keyspace_hits", 0) /
                        max(1, redis_stats.get("keyspace_hits", 0) + redis_stats.get("keyspace_misses", 0))
                ),
                "memory_used_mb": memory_stats.get("used_memory", 0) / 1024 / 1024,
                "connected_clients": redis_stats.get("connected_clients", 0),
                "expired_keys": redis_stats.get("expired_keys", 0),
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

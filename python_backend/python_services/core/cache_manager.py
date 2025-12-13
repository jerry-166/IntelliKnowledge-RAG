"""
缓存管理器 - 支持多种后端
"""
import logging
from typing import Any, Optional, Dict
import python_services.core.logger_config  # 导入日志配置
from python_services.caches.disk_cache import DiskCacheBackend
from python_services.caches.memory_cache import MemoryCacheBackend
from python_services.caches.redis_cache import RedisCacheBackend
from python_services.core.settings import CacheConfig

logger = logging.getLogger(__name__)

"""
1. 在哪里logger比较好呢？是每一个python文件都要吗？logger打印的信息怎么看？可以写入文件吗？
1.1 关键业务逻辑点：在重要的函数入口和出口记录日志
    异常处理位置：在 try-except 块中记录错误信息
    系统初始化：记录组件初始化成功或失败的状态
    重要操作结果：记录关键操作的执行结果（成功/失败
1.2 建议每个模块都有日志：使用 logging.getLogger(__name__) 创建模块级日志记录器
    按需记录：不是每行代码都要记录，而是记录有价值的信息
    不同级别：合理使用 debug、info、warning、error 等不同日志级别
1.3 查看方式：
    控制台输出（开发调试时）
    日志文件查看（生产环境）
    '''
    import logging
    # 配置日志写入文件
    logging.basicConfig(
        filename='app.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 或者使用FileHandler
    handler = logging.FileHandler('app.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    '''
"""


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
            )
        elif self.config.backend == "redis":
            if not self.config.redis_url:
                raise ValueError("Redis URL is required")
            self.backend = RedisCacheBackend(
                url=self.config.redis_url,
                ttl=self.config.ttl,
                prefix=self.config.redis_prefix,
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

"""
统一配置管理，支持多环境，热加载
todo: reranker,cache...
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml

"""
1. field 的作用包括：
设置默认值工厂: default_factory=ParseConfig 表示当创建 ParseConfig 类的新实例时，
    如果没有显式提供 parser 参数，则会调用 ParseConfig() 来生成一个默认的 ParseConfig 实例作为其初始值。
避免共享可变对象: 如果直接写成 parser: ParseConfig = ParseConfig()，那么所有的类
    实例都会共享同一个 ParseConfig 对象，可能导致意外的行为。使用 default_factory 可以确保每个实例都获得一个新的 ParseConfig 实例。
类型提示与实际赋值分离: 允许你为字段添加类型注解的同时，又能够灵活地控制其实例化过程。

总之，这里的 field 主要是为了给 parser 字段提供一个动态生成的默认值，而不是静态的默认值

2. @lru_cache() 装饰器的作用是实现函数结果缓存，具体说明如下：
主要作用
缓存函数返回值: 当 get_config() 函数被调用时，装饰器会缓存其返回的 RAGConfig 对象
避免重复创建: 相同参数的多次调用将直接返回缓存的结果，而不会重新执行函数体
在当前代码中的效果
单例模式实现:
第一次调用 get_config() 会创建一个新的 RAGConfig 实例并缓存
后续调用（无论是否传入相同参数）都会返回同一个缓存的实例
性能优化:
避免重复的配置文件读取和对象创建开销
特别是在频繁调用 get_config() 时提升明显

3. score_threshold: float = 0.3

4. @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
    
5. # todo:filter_dict: dict = field(default_factory=dict)
"""


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    use_clip: bool = True
    text_embedding_model: str = "BAAI/bge-base-zh-v1.5"
    clip_embedding_model: str = "openai/clip-vit-base-patch32"
    device: str = 'cpu'
    trust_remote_code: bool = True
    normalize: bool = True
    batch_size: int = 64


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    provider: Literal["faiss", "milvus", "qdrant", "chroma"] = "chroma"
    collection_name: str = "multimodal_docs"
    persist_directory: str = "./vector_db"
    distance_metric: Literal["cosine", "ip", "l2"] = "cosine"
    # 检索配置
    default_top_k: int = 10
    score_threshold: float = 0.5
    # todo:filter_dict: dict = field(default_factory=dict)
    # 连接配置
    host: Optional[str] = None
    port: Optional[str] = None
    # 性能配置
    batch_size: int = 64
    max_retries: int = 3
    timeout: int = 60
    save_queries_num: int = 1000


@dataclass
class RerankerConfig:
    """重排器配置"""
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    top_k: int = 5
    batch_size: int = 64
    score_threshold: float = 0.3  # todo:?
    max_length: int = 512


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    backend: Literal["disk", "redis", "memory"] = "memory"
    ttl: int = 3600
    # 内存缓存
    max_size: int = 1000
    # redis缓存
    redis_url: str = ""
    redis_prefix: str = "rag:"
    # 磁盘缓存
    cache_dir: str = "./cache"


@dataclass
class ParseConfig:
    """解析配置"""
    use_ocr: bool = True
    use_vision: bool = True
    ocr_engine: Literal["tesseract", "paddleocr", "easyocr"] = "tesseract"
    min_image_size: int = 100
    max_image_size: int = 4096
    supported_formats: list = field(default_factory=lambda: [
        "pdf", "docx", "pptx", "md", "txt", "html", "png", "jpg", "jpeg"
    ])
    # 并发配置
    max_workers: int = 10


@dataclass
class SplitterConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    strategy: Literal["recursive", "auto", "semantic", "markdown"] = "auto"
    # 语义切分配置
    min_chunk_size: int = 50
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = "percentile"
    breakpoint_threshold_amount: int = 95
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", "。", "！", "？", "；", " "])


@dataclass
class RAGConfig:
    """rag总的配置"""
    parser: ParseConfig = field(default_factory=ParseConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
        """从yaml文件中读取配置"""
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict):
        """从字典中加载配置，返回RAGConfig"""
        return cls(
            parser=ParseConfig(**data.get("parser", {})),
            splitter=SplitterConfig(**data.get("splitter", {})),
            vector_store=VectorStoreConfig(**data.get("vector_store", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            reranker=RerankerConfig(**data.get("reranker", {})),
            cache=CacheConfig(**data.get("cache", {})),
        )


@lru_cache()
def get_config(config_path: Optional[str] = None) -> "RAGConfig":
    """获取配置单例"""
    if config_path and Path(config_path).exists():
        return RAGConfig.from_yaml(config_path)
    return RAGConfig()

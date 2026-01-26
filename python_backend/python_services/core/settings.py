"""
统一配置管理，支持多环境，热加载
"""
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

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

3. todo: score_threshold: float = 0.3  reranker的threshold是什么作用？
    分数阈值配置，用于过滤相关性分数低于该阈值的搜索结果，只有分数高于 0.3 的结果才会被保留
怎么用？
    
4. @classmethod
def from_yaml(cls, path: str) -> "RAGConfig":  这里的cls是这个类的意思吗？可以调用类中的方法？
    cls 确实是指向当前类的引用（在这个上下文中指 RAGConfig 类）
    可以用来调用类中的其他类方法或创建类实例
    实现工厂方法模式，提供不同的对象创建方式

"""

clip_path = {
    "clip_cn": r"C:\Users\ASUS\.cache\huggingface\hub\models--OFA-Sys--chinese-clip-vit-large-patch14\chinese-clip-vit-large-patch14",
    "clip_openai": r"C:\Users\ASUS\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
}


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    use_clip: bool = True
    text_embedding_model: str = r"C:\Users\ASUS\.cache\huggingface\hub\models--BAAI--bge-base-zh-v1.5\snapshots\f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    clip_embedding_model: str = clip_path.get("clip_cn")
    device: str = 'cpu'
    trust_remote_code: bool = True
    normalize: bool = True
    batch_size: int = 64


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    provider: Literal["faiss", "milvus", "qdrant", "chroma"] = "chroma"
    distance_metric: Literal["cosine", "ip", "l2"] = "cosine"
    store_type: Literal["text", "image", "hybrid"] = "hybrid"
    # chroma
    collection_name: str = "multimodal_docs"
    persist_directory: str = "./output/data/vector_db/chroma"
    # faiss的索引方式
    faiss_persist_directory: str = "./output/data/vector_db/faiss"
    faiss_index_type: Literal["flat", "hnsw", "ivf"] = "Flat"
    faiss_image_index_type: Literal["flat", "hnsw", "ivf"] = "Flat"
    faiss_nlist: int = 100
    # 检索配置
    default_top_k: int = 10
    score_threshold: float = 0.5
    # 连接配置
    host: Optional[str] = None
    port: Optional[str] = None
    # 性能配置
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 60
    save_queries_num: int = 1000


@dataclass
class KeywordSearchConfig:
    """关键词检索配置"""
    enabled: bool = True

    # 检索后端: bm25 | elasticsearch | meilisearch
    backend: Literal["bm25", "elasticsearch", "meilisearch"] = "bm25"

    # BM25参数
    bm25_k1: float = 1.5  # 词频饱和参数
    bm25_b: float = 0.75  # 文档长度归一化参数

    # 分词器: jieba | simple | whitespace
    tokenizer: Literal["jieba", "simple", "whitespace"] = "jieba"

    # 停用词
    use_stopwords: bool = True
    custom_stopwords: List[str] = field(default_factory=list)
    persist_path: str = "./output/data/keyword/bm25_index.pkl"

    # elasticsearch参数
    es_host: str = "localhost"
    es_port: int = 9201
    es_index_name: str = "rag_documents"

    # es用户密码信息
    es_username: Optional[str] = None
    es_password: Optional[str] = None


@dataclass
class RetrieverConfig:
    """检索器配置"""
    # 混合检索配置
    hybrid_enabled: bool = True
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    fusion_method: Literal["rrf", "weighted", "dbsf"] = "rrf"  # 效果不好，感觉有reranker，不需要融合
    rrf_k: int = 60


@dataclass
class RerankerConfig:
    """重排器配置"""
    enabled: bool = True
    model_name: str = r"C:\Users\ASUS\.cache\huggingface\hub\models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70"
    device: str = "cpu"
    top_k: int = 5
    batch_size: int = 64
    score_threshold: float = 0.3
    max_length: int = 512


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    backend: Literal["disk", "redis", "memory"] = "memory"
    ttl: int = 3600
    semantic_threshold = 0.85
    # 内存缓存
    max_size: int = 1000
    # redis缓存
    redis_url: str = r"redis://localhost:6377/0?socket_timeout=5&retry_on_timeout=true"
    redis_prefix: str = "rag:"
    faiss_persist_directory: str = "./data/cache/faiss"
    # embed
    embedding = r"C:\Users\ASUS\.cache\huggingface\hub\models--BAAI--bge-base-zh-v1.5\snapshots\f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    # 磁盘缓存
    cache_dir: str = "./data/cache/disk"


@dataclass
class ParseConfig:
    """解析配置"""
    use_ocr: bool = True
    use_vision: bool = True
    ocr_engine: Literal["tesseract", "paddleocr", "easyocr"] = "tesseract"
    extract_images: bool = True
    extract_tables: bool = True
    min_image_size: int = 10
    max_image_size: int = 4096
    supported_formats: list = field(default_factory=lambda: [
        "pdf", "docx", "pptx", "md", "txt", "png", "jpg", "jpeg"
    ])
    # 并发配置
    max_workers: int = 10


@dataclass
class SplitterConfig:
    # semantic embed配置
    semantic_embedding_model: str = r"C:\Users\ASUS\.cache\huggingface\hub\models--BAAI--bge-base-zh-v1.5\snapshots\f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    device: str = "cpu"
    trust_remote_code: bool = True
    normalize_embeddings: bool = True
    # 文本切分配置
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
    keyword_search: KeywordSearchConfig = field(default_factory=KeywordSearchConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
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
            keyword_search=KeywordSearchConfig(**data.get("keyword_search", {})),
            retriever=RetrieverConfig(**data.get("retriever", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            reranker=RerankerConfig(**data.get("reranker", {})),
            cache=CacheConfig(**data.get("cache", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parser": self._obj_to_dict(self.parser),
            "splitter": self._obj_to_dict(self.splitter),
            "vector_store": self._obj_to_dict(self.vector_store),
            "keyword_search": self._obj_to_dict(self.keyword_search),
            "retriever": self._obj_to_dict(self.retriever),
            "embedding": self._obj_to_dict(self.embedding),
            "reranker": self._obj_to_dict(self.reranker),
            "cache": self._obj_to_dict(self.cache),
        }

    def _obj_to_dict(self, obj):
        """递归转化对象为字典"""
        if hasattr(obj, '__dict__') or hasattr(obj, '_asdict'):
            # 如果是数据类实例
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field_name in obj.__dataclass_fields__:
                    value = getattr(obj, field_name)
                    result[field_name] = self._obj_to_dict(value)
                return result
            else:
                try:
                    return {k: v for k, v in obj.__dict__.items()}
                except:
                    return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._obj_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._obj_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    @classmethod
    def save_to_yaml(cls, config: "RAGConfig", file_path: str = "../config.yaml"):
        """将配置信息保存到本地yaml文件中"""
        import os
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        config_dict = config.to_dict()

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

        print(f"配置已保存到 {file_path}")

    @classmethod
    def save_to_sql(cls):
        """将配置信息保存到数据库里"""
        pass


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> "RAGConfig":
    """获取配置单例"""
    if config_path and Path(config_path).exists():
        return RAGConfig.from_yaml(config_path)
    return RAGConfig()


def refresh_config():
    """刷新配置缓存"""
    get_config.cache_clear()

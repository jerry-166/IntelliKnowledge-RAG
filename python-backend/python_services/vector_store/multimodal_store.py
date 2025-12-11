"""
vector_store.py
向量数据库构建与检索
todo:图片向量库---clip
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, DistanceStrategy  # 或FAISS, Milvus等
from langchain_core.documents import Document
from transformers import CLIPModel, CLIPProcessor

from basic_core.llm_factory import qwen_vision
from python_services.core.settings import get_config, RAGConfig
from python_services.parsers.markdown_parser import MultimodalMarkdownParser
from python_services.splitter.integration_splitter import IntegrationSplitter
from python_services.vector_store.base_store import BaseVectorStore, SearchResult

logger = logging.getLogger(__name__)  # 日志

"""
1. 装饰器：

2. mkdir(parents=True, exist_ok=True)

3. field(default_factory=threading.Lock, repr=False)
"""


# 定义重试的装饰器
def retry_on_failure(max_retries=3, delay: float = 1.0):
    """重试的装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"第 {attempt + 1}/{max_retries} 次尝试，failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exception

        return wrapper

    return decorator


# 定义计时的装饰器
def time_counter(func):
    """计时装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} 耗时: {elapsed:.3f}秒")
        return wrapper

    return time_counter


@dataclass
class VectorStoreMetrics:
    """向量库指标"""
    def __init__(self, save_queries_num: int = 1000):
        self.save_queries_num = save_queries_num

    total_documents: int = 0
    total_queries: int = 0
    avg_time_per_query_ms: float = 0.0
    cache_misses: int = 0
    cache_hits: int = 0
    last_update_time: Optional[str] = None

    _query_times: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_query(self, elapsed_time):
        """记录查询时间"""
        with self._lock:
            self._query_times.append(elapsed_time)
            self.total_queries += 1
            # 保留最近1000次查询的时间信息
            if len(self._query_times) > self.save_queries_num:
                self._query_times = self._query_times[-self.save_queries_num:]
            self.avg_time_per_query_ms = sum(self._query_times) / len(self._query_times)


class MultimodalVectorStore(BaseVectorStore):
    """多模态向量数据库"""
    def __init__(self, rag_config: Optional[RAGConfig] = None, vision_llm=None):
        self.rag_config = rag_config if rag_config else get_config()
        self.vision_llm = vision_llm
        self.metrics = VectorStoreMetrics(self.rag_config.vector_store.save_queries_num)

        # 初始化组件
        self._init_embeddings()
        self._init_vector_stores()
        self._init_reranker()
        self._init_cache()

        logger.info(
            f"MultimodalVectorStore初始化成功：{self.rag_config.vector_store.provider}"
        )

    def add_documents(self, documents: list[Document], **kwargs) -> list[str]:
        pass

    def update(self, id: str, document: Document) -> bool:
        pass

    def delete(self, ids: list[str]) -> bool:
        pass

    def search(
            self,
            query: str,
            top_k: int = 10,
            filter_dict: Optional[dict[str, Any]] = None,
            **kwargs
    ) -> list[SearchResult]:
        pass

    def persist(self, persist_directory: str):
        pass

    def load(self, persist_directory: str):
        pass

    def get_stats(self) -> dict[str, Any]:
        pass

    def _init_embeddings(self):
        """初始化嵌入模型"""
        embedding_config = self.rag_config.embedding

        # 文本嵌入
        self.text_embedding = HuggingFaceEmbeddings(
            model_name=embedding_config.text_embedding_model,
            model_kwargs={'device': embedding_config.device},
            encode_kwargs={'normalize_embeddings': embedding_config.normalize}
        )

        # 图片嵌入
        if embedding_config.use_clip and embedding_config.clip_embedding_model:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self.clip_model = CLIPModel.from_pretrained(embedding_config.clip_embedding_model)
                self.clip_processor = CLIPProcessor.from_pretrained(embedding_config.clip_embedding_model)
                self._has_clip = True
            except Exception as e:
                logger.error(f"加载CLIP模型失败: {e}")
                self._has_clip = False
        else:
            self._has_clip = False

    def _init_vector_stores(self):
        """初始化向量数据库"""
        store_config = self.rag_config.vector_store
        persist_directory = Path(store_config.persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        if store_config.provider == "faiss":
            self._init_faiss_store(store_config)
        elif store_config.provider == "chroma":
            self._init_chroma_store(store_config, persist_directory)
        elif store_config.provider == "milvus":
            self._init_milvus_store(store_config)
        elif store_config.provider == "qdrant":
            self._init_qdrant_store(store_config)
        else:
            raise ValueError(f"不支持的向量数据库提供者: {store_config.provider}")

    # todo: faiss的向量库初始化
    def _init_faiss_store(self, store_config):
        """初始化faiss向量数据库"""
        try:
            from langchain_community.vectorstores import FAISS
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore

            embedding_dim = len(self.text_embedding.embed_query("hello world"))
            index = faiss.IndexFlatL2(embedding_dim)

            self.text_store = FAISS(
                embedding_function=self.text_embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=store_config.distance_metric,
            )

            if self._has_clip:
                self.image_store = FAISS(
                    embedding_function=self.text_embedding,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                    distance_strategy=store_config.distance_metric,
                )
            else:
                self.image_store = None
        except ImportError:
            raise ImportError("请安装faiss-cpu来使用Faiss向量数据库")

    def _init_chroma_store(self, store_config, persist_directory: Path):
        """初始化chroma向量数据库"""
        try:
            from langchain_chroma.vectorstores import Chroma
            self.text_store = Chroma(
                collection_name=store_config.collection_name+"_text",
                embedding_function=self.text_embedding,
                persist_directory=str(persist_directory / "text"),
            )
            if self._has_clip:
                self.image_store = Chroma(
                    collection_name=store_config.collection_name+"_image",
                    embedding_function=self.text_embedding,
                    persist_directory=str(persist_directory / "image"),
                )
            self.image_store = None
        except ImportError:
            raise ImportError("请安装langchain_chroma or langchain_community（对Chroma已弃用）来使用Chroma向量数据库")

    def _init_milvus_store(self, store_config):
        """初始化milvus向量数据库"""
        try:
            from langchain_community.vectorstores import Milvus

            connection_args = {
                "host": store_config.host if store_config.host else "localhost",
                "port": store_config.port if store_config.port else 19530,
            }

            self.text_store = Milvus(
                embedding_function=self.text_embedding,
                collection_name=store_config.collection_name+"_text",
                connection_args=connection_args,
            )

            if self._has_clip:
                self.image_store = Milvus(
                    embedding_function=self.text_embedding,
                    collection_name=store_config.collection_name+"_image",
                    connection_args=connection_args,
                )
            self.image_store = None
        except ImportError:
            raise ImportError("请安装langchain_milvus来使用Milvus向量数据库")

    def _init_qdrant_store(self, store_config):
        """初始化qdrant向量数据库"""
        try:
            from langchain_qdrant.vectorstores import Qdrant
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(
                host=store_config.host if store_config.host else "localhost",
                port=store_config.port if store_config.port else 6333,
            )
            self.text_store = Qdrant(
                client=qdrant_client,
                collection_name=store_config.collection_name+"_text",
                embeddings=self.text_embedding,
                distance_strategy=store_config.distance_metric,
            )
            if self._has_clip:
                self.image_store = Qdrant(
                    client=qdrant_client,
                    collection_name=store_config.collection_name+"_image",
                    embeddings=self.text_embedding,
                    distance_strategy=store_config.distance_metric,
                )
            self.image_store = None
        except ImportError:
            raise ImportError("请安装langchain_qdrant, qdrant_client来使用Qdrant向量数据库")

    def _init_reranker(self):
        pass

    def _init_cache(self):
        pass


if __name__ == '__main__':
    file_path = r"C:\Users\ASUS\Desktop\makedown\deepAgent.md"
    markdown_parser = MultimodalMarkdownParser(qwen_vision, use_vision=True)
    documents = markdown_parser.parse(file_path)
    document_split = IntegrationSplitter(
        embeddings=HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        ),
    ).split_documents_(documents, file_type='md')

    multimodal_vector_store = MultimodalVectorStore(strategy='text_only', persist_directory='./vector_db')
    multimodal_vector_store.add_text_documents(document_split)
    multimodal_vector_store.persist()
    outputs = multimodal_vector_store.search(query='deep-agent主要内容有什么？', top_k=10)
    for output in outputs:
        print(output)

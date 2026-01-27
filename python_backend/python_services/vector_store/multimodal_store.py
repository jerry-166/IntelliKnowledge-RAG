"""
企业级多模态向量存储
- 支持多种后端(Chroma/Milvus/Qdrant)
- 批量处理优化
- 重试机制
- 缓存层
- 监控指标
"""
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Literal, List, Dict
from langchain_core.documents import Document
from basic_core.llm_factory import qwen_vision, init_embedding_basic
from python_services.core.cache_manager import CacheManager
from python_services.core.settings import get_config, RAGConfig
from python_services.embeddings.clip_embedding import CLIPEmbeddings
from python_services.parsers.pdf_parser import PDFParser
from python_services.reranker.reranker_ import Reranker
from python_services.splitter.integration_splitter import IntegrationSplitter
from python_services.utils.index_util import IndexUtil
from python_services.vector_store.base_store import BaseVectorStore, SearchResult

logger = logging.getLogger(__name__)  # 日志

"""
1. 装饰器：
    1.1 带参数的装饰器
    1.2 不带参数的装饰器
2. mkdir(parents=True, exist_ok=True)
    这行代码调用了 Python 标准库中的 pathlib.Path.mkdir() 方法，参数含义如下：
    parents=True: 如果目标路径的父目录不存在，则递归创建所有缺失的父目录。
    exist_ok=True: 如果目标目录已经存在，不会引发异常
    目的是确保指定的目录结构存在，避免因目录不存在而导致程序崩溃。
3. field(default_factory=threading.Lock, repr=False)
    这是使用了 Python 数据类 (dataclass) 中的 field 来自定义字段行为：
    default_factory=threading.Lock: 表示这个字段默认值工厂函数返回的结果 —— 创建一个线程锁对象。这样每个实例都会拥有自己独立的锁对象。
    repr=False: 表示在打印该对象时，不显示这个字段的内容。
    这种写法常见于需要并发安全访问共享资源的数据类中，比如记录指标时防止多个线程同时修改 _query_times 列表导致竞争条件。
6. faiss向量数据库
    flat 暴力遍历
    lvf 聚类
    hnsw 图搜索
7. hashlib.md5(content.encode()).hexdigest()
    MD5主要用于：
    文档ID生成 (_generate_doc_id)
    缓存键生成 (_build_cache_key)
    对于这些用途，MD5是合适的：
    不涉及安全敏感操作（密码存储等）
    仅需要内容一致性保证和去重功能
    性能较好，计算速度快
    # 可替换：sha1、sha256、blake2b（安全性低到高）不替换

8. self.metrics.cache_misses += 1 和 cache中的misses ？
    确实有重复，但是cache的可以做内部调试，暂时不移除 

10. _build_cache_key缓存机制能用到吗？那个key感觉不是很好命中啊...
    存储的key是按语义的，而不是有一个语义缓存 ===》 key构建问题
    
12. 哪个重试？计时？
    需要重试的操作（涉及 I/O、网络、外部服务）
    成本耗时关注
    
13. sorted(filter_dict.items())作用是哈？ 
    保证相同过滤条件的不同字典顺序会产生相同的缓存键。
    
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
                    return func(*args, **kwargs)
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
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} 耗时: {elapsed:.3f}秒")
        return result

    return wrapper


@dataclass
class VectorStoreMetrics:
    """向量库指标"""
    # 文档指标 解析文档时更新
    total_documents: int = 0
    # 查询指标 record_query
    total_queries: int = 0
    avg_time_per_query_ms: float = 0.0
    # 缓存指标 自己更新
    cache_misses: int = 0
    cache_hits: int = 0
    # 每次更新的时间更新
    last_update_time: Optional[str] = None

    _query_times: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    save_queries_num: int = 1000

    def record_query(self, elapsed_time):
        """记录查询时间"""
        with self._lock:
            self._query_times.append(elapsed_time)
            self.total_queries += 1
            # 默认保留最近1000次查询的时间信息
            if len(self._query_times) > self.save_queries_num:
                self._query_times = self._query_times[-self.save_queries_num:]
            self.avg_time_per_query_ms = sum(self._query_times) / len(self._query_times)


class MultimodalVectorStore(BaseVectorStore):
    """
    企业级多模态向量存储

    Features:
    - 多后端支持 (Chroma/Milvus/Qdrant)
    - 文本+图片双向量空间
    - 批量处理与并发优化
    - 自动重试与错误恢复
    - 查询缓存
    - 重排序支持
    - 监控指标收集

    Example:
         config = get_config()
         store = MultimodalVectorStore(config)
         store.add_documents(documents)
         results = store.search("query", top_k=10, use_reranker=True)
    """

    def __init__(self, rag_config: Optional[RAGConfig] = None):
        self.store_type = rag_config.vector_store.store_type
        self.rag_config = rag_config if rag_config else get_config()
        self.metrics = VectorStoreMetrics(self.rag_config.vector_store.save_queries_num)

        # 初始化组件
        self._init_embeddings()
        self._init_vector_stores()
        self._init_reranker()
        self._init_cache()

        logger.info(
            f"✅️MultimodalVectorStore初始化成功：{self.rag_config.vector_store.provider}"
        )

    @time_counter  # 检索延迟是核心指标
    @retry_on_failure()  # 查询可能超时，失败
    def search(
            self,
            query: str,
            top_k: int = 10,
            filter_dict: Optional[dict[str, Any]] = None,
            search_type: Literal["text", "image", "hybrid"] = "text",
            use_reranker: bool = True,
            use_cache: bool = True,
    ) -> list[SearchResult]:
        start_time = time.perf_counter()
        top_k = top_k or self.rag_config.vector_store.default_top_k

        # 查缓存
        if use_cache and self.cache:
            cached = self.cache.get(query)
            if cached:
                logger.info(f"✅️缓存命中，问题：{query[:50]}")
                self.metrics.cache_hits += 1
                # 对缓存结果进行过滤，需要吗---需要top_k过滤
                return cached[:top_k]
            self.metrics.cache_misses += 1

        # 执行检索
        results = self._execute_search_store(query, top_k, filter_dict, search_type)

        # 重排序
        if use_reranker and self.reranker and results:
            results = self.reranker.rerank(query, results, top_k)

        # 写入缓存
        if use_cache and self.cache:
            self.cache.set(query, results)

        # 记录指标
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.record_query(elapsed_ms)
        logger.info(f"✅️MultimodalVectorStore查询完成，结果：{len(results)}，问题：{query[:50]}")

        return results

    @time_counter  # 索引构建时间
    @retry_on_failure()  # 向量库写入可能超时（网络，io等）
    def add_documents(
            self,
            documents: list[Document],
            batch_size: Optional[int] = None,
            show_progress: bool = True,
    ) -> list[str]:
        """
        添加document到向量数据库中
        :param documents: 文档
        :param batch_size: 批次大小
        :param show_progress: 是否显示进度
        :return: 返回ids列表
        """
        # 清理复杂元数据
        cleaned_documents = self._clean_metadata(documents)

        # 基于内容去重
        unique_docs = {}
        for doc in cleaned_documents:
            doc_id = self._generate_doc_id(doc)
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc

        deduplicated_documents = list(unique_docs.values())

        if not deduplicated_documents:
            return []

        ids = []
        batch_size = batch_size or self.rag_config.vector_store.batch_size

        # 分类
        text_docs = []
        image_docs = []
        if self.store_type == "image" and self._has_clip:  # 如果使用clip，默认放在clip嵌入的向量空间中
            image_docs = documents
        elif self.store_type == "text":  # 不使用，默认图片和文本在一个向量空间
            text_docs = documents
        else:  # 混合模式则是分开放
            for doc in deduplicated_documents:
                if doc.metadata.get('type', '') == "image":  # 跨模态
                    image_docs.append(doc)
                elif doc.metadata.get('type', '') == "text":  # 深度
                    text_docs.append(doc)

        # 批量加入store中
        if text_docs:
            text_ids = self._batch_store(
                self.text_store,
                text_docs,
                batch_size,
                "text",
                show_progress
            )
            ids.extend(text_ids)
        if image_docs:
            image_ids = self._batch_store(
                self.image_store,
                image_docs,
                batch_size,
                "image",
                show_progress
            )
            ids.extend(image_ids)

        # 更新监控指标
        self.metrics.total_documents += len(deduplicated_documents)
        self.metrics.last_update_time = time.strftime("%Y-%m-%D %H:%M:%S")

        logger.info(
            f"✅️添加{len(text_docs)}个文本文档和{len(image_docs)}个图片文档成功"
        )

        return ids

    def _execute_search_store(
            self,
            query: str,
            top_k: int,
            filter_dict: Optional[dict[str, Any]] = None,
            search_type: Literal["text", "image", "hybrid"] = "text",
    ) -> list[SearchResult]:
        """从store中查询"""
        all_results = []
        if search_type in ["text", "hybrid"] and self.text_store:
            text_results = self._search_store(self.text_store, query, top_k, filter_dict)
            all_results.extend(text_results)

        if search_type in ["image", "hybrid"] and hasattr(self, "image_store"):
            image_results = self._search_store(self.image_store, query, top_k, filter_dict)
            all_results.extend(image_results)

        # 对hybrid的结果进行排序（其他两种模式返回结果本就是有序的）
        if search_type == "hybrid" and hasattr(self, "image_store") and self.image_store:
            all_results.sort(key=lambda x: x.score, reverse=True)
            # 取前top_k
            all_results = all_results[:top_k]

        # 给结果rank编号
        for i, result in enumerate(all_results):
            result.rank = i + 1

        return all_results

    def _search_store(
            self,
            store,
            query: str,
            top_k: int,
            filter_dict: Optional[Dict],
    ) -> List[SearchResult]:
        """从指定存储检索"""
        try:
            if not filter_dict:
                filter_dict = None

            docs_with_scores = store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_dict,
            )

            return [
                SearchResult(
                    document=document,
                    score=score,
                    rank=1,
                ) for document, score in docs_with_scores
            ]
        except Exception as e:
            logger.error(f"❌️MultimodalVectorStore查询'{store.__str__()}'失败，{e}")
            return []

    def _clean_metadata(self, documents: list[Document]) -> list[Document]:
        """清理文档中的复杂元数据"""
        cleaned_docs = []
        for doc in documents:
            # 创建新的元数据字典，确保所有值都是基本类型
            clean_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    clean_metadata[key] = value
                elif isinstance(value, (set, list, tuple)):
                    # 将集合、列表、元组转换为字符串
                    clean_metadata[key] = str(value)
                elif isinstance(value, dict):
                    # 将字典转换为字符串
                    clean_metadata[key] = str(value)
                else:
                    # 其他复杂类型也转换为字符串
                    clean_metadata[key] = str(value)

            # 按键排序以确保一致性
            sorted_metadata = dict(sorted(clean_metadata.items(), key=lambda item: item[0]))

            # 创建新的文档对象
            cleaned_doc = Document(
                page_content=doc.page_content,
                metadata=sorted_metadata
            )
            cleaned_docs.append(cleaned_doc)

        return cleaned_docs

    def _batch_store(
            self,
            store,
            documents: list[Document],
            batch_size: int,
            doc_type: str,
            show_progress: bool,
    ) -> list[str]:
        """批量存储 不舍弃最后的document"""
        if store and documents:
            ids = []
            total_batches = (len(documents) + batch_size - 1) // batch_size

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1

                # 生成确定性id集合
                try:
                    generated_ids = [self._generate_doc_id(doc) for doc in batch]
                    store.add_documents(batch, ids=generated_ids)
                    ids.extend(generated_ids)

                    if show_progress:
                        logger.info(
                            f"[{doc_type}] Batch {batch_num}/{total_batches} "
                            f"({len(batch)} docs)"
                        )
                except Exception as e:
                    logger.error(f"❌️Failed to add batch {batch_num}: {e}")
                    raise
            return ids

    def _generate_doc_id(self, doc: Document) -> str:
        """根据doc内容生成加密id"""
        content = doc.page_content + str(sorted(doc.metadata.items(), key=lambda x: x[0]))
        return hashlib.md5(content.encode()).hexdigest()

    @retry_on_failure()  # 删除和添加
    def update(self, id: str, document: Document) -> bool:
        """更新store"""
        try:
            # 先删除,再添加
            self.delete([id])
            self.add_documents([document])
            return True
        except Exception as e:
            logger.error(f"更新失败：{e}")
            return False

    @retry_on_failure()  # 删除可能失败
    def delete(self, ids: list[str]) -> bool:
        """根据id删除对应文档"""
        try:
            self.text_store.delete(ids)
            if self.image_store:
                self.image_store.delete(ids)
            self.metrics.total_documents -= len(ids)
            self.metrics.last_update_time = time.strftime("%Y-%m-%D %H:%M:%S")
            return True
        except Exception as e:
            logger.error(f"文档删除失败：{e}")
            return False

    def clear(self) -> bool:
        """清空存储"""
        try:
            # 如果是Chroma，需要删除集合
            if self.rag_config.vector_store.provider == "chroma":
                self.text_store.delete_collection()
                if self.image_store:
                    self.image_store.delete_collection()
            # 初始化向量库
            self._init_vector_stores()
            # 重置统计信息
            self.metrics.total_documents = 0

            # 清空缓存
            if self.cache:
                self.cache.clear()

            logger.info("✅️存储清空成功")
            return True
        except Exception as e:
            logger.error(f"❌️清空失败：{e}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """返回统计信息"""
        return {
            "total_documents": self.metrics.total_documents,
            "total_queries": self.metrics.total_queries,
            "avg_query_time_ms": round(self.metrics.avg_time_per_query_ms, 2),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": self.metrics.cache_hits / (
                max(1, self.metrics.cache_misses + self.metrics.cache_hits)
            ),
            "last_update_time": self.metrics.last_update_time,
            "backend": self.rag_config.vector_store.provider,
        }

    def as_retriever(self, **kwargs):
        """返回langchain的检索器"""
        search_kwargs = {
            "k": kwargs.get("k", self.rag_config.vector_store.default_top_k)
        }
        if "filter" in kwargs:
            search_kwargs["filter"] = kwargs["filter"]
        return self.text_store.as_retriever(search_kwargs=search_kwargs)

    def _init_embeddings(self):
        """初始化嵌入模型"""
        embedding_config = self.rag_config.embedding

        # 文本嵌入
        self.text_embedding = init_embedding_basic(
            embedding_config.text_embedding_model,
            embedding_config.device,
            embedding_config.normalize,
        )
        logger.info(f"✅️text_embedding 初始化成功 模型：{embedding_config.text_embedding_model}")

        # 图片嵌入
        if embedding_config.use_clip and embedding_config.clip_embedding_model:
            try:
                self.image_embedding = CLIPEmbeddings(model_name=embedding_config.clip_embedding_model,
                                                      batch_size=embedding_config.batch_size)
                self._has_clip = True
                logger.info(f"✅️clip_embedding 初始化成功 模型：{embedding_config.clip_embedding_model}")
            except Exception as e:
                logger.error(f"❌️加载CLIP模型失败: {e}")
                self._has_clip = False
        else:
            logger.info("CLIP模型未启用...")
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
        logger.info(f"✅️向量数据库初始化成功：{store_config.provider}")

    def _init_faiss_store(self, store_config):
        """初始化faiss向量数据库"""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.docstore.in_memory import InMemoryDocstore

            index = IndexUtil.get_faiss_index(
                self.text_embedding,
                store_config.faiss_index_type,
                store_config.faiss_nlist
            )
            # 如果不使用clip不需要分库存储
            self.text_store = FAISS(
                embedding_function=self.text_embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=store_config.distance_metric,
            )
            logger.info("✅FAISS-text_store 初始化成功")

            if self._has_clip:
                index = IndexUtil.get_faiss_index(
                    self.image_embedding,
                    store_config.faiss_index_type,
                    store_config.faiss_nlist
                )

                self.image_store = FAISS(
                    embedding_function=self.image_embedding,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                    distance_strategy=store_config.distance_metric,
                )
                logger.info("✅FAISS-image_store 初始化成功")
        except ImportError:
            raise ImportError("请安装faiss-cpu来使用Faiss向量数据库")

    def _init_chroma_store(self, store_config, persist_directory: Path):
        """初始化chroma向量数据库"""
        try:
            from langchain_chroma.vectorstores import Chroma
            self.text_store = Chroma(
                collection_name=store_config.collection_name + "_text",
                embedding_function=self.text_embedding,
                persist_directory=str(persist_directory / "text"),
            )
            logger.info(f"✅Chroma-text_store 初始化成功")
            if self._has_clip:
                self.image_store = Chroma(
                    collection_name=store_config.collection_name + "_image",
                    embedding_function=self.image_embedding,
                    persist_directory=str(persist_directory / "image"),
                )
            logger.info(f"✅Chroma-image_store 初始化成功")
        except ImportError:
            raise ImportError("请安装langchain_chroma or langchain_community（对Chroma已弃用）来使用Chroma向量数据库")

    @retry_on_failure(max_retries=3, delay=2.0)  # 远程服务
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
                collection_name=store_config.collection_name + "_text",
                connection_args=connection_args,
            )
            logger.info("✅️Milvus-text_store 初始化成功")

            if self._has_clip:
                self.image_store = Milvus(
                    embedding_function=self.image_embedding,
                    collection_name=store_config.collection_name + "_image",
                    connection_args=connection_args,
                )
            logger.info("✅️Milvus-image_store 初始化成功")
        except ImportError:
            raise ImportError("请安装langchain_milvus来使用Milvus向量数据库")

    @retry_on_failure(max_retries=3, delay=2.0)  # 远程服务
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
                collection_name=store_config.collection_name + "_text",
                embeddings=self.text_embedding,
                distance_strategy=store_config.distance_metric,
            )
            logger.info(f"✅️Qdrant-text_store 初始化成功")

            if self._has_clip:
                self.image_store = Qdrant(
                    client=qdrant_client,
                    collection_name=store_config.collection_name + "_image",
                    embeddings=self.image_embedding,
                    distance_strategy=store_config.distance_metric,
                )
            logger.info(f"✅️Qdrant-image_store 初始化成功")
        except ImportError:
            raise ImportError("请安装langchain_qdrant, qdrant_client来使用Qdrant向量数据库")

    def _init_reranker(self):
        reranker_config = self.rag_config.reranker
        if reranker_config.enabled is True:
            self.reranker = Reranker(
                model_name=reranker_config.model_name,
                device=reranker_config.device,
                top_k=reranker_config.top_k,
                batch_size=reranker_config.batch_size,
                max_length=reranker_config.max_length,
            )
            logger.info(f"✅️Cross-Encoder排序器初始化完成，{reranker_config.model_name}")
        else:
            logger.info("❌️未启用Cross-Encoder排序器")
            self.reranker = None

    @retry_on_failure()  # redis可能连接失败
    def _init_cache(self):
        cache_config = self.rag_config.cache
        if cache_config.enabled is True:
            self.cache = CacheManager(cache_config)
            logger.info(f"✅️cache 初始化成功")
        else:
            self.cache = None
            logger.info("❌cache 初始化失败（未启用）")


if __name__ == '__main__':
    file_path = r"C:\Users\ASUS\Desktop\video.pdf"
    # markdown_parser = MultimodalMarkdownParser(vision_llm=qwen_vision, use_vision=True)
    # documents = markdown_parser.parse(file_path)
    pdf_parser = PDFParser(vision_llm=qwen_vision)
    documents = pdf_parser.parse(file_path)
    document_split = IntegrationSplitter(
        semantic_embedding=init_embedding_basic(
            "BAAI/bge-base-zh-v1.5", 'cpu', True,
        ),
    ).split_documents_(documents, file_suffix='md')

    vector_store = MultimodalVectorStore(get_config())
    vector_store.add_documents(document_split, batch_size=2)
    results = vector_store.search("How to use it with ChatGPT", top_k=5, search_type="hybrid")
    for result in results:
        print(result)

    print("-" * 100)
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

# services/vector_store/multimodal_store.py
"""
企业级多模态向量存储
- 支持多种后端(Chroma/Milvus/Qdrant)
- 批量处理优化
- 重试机制
- 缓存层
- 监控指标
todo: 为什么没有用到clip模型？
"""
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal

from langchain_core.documents import Document

from python_services.vector_store.base_store import BaseVectorStore

logger = logging.getLogger(__name__)


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

    def _init_reranker(self):
        """初始化重排序器"""
        if self.config.reranker.enabled:
            self.reranker = Reranker(
                model_name=self.config.reranker.model,
                top_k=self.config.reranker.top_k,
            )
        else:
            self.reranker = None
    
    def _init_cache(self):
        """初始化缓存"""
        if self.config.cache.enabled:
            self.cache = CacheManager(self.config.cache)
        else:
            self.cache = None
    
    # ==================== 核心API ====================
    
    @timed
    @retry_on_failure(max_retries=3)
    def add_documents(
        self, 
        documents: List[Document],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """
        批量添加文档
        
        Args:
            documents: 文档列表
            batch_size: 批次大小
            show_progress: 是否显示进度
            
        Returns:
            文档ID列表
        """
        if not documents:
            return []
        
        batch_size = batch_size or self.config.vector_store.batch_size
        
        # 分离文本和图片文档
        text_docs = []
        image_docs = []
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "image":
                image_docs.append(doc)
            else:
                text_docs.append(doc)
        
        all_ids = []
        
        # 批量添加文本文档
        if text_docs:
            text_ids = self._batch_add_to_store(
                self.text_store, 
                text_docs, 
                batch_size,
                "text",
                show_progress
            )
            all_ids.extend(text_ids)
        
        # 批量添加图片文档
        if image_docs and self.image_store:
            image_ids = self._batch_add_to_store(
                self.image_store,
                image_docs,
                batch_size,
                "image", 
                show_progress
            )
            all_ids.extend(image_ids)
        
        # 更新指标
        self.metrics.total_documents += len(documents)
        self.metrics.last_update_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(
            f"Added {len(text_docs)} text docs and {len(image_docs)} image docs"
        )
        
        return all_ids
    
    def _batch_add_to_store(
        self,
        store,
        documents: List[Document],
        batch_size: int,
        doc_type: str,
        show_progress: bool,
    ) -> List[str]:
        """批量添加到指定存储"""
        all_ids = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # 生成确定性ID (基于内容hash)
                ids = [self._generate_doc_id(doc) for doc in batch]
                
                # 添加到存储
                store.add_documents(batch, ids=ids)
                all_ids.extend(ids)
                
                if show_progress:
                    logger.info(
                        f"[{doc_type}] Batch {batch_num}/{total_batches} "
                        f"({len(batch)} docs)"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to add batch {batch_num}: {e}")
                raise
        
        return all_ids
    
    def _generate_doc_id(self, doc: Document) -> str:
        """生成文档ID (基于内容hash)"""
        content = doc.page_content + str(sorted(doc.metadata.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    @timed
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
        search_type: Literal["text", "image", "hybrid"] = "text",
        use_reranker: bool = True,
        use_cache: bool = True,
    ) -> List[SearchResult]:
        """
        检索文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_dict: 元数据过滤
            search_type: 检索类型
            use_reranker: 是否使用重排序
            use_cache: 是否使用缓存
            
        Returns:
            检索结果列表
        """
        start_time = time.perf_counter()
        top_k = top_k or self.config.vector_store.default_top_k
        
        # 尝试从缓存获取
        cache_key = None
        if use_cache and self.cache:
            cache_key = self._build_cache_key(query, top_k, filter_dict, search_type)
            cached = self.cache.get(cache_key)
            if cached:
                self.metrics.cache_hits += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached
            self.metrics.cache_misses += 1
        
        # 执行检索
        results = self._execute_search(query, top_k, filter_dict, search_type)
        
        # 重排序
        if use_reranker and self.reranker and results:
            results = self.reranker.rerank(query, results)
        
        # 缓存结果
        if cache_key and self.cache:
            self.cache.set(cache_key, results)
        
        # 记录指标
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.record_query(duration_ms)
        
        return results
    
    def _execute_search(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict],
        search_type: str,
    ) -> List[SearchResult]:
        """执行检索"""
        results = []
        
        if search_type in ("text", "hybrid"):
            text_results = self._search_store(
                self.text_store, query, top_k, filter_dict
            )
            results.extend(text_results)
        
        if search_type in ("image", "hybrid") and self.image_store:
            image_results = self._search_store(
                self.image_store, query, top_k, filter_dict
            )
            results.extend(image_results)
        
        # 混合检索时按分数排序
        if search_type == "hybrid":
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
        
        # 设置排名
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _search_store(
        self,
        store,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict],
    ) -> List[SearchResult]:
        """从指定存储检索"""
        try:
            docs_with_scores = store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_dict,
            )
            
            return [
                SearchResult(
                    document=doc,
                    score=float(score),
                    rank=0,  # 稍后设置
                )
                for doc, score in docs_with_scores
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _build_cache_key(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict],
        search_type: str,
    ) -> str:
        """构建缓存键"""
        key_parts = [query, str(top_k), search_type]
        if filter_dict:
            key_parts.append(str(sorted(filter_dict.items())))
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def delete(self, ids: List[str]) -> bool:
        """删除文档"""
        try:
            self.text_store.delete(ids)
            if self.image_store:
                self.image_store.delete(ids)
            self.metrics.total_documents -= len(ids)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    def update(self, id: str, document: Document) -> bool:
        """更新文档 (删除后重新添加)"""
        try:
            self.delete([id])
            self.add_documents([document])
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": self.metrics.total_documents,
            "total_queries": self.metrics.total_queries,
            "avg_query_time_ms": round(self.metrics.avg_query_time_ms, 2),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": (
                self.metrics.cache_hits / 
                max(1, self.metrics.cache_hits + self.metrics.cache_misses)
            ),
            "last_update_time": self.metrics.last_update_time,
            "backend": self.config.vector_store.provider,
        }
    
    def as_retriever(self, **kwargs):
        """返回LangChain Retriever"""
        search_kwargs = {
            "k": kwargs.get("k", self.config.vector_store.default_top_k),
        }
        if "filter" in kwargs:
            search_kwargs["filter"] = kwargs["filter"]
        
        return self.text_store.as_retriever(search_kwargs=search_kwargs)
    
    def clear(self):
        """清空向量库"""
        try:
            # Chroma需要删除集合重建
            if self.config.vector_store.provider == "chroma":
                self.text_store.delete_collection()
                if self.image_store:
                    self.image_store.delete_collection()
                self._init_vector_stores()
            
            self.metrics.total_documents = 0
            
            # 清空缓存
            if self.cache:
                self.cache.clear()
                
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            raise
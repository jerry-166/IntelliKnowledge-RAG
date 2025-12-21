"""
混合检索---向量检索 + 关键词检索
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Any

from langchain_core.documents import Document

from core.search_results import SearchResult
from retriever.keyword_retriever import BM25Retriever, ElasticSearchRetriever, KeywordSearchResult
from vector_store.multimodal_store import MultimodalVectorStore

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """混合检索结果"""
    document: Document
    final_score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    matched_terms: List[str] = None
    rank: int = 0


class HybridRetriever:
    """
    混合检索器

    融合策略：
    1. RRF (Reciprocal Rank Fusion) - 推荐，对分数尺度不敏感
    2. Weighted Sum - 简单加权，需要分数归一化
    3. DBSF (Distribution-Based Score Fusion) - 基于分布的融合
    """

    def __init__(
            self,
            vector_store,  # 向量存储实例
            keyword_config,
            vector_weight: float = 0.6,
            keyword_weight: float = 0.4,
            fusion_method: Literal["rrf", "weighted", "dbsf"] = "rrf",
            rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

        self._init_keyword_retriever(keyword_config)

    def _init_keyword_retriever(self, config: dict):
        """初始化关键词检索器"""
        backend = config.get("backend", "bm25")

        if backend == "bm25":
            self.keyword_retriever = BM25Retriever(
                k1=config.get("bm25_k1", 1.5),
                b=config.get("bm25_b", 0.75),
                tokenizer=config.get("tokenizer", "jieba"),
                use_stopwords=config.get("use_stopwords", True),
                custom_stopwords=config.get("custom_stopwords", []),
                persist_path=config.get("persist_path"),
            )
        elif backend == "elasticsearch":
            self.keyword_retriever = ElasticSearchRetriever(
                host=config.get("es_host", "localhost"),
                port=config.get("es_port", 9200),
                index_name=config.get("es_index_name", "rag_documents"),
            )
        else:
            raise ValueError(f"不支持的关键词检索后端: {backend}")

        logger.info(f"✅ 关键词检索器初始化完成: {backend}")

    def add_documents(self, documents: list[Document]):
        """添加文档到向量存储和关键词存储库中"""
        self.vector_store.add_documents(documents=documents, batch_size=16, show_progress=False)
        self.keyword_retriever.add_documents(documents=documents)

    def search(
            self,
            query: str,
            top_k: int = 10,
            filter_dict: Optional[Dict[str, Any]] = None,
            vector_top_k: Optional[int] = None,
            keyword_top_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        """
        混合检索

        :param query: 查询文本
        :param top_k: 最终返回数量
        :param filter_dict: 过滤条件（仅向量检索支持）
        :param vector_top_k: 向量检索召回数量（默认top_k*2）
        :param keyword_top_k: 关键词检索召回数量（默认top_k*2）
        :return: 混合检索结果
        """
        # 查询缓存
        cached = self.vector_store.cache.get(query)
        if cached:
            logger.info(f"✅️缓存命中，问题：{query[:50]}")
            # 对缓存结果进行过滤，需要吗---需要top_k过滤
            return cached[:top_k]

        # 扩大召回范围以获得更好的融合效果
        vector_k = vector_top_k or top_k * 2
        keyword_k = keyword_top_k or top_k * 2

        # 1. 向量检索
        vector_results = self.vector_store.results(
            query=query,
            top_k=vector_k,
            filter_dict=filter_dict,
            search_type="text",
            use_reranker=False,  # 重排序在融合后进行
            use_cache=False,  # 混合检索单独管理缓存  # todo: 缓存机制有点重复... 混合和之前的向量混在一个缓存目录下
        )
        logger.info("向量召回{len(vector_results)}个")

        # 2. 关键词检索
        keyword_results = self.keyword_retriever.search(query, top_k=keyword_k)
        logger.info("关键词召回{len(keyword_results)}个")

        # 根据不同的融合方法 - 融合结果
        fused_results = []
        if self.fusion_method == "rrf":
            fused_results = self._rrf_fusion(vector_results, keyword_results, top_k)
        elif self.fusion_method == "weighted":
            fused_results = self._weighted_fusion(vector_results, keyword_results, top_k)
        else:
            fused_results = self._dbsf_fusion(vector_results, keyword_results, top_k)

        logger.info(f"混合检索完成: 融合后{len(fused_results)}个")

        # 对融合结果进行重排
        final_results = self.vector_store.reranker.rerank(query, fused_results)
        # 写入缓存
        self.vector_store.cache.set(query, final_results)

        # 指标记录...

        return final_results

    def _rrf_fusion(
            self,
            vector_results: List[SearchResult],
            keyword_results: List[KeywordSearchResult],
            top_k: int
    ) -> list[HybridSearchResult]:
        """
        RRF (Reciprocal Rank Fusion) 融合

        公式: RRF(d) = Σ weight * 1/(rrf_k + rank(d))

        优点：
        - 对分数尺度不敏感
        - 不需要归一化
        - 效果稳定
        """
        doc_scores: Dict[str, Dict] = {}
        # 处理向量检索到的文档
        for result in vector_results:
            doc_id = self._generate_doc_id(result.document)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": result.document,
                    "vector_score": result.score,
                    "vector_rank": result.rank,
                    "keyword_score": None,
                    "keyword_rank": None,
                    "matched_terms": [],
                    "rrf_score": 0.0
                }
            # 计算 RRF 分数
            doc_scores[doc_id]["rrf_score"] += self.vector_weight / (self.rrf_k + result.rank)

        # 处理关键词检索结果
        for result in keyword_results:
            doc_id = self._generate_doc_id(result.document)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": result.document,
                    "vector_score": None,
                    "vector_rank": None,
                    "keyword_score": result.score,
                    "keyword_rank": result.rank,
                    "matched_terms": result.matched_terms,
                    "rrf_score": 0.0
                }
            else:
                # 为什么这里还需要else？，为什么上面对于 vector_results 没有else？
                # 文档同时出现在向量和关键词检索结果中，更新关键词相关信息
                doc_scores[doc_id]["keyword_score"] = result.score
                doc_scores[doc_id]["keyword_rank"] = result.rank
                doc_scores[doc_id]["matched_terms"] = result.matched_terms
            # 计算 RRF 分数
            doc_scores[doc_id]["rrf_score"] += self.keyword_weight / (self.rrf_k + result.rank)

        # 排序
        sorted_results = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        # 封装并返回
        results = []
        for rank, doc_data in enumerate(sorted_results[:top_k], 1):
            results.append(HybridSearchResult(
                document=doc_data["document"],
                final_score=doc_data["rrf_score"],
                vector_score=doc_data["vector_score"],
                keyword_score=doc_data["keyword_score"],
                vector_rank=doc_data["vector_rank"],
                keyword_rank=doc_data["keyword_rank"],
                matched_terms=doc_data["matched_terms"],
                rank=rank
            ))

        return results

    def _weighted_fusion(
            self,
            vector_results: List[SearchResult],
            keyword_results: List[KeywordSearchResult],
            top_k: int
    ) -> list[HybridSearchResult]:
        """
        weighted_fusion:
            norm_score = (score - min_score) / (max_score - min_score)
            weight * norm_score
        """
        doc_scores: Dict[str, Dict] = {}

        # 归一化向量分数
        min_v, max_v, range_v = 0, 0, 1
        if vector_results:
            max_v = max(r.score for r in vector_results)
            min_v = min(r.score for r in vector_results)
            range_v = max_v - min_v if max_v != min_v else 1

        for result in vector_results:
            doc_id = self._generate_doc_id(result.document)
            norm_score = (result.score - min_v) / range_v if range_v else 0

            doc_scores[doc_id] = {
                "document": result.document,
                "vector_score": result.score,
                "vector_rank": result.rank,
                "keyword_score": None,
                "keyword_rank": None,
                "matched_terms": [],
                "weighted_score": self.vector_weight * norm_score
            }

        # 归一化关键词分数
        min_k, max_k, range_k = 0, 0, 1
        if keyword_results:
            max_k = max(r.score for r in keyword_results)
            min_k = min(r.score for r in keyword_results)
            range_k = max_k - min_k if max_k != min_k else 1

        for result in keyword_results:
            doc_id = self._generate_doc_id(result.document)
            norm_score = (result.score - min_k) / range_k if range_k else 0

            # 如果在向量检索和关键词检索中都出现了，则对未赋值的部分进行赋值
            if doc_id in doc_scores:
                doc_scores[doc_id]["keyword_score"] = result.score
                doc_scores[doc_id]["keyword_rank"] = result.rank
                doc_scores[doc_id]["matched_terms"] = result.matched_terms
                doc_scores[doc_id]["weighted_score"] += self.keyword_weight * norm_score
            else:
                doc_scores[doc_id] = {
                    "document": result.document,
                    "vector_score": None,
                    "vector_rank": None,
                    "keyword_score": result.score,
                    "keyword_rank": result.rank,
                    "matched_terms": result.matched_terms,
                    "weighted_score": self.keyword_weight * norm_score
                }
        # 排序
        sorted_results = sorted(doc_scores.values(), key=lambda x: x["weighted_score"], reverse=True)
        # 封装并返回
        results = []
        for rank, doc in enumerate(sorted_results[:top_k], 1):
            results.append(HybridSearchResult(
                document=doc["document"],
                final_score=doc["weighted_score"],
                vector_score=doc["vector_score"],
                keyword_score=doc["keyword_score"],
                vector_rank=doc["vector_rank"],
                keyword_rank=doc["keyword_rank"],
                matched_terms=doc["matched_terms"],
                rank=rank
            ))
        return results

    def _dbsf_fusion(
            self,
            vector_results,
            keyword_results,
            top_k
    ) -> list[HybridSearchResult]:
        """
        DBSF (Distribution-Based Score Fusion)
        基于分数分布的融合，使用z-score标准化:
            z_score * weight (z_score = (score - mean) / std)
        """
        import statistics

        doc_scores: Dict[str, Dict] = {}

        if vector_results:
            v_scores = [r.score for r in vector_results]
            mean = statistics.mean(v_scores)
            std = statistics.stdev(v_scores) if len(v_scores) > 1 else 1
        for result in vector_results:
            doc_id = self._generate_doc_id(result.document)
            z_score = (result.score - mean) / std if std else 0

            doc_scores[doc_id] = {
                "document": result.document,
                "vector_score": result.score,
                "vector_rank": result.rank,
                "keyword_score": None,
                "keyword_rank": None,
                "matched_terms": [],
                "dbsf_score": self.vector_weight * z_score
            }

        if keyword_results:
            k_scores = [k.score for k in keyword_results]
            mean = statistics.mean(k_scores)
            std = statistics.stdev(k_scores) if len(k_scores) > 1 else 1

        for result in keyword_results:
            doc_id = self._generate_doc_id(result.document)
            z_score = (result.score - mean) / std if std else 0
            if doc_id in doc_scores:
                # 补全
                doc_scores[doc_id]["keyword_score"] = result.score
                doc_scores[doc_id]["keyword_rank"] = result.rank
                doc_scores[doc_id]["matched_terms"] = result.matched_terms
                doc_scores[doc_id]["dbsf_score"] += self.keyword_weight * z_score
            else:
                # 新建
                doc_scores[doc_id] = {
                    "document": result.document,
                    "vector_score": None,
                    "vector_rank": None,
                    "keyword_score": result.score,
                    "keyword_rank": result.rank,
                    "matched_terms": result.matched_terms,
                    "dbsf_score": self.keyword_weight * z_score
                }

        # 排序
        sorted_results = sorted(doc_scores.values(), key=lambda x: x["dbsf_score"], reverse=True)
        # 封装并返回
        results = []
        for rank, doc_data in enumerate(sorted_results[:top_k], 1):
            results.append(HybridSearchResult(
                document=doc_data["document"],
                final_score=doc_data["dbsf_score"],
                vector_score=doc_data["vector_score"],
                keyword_score=doc_data["keyword_score"],
                vector_rank=doc_data["vector_rank"],
                keyword_rank=doc_data["keyword_rank"],
                matched_terms=doc_data["matched_terms"],
                rank=rank
            ))

        return results

    def clear(self):
        """清空索引"""
        self.vector_store.clear()
        self.keyword_retriever.clear()

    def _generate_doc_id(self, document):
        """获取文档唯一ID"""
        import hashlib
        content = document.page_content[:500]  # 使用前500字符进行hash加密
        return hashlib.md5(content.encode()).hexdigest()


if __name__ == '__main__':
    from core.settings import RetrieverConfig

    multimodal_vector_store = MultimodalVectorStore()
    retriever_config = RetrieverConfig()
    hybrid_retriever = HybridRetriever(
        vector_store=multimodal_vector_store,
        keyword_config=multimodal_vector_store.rag_config.keyword_config,
        vector_weight=retriever_config.vector_weight,
        keyword_weight=retriever_config.keyword_weight,
        fusion_method=retriever_config.fusion_method,
        rrf_k=retriever_config.rrf_k,
    )

    hybrid_retriever.add_documents([])
    query = ""
    results = hybrid_retriever.search(query=query)

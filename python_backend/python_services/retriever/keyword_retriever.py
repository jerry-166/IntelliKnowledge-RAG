"""
关键词检索器
- BM25
- elasticSearch
"""
import hashlib
import logging
import math
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

"""
1. 什么时候使用field？

2. jieba库是干嘛的？
    NLP中中文分词器
3. set的update是添加还是替换 --- 添加

4. 熟悉一下es的操作命令

"""


@dataclass
class KeywordSearchResult:
    """
    检索结果
    """
    document: Document
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)  # 存储匹配到的词
    rank: int = 0


class BaseKeywordRetriever(ABC):
    """
    基础检索器
    """

    @abstractmethod
    def add_documents(self, documents: list[Document]):
        """添加文档"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[KeywordSearchResult]:
        """检索"""
        pass

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> bool:
        """删除"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """清空"""
        pass


class ChineseTokenizer:
    """中文分词器"""

    def __init__(
            self,
            tokenizer_type: str = "jieba",
            use_stopwords: bool = True,
            custom_stopwords: Optional[List[str]] = None
    ):
        self.tokenizer_type = tokenizer_type
        self.use_stopwords = use_stopwords
        self.stopwords = self._load_stopwords(custom_stopwords)

        if self.tokenizer_type == "jieba":
            try:
                import jieba
                self.jieba = jieba
                # 启用paddle模式获得更好的分词效果（可选）
                self.jieba.enable_paddle()
            except ImportError:
                logger.warning("jieba未安装，使用简单分词")
                self.tokenizer_type = "simple"

    def _load_stopwords(self, custom_stopwords: Optional[List[str]]) -> set:
        """加载停用词"""
        stopwords = set()

        # 基础中文停用词
        base_stopwords = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
            "看", "好", "自己", "这", "那", "什么", "如何", "怎么", "为什么", "可以",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "of", "to", "in", "for", "on", "with", "at", "by", "from", "as",
            "and", "or", "but", "if", "then", "else", "when", "where", "how",
        }
        stopwords.update(base_stopwords)

        if custom_stopwords:
            stopwords.update(custom_stopwords)

        return stopwords

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []

        text = text.lower()
        if self.tokenizer_type == "jieba":
            # 使用jieba分词
            tokens = list(self.jieba.cut_for_search(text))
        elif self.tokenizer_type == "simple":
            # 简单分词：按空格和标点切分
            tokens = re.findall(r'\b\w+\b|[\u4e00-\u9fff]+', text)
        else:
            tokens = text.split(" ")

        # 检查并去除停用词
        if self.use_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords and len(token) > 1]
        return tokens


class BM25Retriever(BaseKeywordRetriever):
    """
    BM25关键词检索器

    特点：
    - 纯内存实现，无需额外服务
    - 支持中文分词
    - 支持持久化
    - 增量更新
    """

    def __init__(
            self,
            k1: float = 1.5,
            b: float = 0.75,
            tokenizer: str = "jieba",
            use_stopwords: bool = True,
            custom_stopwords: Optional[List[str]] = None,
            persist_path: Optional[str] = None,
    ):
        # 基础值
        self.k1 = k1
        self.b = b
        self.tokenizer = ChineseTokenizer(
            tokenizer, use_stopwords, custom_stopwords
        )
        self.persist_path = persist_path
        # 索引数据结构
        self.documents: List[Document] = []  # [doc1, doc2, ...] 与文档id对应
        self.doc_ids: List[str] = []  # [doc_id1, doc_id2, ...] 与文档对应
        self.tokenized_docs: List[List[str]] = []  # [[term1, term2...], [], ...]
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        # 倒排索引 {term: [(doc_idx, term_freq), (doc_idx, term_freq), ...], ...}
        self.inverted_index: Dict[str, Tuple[int, int]] = {}
        # doc_frequencies {term: doc_freq, ...} term -> 包含该词的文档数
        self.doc_freqs: Dict[str, int] = {}

        # 尝试从本地加载索引
        if self.persist_path and Path(self.persist_path).exists():
            self._load_index()

    def add_documents(self, documents: list[Document]):
        if not documents:
            return

        for doc in documents:
            doc_id = self._generate_doc_id(doc)

            # 跳过已存在的文档
            if doc_id in self.doc_ids:
                continue

            tokens = self.tokenizer.tokenize(doc.page_content)
            # 分词为空，跳过（没有意义，类似空文档）
            if tokens is None:
                continue

            doc_idx = len(self.documents)
            self.documents.append(doc)
            self.doc_ids.append(doc_id)
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))  # todo：？？？

            # 构建倒排索引
            term_freqs = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1

            for term, freq in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                    self.doc_freqs[term] = 0
                self.inverted_index[term].append((doc_idx, freq))  # term在文档中的频率
                self.doc_freqs[term] += 1  # 包含该词的文档数

            # 更新平均文档长度
            if self.doc_lengths:
                self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)

            # 持久化
            if self.persist_path:
                self._save_index()

            logger.info(f"✅ BM25索引更新完成，共{len(self.documents)}个文档")

    def search(self, query: str, top_k: int = 5) -> list[KeywordSearchResult]:
        # 非空判断
        if not query or not self.documents:
            return []

        # 把query进行分词
        query_tokens = self.tokenizer.tokenize(query)
        # 非空判断
        if not query_tokens:
            return []

        # 然后查询构建分数
        scores = {}  # 每一个文档的分数
        matched_terms = {}  # 匹配的词

        for term in query_tokens:
            # 索引中没有该词，跳过
            if term not in self.inverted_index:
                continue

            idf = self._cal_idf(len(self.documents), self.doc_freqs[term])
            # 遍历含该词的所有文档
            for doc_idx, term_freq in self.inverted_index[term]:
                # 初始化赋值
                if doc_idx not in scores:
                    scores[doc_idx] = 0.0
                    matched_terms[doc_idx] = []
                # 计算BM25分数 + 匹配词
                scores[doc_idx] += self._cal_bm25_score(term_freq, self.doc_lengths[doc_idx], idf)
                matched_terms[doc_idx].append(term)

        # 对结果排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 封装结果并返回
        return [
            KeywordSearchResult(
                document=self.documents[doc_idx],
                score=score,
                matched_terms=matched_terms[doc_idx],
                rank=i,
            ) for i, (doc_idx, score) in enumerate(sorted_results[:top_k], 1)
        ]

    def _cal_bm25_score(self, tf: int, doc_len: int, idf: float) -> float:
        """计算单个词的BM25分数"""
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        return idf * numerator / denominator

    def _cal_idf(self, n: int, df: int) -> float:
        """计算idf"""
        import math
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def delete(self, ids: list[str]) -> bool:
        """删除索引较复杂，这里简化使用排除需删除的ids，然后重建索引"""
        remaining_docs = [
            doc for doc, doc_id in zip(self.documents, self.doc_ids)
            if doc_id not in ids
        ]
        # 删除索引
        self.clear()
        # 重建索引
        self.add_documents(remaining_docs)
        return True

    def clear(self) -> bool:
        """清空索引"""
        self.documents = []
        self.doc_ids = []
        self.tokenized_docs = []
        self.doc_lengths = []
        self.avgdl = 0.0
        self.inverted_index = {}
        self.doc_freqs = {}

        # 删除索引文件
        if self.persist_path:
            Path(self.persist_path).unlink(missing_ok=True)
        return True

    def _load_index(self):
        """从本地加载索引"""
        if not self.persist_path and not Path(self.persist_path).exists():
            return
        try:
            # 读取索引文件
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            # 恢复索引数据
            self.documents = data["documents"]
            self.doc_ids = data["doc_ids"]
            self.tokenized_docs = data["tokenized_docs"]
            self.doc_lengths = data["doc_lengths"]
            self.avgdl = data["avgdl"]
            self.inverted_index = data["inverted_index"]
            self.doc_freqs = data["doc_freqs"]

            logger.info(f"✅️ 索引已从{self.persist_path}加载，文档数：{len(self.documents)}")
        except Exception as e:
            logger.error(f"❌ 加载BM25索引失败: {e}")

    def _save_index(self):
        if not self.persist_path:
            return

        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "tokenized_docs": self.tokenized_docs,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "inverted_index": self.inverted_index,
            "doc_freqs": self.doc_freqs,
        }

        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_doc_id(self, doc: Document) -> str:
        """生成文档ID"""
        content = doc.page_content + str(sorted(doc.metadata.items()))
        return hashlib.md5(content.encode()).hexdigest()


class ElasticSearchRetriever(BaseKeywordRetriever):
    """
    ElasticSearch 检索器
    """

    def __init__(
            self,
            host: str = "localhost",
            port: int = 9200,
            index_name: str = "rag_documents",
            username: Optional[str] = None,
            password: Optional[str] = None,
    ):
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError("请安装 elasticsearch: pip install elasticsearch")

        es_config = {"hosts": [f"http://{host}:{port}"]}
        if username and password:
            es_config["basic_auth"] = (username, password)

        # 连接ES
        self.es = Elasticsearch(**es_config)
        self.index_name = index_name
        # 创建索引
        self._create_index()

    def _create_index(self):
        # 判断索引是否已存在
        if self.es.indices.exists(index=self.index_name):
            return

        # 创建索引
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "ik_smart_analyzer": {
                            "type": "custom",
                            "tokenizer": "ik_max_word",  # 需要安装IK分词插件
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer",
                        "search_analyzer": "ik_smart"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    },
                    "doc_id": {
                        "type": "keyword"
                    }
                }
            }
        }

        try:
            self.es.indices.create(index=self.index_name, body=mapping)
        except Exception as e:
            # 可能是IK分词器未安装
            logger.warning(f"IK分词器可能未安装，使用标准分词: {e}")
            mapping["settings"] = {}
            mapping["mappings"]["properties"]["content"]["analyzer"] = "standard"
            self.es.indices.create(index=self.index_name, body=mapping)

    def add_documents(self, documents: list[Document]):
        """批量添加文档"""
        from elasticsearch.helpers import bulk

        actions = []
        for doc in documents:
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            actions.append({
                "_index": self.index_name,
                "_id": doc_id,
                "_source": {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "doc_id": doc_id,
                }
            })

        bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)
        logger.info(f"✅ ES索引更新完成，添加{len(documents)}个文档")

    def search(self, query: str, top_k: int = 5) -> list[KeywordSearchResult]:
        """es检索"""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content"],
                    "type": "best_fields",
                }
            },
            "size": top_k,
            "highlight": {
                "field": {
                    "content": {}
                }
            }
        }

        response = self.es.search(index=self.index_name, body=body)
        if not response:
            return []

        results = []
        for rank, hit in enumerate(response["hits"]["hits"], 1):
            doc = Document(
                page_content=hit["_source"]["content"],
                metadata=hit["_source"].get("metadata", {})
            )
            results.append(KeywordSearchResult(
                document=doc,
                score=hit["_score"],
                matched_terms=[],  # ES不直接返回匹配词
                rank=rank
            ))

        return results

    def delete(self, doc_ids: list[str]) -> bool:
        """删除文档"""
        for doc_id in doc_ids:
            self.es.delete(index=self.index_name, id=doc_id, ignore=[404])
        return True

    def clear(self) -> bool:
        """清空索引"""
        self.es.delete(index=self.index_name, ignore=[404])
        self._create_index()
        return True

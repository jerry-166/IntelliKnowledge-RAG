"""
重排器-reranker
"""
import logging
from typing import Optional
from langchain_core.documents import Document
from python_backend.python_services.core.search_results import SearchResult

logger = logging.getLogger(__name__)

"""
1. 之前检索时也会有score，是在document里面吗？
    SearchResult(
            document=Document(page_content="我喜欢吃pizza"),
            score=0.7,
        ),
2.score = self.model.predict((query, result.document.page_content), show_progress_bar=False)[0]
    返回的是列表，[0]指第一个
    
"""


class Reranker:
    """Cross_encoder-重排"""
    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-base",
            device: str = 'cpu',
            top_k: int = 5,
            batch_size: int = 16,
            max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_length = max_length

        self._init_reranker_model()

    def _init_reranker_model(self):
        """初始化Cross_Encoder模型"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                model_name_or_path=self.model_name,
                trust_remote_code=True,
                device=self.device,
                max_length=self.max_length,
            )
        except ImportError:
            logger.error("❌️请安装 sentence_transformers 库")
            self.model = None
        except Exception as e:
            logger.error(f"❌️Cross-Encoder排序器初始化失败: {e}")
            self.model = None

    def rerank(self, query: str, results: list[SearchResult], top_k: Optional[int] = None) -> list[SearchResult]:
        """重排"""
        # 若模型未初始化成功，则返回原始文档
        if self.model is None:
            logger.warning("Cross-Encoder排序器未初始化，返回原始文档")
            return results

        top_k = top_k or self.top_k
        try:
            # 构建 问题-文档 列表
            query_doc_list = [
                (query, result.document.page_content) for result in results
            ]
            # 获得每个文档匹配问题的分数
            relevant_scores = self.model.predict(query_doc_list, batch_size=self.batch_size, show_progress_bar=False)
            # 为结果文档添加分数
            for result, score in zip(results, relevant_scores):
                result.score = score
            # 排序
            results.sort(key=lambda x: x.score, reverse=True)

            # 为top_k个results进行排序
            for i, result in enumerate(results[:top_k]):
                result.rank = i + 1

            logger.info(f"✅️Cross-Encoder排序器重排完成，top_k={top_k}")
            return results[:top_k]
        except Exception as e:
            logger.error(f"❌️Cross-Encoder排序器重排失败: {e}")
            return results[:top_k]

    def compute_score(self, query: str, result: SearchResult):
        """计算文档和提问的相关性分数"""
        if self.model is None:
            return 0.0

        try:
            score = self.model.predict((query, result.document.page_content), show_progress_bar=False)[0]
            return float(score)
        except Exception as e:
            logger.error(f"❌️Cross-Encoder排序器计算分数失败，{e}")
            return 0.0


if __name__ == '__main__':
    reranker = Reranker()
    query = "我喜欢吃什么？"
    results = [
        SearchResult(
            document=Document(page_content="我喜欢吃pizza"),
            score=0.7,
        ),
        SearchResult(
            document=Document(page_content="我喜欢吃牛排"),
            score=0.8,
        ),
        SearchResult(
            document=Document(page_content="我喜欢小牛牛"),
            score=0.8,
        ),
        SearchResult(
            document=Document(page_content="我喜欢小猫"),
            score=0.9,
        ),
    ]
    rerank_results = reranker.rerank(query, results)
    print(rerank_results)

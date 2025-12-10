"""
vector_store.py
向量数据库构建与检索
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # 或FAISS, Milvus等
from langchain_core.documents import Document
from transformers import CLIPModel, CLIPProcessor


class MultimodalVectorStore:
    """多模态向量数据库"""

    def __init__(
            self,
            persist_directory: str = "./vector_db",
            text_embedding_model: str = "BAAI/bge-base-zh-v1.5",
            clip_embedding_model: str = "openai/clip-vit-base-patch32",
            collection_name: str = "multimodal_docs"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # 文本嵌入
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=text_embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # 图片嵌入
        self.clip_model = CLIPModel.from_pretrained(clip_embedding_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_embedding_model)
        self.clip_embeddings = HuggingFaceEmbeddings(
            model_name=clip_embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 文本向量库
        self.text_vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.text_embeddings,
            persist_directory=str(self.persist_directory)
        )
        # 图片向量库
        self.image_vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.clip_embeddings,
            persist_directory=str(self.persist_directory)
        )

    def add_text_documents(self, documents: List[Document],):
        """添加文档到向量库"""
        self.text_vectorstore.add_documents(documents)

    def as_retriever(self, **kwargs):
        """返回LangChain Retriever"""
        return self.vectorstore.as_retriever(**kwargs)

    def persist(self):
        """持久化向量库"""
        self.vectorstore.persist()

    def search(
            self,
            query: str,
            top_k: int = 10,
            filter_dict: Optional[Dict] = None,
            with_images_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        检索文档

        Args:
            query: 查询文本
            top_k: 返回数量
            filter_dict: 过滤条件
            with_images_only: 只返回包含图片的文档

        Returns:
            检索结果列表
        """
        if with_images_only:
            filter_dict = filter_dict or {}
            filter_dict['has_images'] = True

        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
            filter=filter_dict
        )

        output = []
        for doc, score in results:
            result = {
                'content': doc.page_content,
                'score': float(score),
                'metadata': doc.metadata,
                'has_image': doc.metadata.get('has_images', False),
                'image_paths': doc.metadata.get('image_paths', [])
            }
            output.append(result)

        return output

    def recall(self):
        pass

    def reranker(self):
        pass


if __name__ == '__main__':
    pass

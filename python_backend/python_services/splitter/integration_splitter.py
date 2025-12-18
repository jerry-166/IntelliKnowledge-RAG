"""
整合的splitter
"""
import os

from python_services.core.settings import get_config

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from basic_core.llm_factory import qwen_vision
from python_services.parsers.image_parser import ImageParser
from python_services.parsers.markdown_parser import MultimodalMarkdownParser
from python_services.parsers.pdf_parser import PDFParser
from python_services.parsers.ppt_parser import PPtParser
from python_services.parsers.text_parser import TextParser
from python_services.parsers.word_parser import WordParser

RAGConfig = get_config()


class IntegrationSplitter:
    def __init__(
            self,
            semantic_embedding,
            chunk_size: int = RAGConfig.splitter.chunk_size,
            chunk_overlap: int = RAGConfig.splitter.chunk_overlap,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=RAGConfig.splitter.separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.semantic_splitter = SemanticChunker(
            embeddings=semantic_embedding,
            breakpoint_threshold_type=RAGConfig.splitter.breakpoint_threshold_type,
            min_chunk_size=RAGConfig.splitter.min_chunk_size,
            breakpoint_threshold_amount=RAGConfig.splitter.breakpoint_threshold_amount,
        )

        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents_(self, documents: List[Document], file_suffix: str = 'txt', ) -> List[Document]:
        """切分Document"""
        docs = []
        text_docs = []
        image_docs = []

        for document in documents:
            if document.metadata.get('type') == 'text':
                text_docs.append(document)
            elif document.metadata.get('type') == 'image':
                image_docs.append(document)

        # todo: 根据type类型选择不同的切分器，切分文本的Document
        if file_suffix == 'md':
            splitter_md_documents = self.markdown_splitter.split_documents(text_docs)
            docs.extend(splitter_md_documents)
        elif file_suffix == 'txt':
            splitter_text_documents = self.text_splitter.split_documents(text_docs)
            docs.extend(splitter_text_documents)
        elif file_suffix == 'pdf':
            splitter_text_documents = self.text_splitter.split_documents(text_docs)
            docs.extend(splitter_text_documents)
        elif file_suffix in ['docx', 'pptx']:
            splitter_text_documents = self.semantic_splitter.split_documents(text_docs)
            docs.extend(splitter_text_documents)

        docs.extend(image_docs)

        return docs


if __name__ == '__main__':
    # 数据加载器
    loader = {
        "md": MultimodalMarkdownParser(qwen_vision, use_vision=True),
        "pdf": PDFParser(qwen_vision),
        "docx": WordParser(),
        "pptx": PPtParser(),
        "txt": TextParser(),
        "png": ImageParser(),
        "jpg": ImageParser(),
        "jpeg": ImageParser(),
        # "html": HtmlParser(),
        # "epub": EpubParser(),
        # "csv": CsvParser(),
    }
    model_kwargs = {
        'device': 'cpu',
        'trust_remote_code': True
    }
    splitter = IntegrationSplitter(
        semantic_embedding=HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-zh-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}
        ),
    )
    file_path = r"C:\Users\ASUS\Desktop\makedown\deepAgent.md"
    documents = loader['md'].parse(file_path)
    documents_split = splitter.split_documents_(documents, file_type='md')
    print(len(documents_split))
    pass

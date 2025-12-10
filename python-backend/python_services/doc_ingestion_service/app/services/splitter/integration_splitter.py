"""
整合的splitter
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from basic_core.llm_factory import qwen_vision
from python_services.doc_ingestion_service.app.services.parsers.image_parser import ImageParser
from python_services.doc_ingestion_service.app.services.parsers.markdown_parser import MultimodalMarkdownParser
from python_services.doc_ingestion_service.app.services.parsers.pdf_parser import PDFParser
from python_services.doc_ingestion_service.app.services.parsers.ppt_parser import PPtParser
from python_services.doc_ingestion_service.app.services.parsers.text_parser import TextParser
from python_services.doc_ingestion_service.app.services.parsers.word_parser import WordParser


class IntegrationSplitter:
    def __init__(self, embeddings, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ],
        )

    def _split_documents(self, documents: List[Document], file_type: str = 'txt',) -> List[Document]:
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
        splitter_text_documents = self.text_splitter.split_documents(text_docs)
        # 存入切分好的和不切分的图片
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

    pass

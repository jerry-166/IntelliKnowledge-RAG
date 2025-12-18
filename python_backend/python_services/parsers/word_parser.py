"""
word的解析器
"""
from langchain_core.documents import Document

from python_services.parsers.base_parser import BaseParser
from langchain_community.document_loaders.word_document import (
    UnstructuredWordDocumentLoader,
    )


class WordParser(BaseParser):
    """
    word的解析器
    """
    def __init__(self):
        super().__init__("Word解析器", ["doc", "docx"])

    def parse_impl(self, file_path_or_url: str) -> list[Document]:
        """解析Word的主要函数"""
        loader = UnstructuredWordDocumentLoader(file_path_or_url)
        docs = loader.load()
        print(f"目前解析word文件的主要方式是：UnstructuredWordDocumentLoader")
        return docs


if __name__ == '__main__':
    parser = WordParser()
    doc_path = r"C:\Users\ASUS\Desktop\word\test.docx"
    docs = parser.parse(doc_path)
    print(docs)

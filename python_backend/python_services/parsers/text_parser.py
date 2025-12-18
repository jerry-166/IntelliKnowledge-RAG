"""
对普通的文本文件、代码文件等单一格式文件解析
"""
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from python_services.parsers.base_parser import BaseParser


class TextParser(BaseParser):
    def __init__(self):
        super().__init__("Text解析器", ["txt"])

    def parse_impl(self, file_path_or_url: str) -> list[Document]:
        """解析Text的主要函数"""
        loader = TextLoader(file_path_or_url, encoding='utf-8')
        loader_docs = loader.load()
        return loader_docs

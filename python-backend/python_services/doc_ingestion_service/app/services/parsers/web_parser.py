"""
web解析器
"""
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from python_services.doc_ingestion_service.app.services.parsers.base_parser import BaseParser


class WebParser(BaseParser):
    def __init__(self, use_ocr: bool = True, use_vision: bool = False):
        super().__init__("Web解析器", "Web")
        self.use_ocr = use_ocr
        self.use_vision = use_vision

    def parse(self, url: str) -> list[Document]:
        """解析Web页面的主要函数"""
        url_loader = UnstructuredURLLoader(urls=[url])
        web_loader = WebBaseLoader(url)
        doc1 = url_loader.load()
        doc2 = web_loader.load()
        print(f"目前对Web页面解析的方式：UnstructuredURLLoader，请等待后续更新~~~")
        return doc1 + doc2


if __name__ == '__main__':
    parser = WebParser()
    url = "https://example.com"
    documents = parser.parse(url)
    print(documents)

"""
ppt解析器---待完成
"""
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from python_services.parsers.base_parser import BaseParser


class PPtParser(BaseParser):

    def __init__(self, vision_llm=None, use_ocr: bool = True, use_vision: bool = False):
        super().__init__("PPT解析器", ["pptx", "ppt"])
        self.vision_llm = vision_llm
        self.use_ocr = use_ocr
        self.use_vision = use_vision
        # 存储图片uuid和图片内容的映射关系
        self.image_mapping: dict = {}

    def parse_impl(self, file_path_or_url: str) -> list[Document]:
        """解析PPT的主要函数"""
        power_point_loader = UnstructuredPowerPointLoader(file_path_or_url)
        loader_docs = power_point_loader.load()
        print(f"目前对PPT解析的方式：UnstructuredPowerPointLoader，请等待后续更新~~~")
        return loader_docs


if __name__ == '__main__':
    parser = PPtParser(vision_llm=None, use_ocr=True, use_vision=True)
    ppt_path = r"C:\Users\ASUS\Desktop\ppt\ppt.pptx"
    documents = parser.parse(ppt_path)
    print(documents)
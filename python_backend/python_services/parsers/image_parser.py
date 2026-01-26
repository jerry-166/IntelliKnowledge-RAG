"""
对上传的图片进行解析
todo: 需要解析？还是直接通过多模态模型转向量？？？
"""
from typing import Optional

from langchain_core.documents import Document

from basic_core.llm_factory import qwen_vision
from python_services.parsers.base_parser import BaseParser
from python_services.utils.ocr_util import OcrUtil
from python_services.utils.image_util import ImageUtil

import os
from pathlib import Path


class ImageParser(BaseParser):
    def __init__(self, vision_llm: Optional[object]=None):
        super().__init__("图片解析器", ["png", "jpg", "jpeg"])
        self.vision_llm = vision_llm

    def parse_impl(self, file_path_or_url: str) -> Document:
        """解析图片的主要函数"""
        image = ImageUtil.load_image(file_path_or_url, Path(os.path.dirname(file_path_or_url)))
        base64_str = ImageUtil.image_to_bytes(image)
        res = ImageUtil.image_to_bytes(image=image)
        content = OcrUtil.describe_image(vision_llm=self.vision_llm, image_bytes=res[0])

        document = Document(
            page_content=content,
            metadata={
                "source": file_path_or_url,
                "type": "image",
                "base64": base64_str,
            }
        )
        return document


if __name__ == '__main__':
    parser = ImageParser(vision_llm=qwen_vision)
    file_path = "C:\\Users\\ASUS\\Pictures\\美女.png"
    docs = parser.parse(file_path)
    for doc in docs:
        print(f"文档内容: {doc.page_content[:100]}...")
        print(f"元数据: {doc.metadata}")

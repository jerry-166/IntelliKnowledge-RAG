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
    def __init__(self, vision_llm=None, use_vision: bool = False, use_ocr: bool = True):
        super().__init__("图片解析器", ["png", "jpg", "jpeg"])
        self.vision_llm = vision_llm
        self.use_vision = use_vision
        self.use_ocr = use_ocr

    def parse_impl(self, file_path_or_url: str) -> Document | None:
        """解析图片的主要函数"""
        try:
            # 使用OCR工具处理图片
            image = ImageUtil.load_image(file_path_or_url, Path(os.path.dirname(file_path_or_url)))
            image_bytes = image.tobytes()
            if not image_bytes:
                print(f"图片加载失败")
                return None
            if self.use_ocr and self.use_vision:
                try:
                    vision_content = OcrUtil.vision_ocr(vision_llm=self.vision_llm, image_bytes=image_bytes)
                    vision_doc = Document(
                        page_content=vision_content,
                        metadata={
                            "source": file_path_or_url,
                            "type": "image",
                            "ocr_": self.vision_llm.model_name
                        }
                    )
                    print(f"使用视觉模型成功解析图片")
                    return vision_doc
                except Exception as e:
                    print(f"视觉模型: {self.vision_llm.model_name} 解析失败: {e}")
            elif self.use_ocr:
                try:
                    ocr_content = OcrUtil.tesseract_ocr(image_bytes)
                    ocr_doc = Document(
                        page_content=ocr_content,
                        metadata={
                            "source": file_path_or_url,
                            "type": "image",
                            "ocr_": "Tesseract OCR"
                        }
                    )
                    print(f"使用Tesseract OCR成功解析图片")
                    return ocr_doc
                except Exception as e:
                    print(f"Tesseract OCR解析失败: {e}")
        except Exception as e:
            print(f"图片处理过程中出现错误: {e}")

        return None


if __name__ == '__main__':
    parser = ImageParser(vision_llm=qwen_vision, use_vision=True, use_ocr=True)
    file_path = "C:\\Users\\ASUS\\Pictures\\美女.png"
    docs = parser.parse(file_path)
    for doc in docs:
        print(f"文档内容: {doc.page_content[:100]}...")
        print(f"元数据: {doc.metadata}")

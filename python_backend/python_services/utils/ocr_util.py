"""
ocr的工具类
"""

import base64
import io
from typing import Optional
from PIL import Image

# 静态方法与普通方法对比
"""
特性      |    静态方法 @staticmethod   |   普通方法
调用方式   |    ClassName.method()     |  instance.method()
访问权限   |    无法访问 self           |  可以访问 self 和实例属性
依赖关系   |    独立于类实例             |   依赖类实例和实例状态

建议采用静态方法，因为OCR工具函数通常：
不需要维护实例状态
是纯函数式的工具方法
便于在不同场景下复用
符合工具类的设计模式
"""


class OcrUtil:
    @staticmethod
    def vision_ocr(vision_llm, image: Image.Image) -> str:
        """
        使用视觉LLM进行OCR识别

        :param vision_llm: 视觉LLM实例
        :param image: PIL.Image对象（不是bytes！）
        :return: 识别的文本
        """
        if not vision_llm:
            return ""

        try:
            # 🔥 关键修复：正确转换为PNG/JPEG字节流
            from python_services.utils.image_util import ImageUtil

            # 优先使用JPEG（阿里云兼容性更好，体积更小）
            # 如果图片有透明通道，使用PNG
            if image.mode in ('RGBA', 'LA', 'P'):
                img_bytes, img_format = ImageUtil.image_to_bytes(image, format="PNG")
            else:
                img_bytes, img_format = ImageUtil.image_to_bytes(image, format="JPEG", quality=90)

            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请提取图片中的所有文字内容，保持原有结构和格式。如果有表格，以Markdown表格格式输出。如果是图形/图表，请描述其内容。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            response = vision_llm.invoke(messages)
            return response.content

        except Exception as e:
            print(f"视觉LLM OCR识别失败：{e}")
            return ""

    @staticmethod
    def tesseract_ocr(image: Image.Image) -> str:
        """
        使用pytesseract进行OCR

        :param image: PIL.Image对象（不是bytes！）
        """
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r'D:\ASUS\develop\tesseract\Tesseract-OCR\tesseract.exe'

            # 转换为RGB模式（tesseract兼容性更好）
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return pytesseract.image_to_string(
                image,  # 直接传PIL.Image，不需要转bytes
                lang="chi_sim+eng",
                config=r'--tessdata-dir D:/ASUS/develop/tesseract/Tesseract-OCR/tessdata'
            )
        except ImportError:
            print("pytesseract未安装，跳过OCR")
            return ""
        except Exception as e:
            print(f"pytesseract OCR失败: {e}")
            return ""

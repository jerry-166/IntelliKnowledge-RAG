"""
ocr的工具类
"""

import base64
import io

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
    def vision_ocr(vision_llm, image_bytes: bytes) -> str:
        """使用视觉LLM进行OCR识别和理解"""
        if not vision_llm:
            return ""

        try:
            base64_image = base64.b64encode(image_bytes).decode()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "根据情况提取文档中的所有文字内容或者分析图片内容。当提取这个文档页面中的所有文字内容时，保持原有的结构和格式（如果有表格，以Markdown表格格式输出）；当分析图片内容时，返回一个对于图片内容的段落描述。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64, {base64_image}"
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
    def tesseract_ocr(img_bytes: bytes) -> str:
        """使用pytesseract进行OCR"""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r'D:\ASUS\develop\tesseract\Tesseract-OCR\tesseract.exe'
            pil_image = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(
                pil_image,
                lang="chi_sim+eng",
                config=r'--tessdata-dir D:/ASUS/develop/tesseract/Tesseract-OCR/tessdata'
            )
        except ImportError:
            print("pytesseract未安装，跳过OCR")
            return ""
        except Exception as e:
            print(f"pytesseract OCR失败: {e}")
            return ""

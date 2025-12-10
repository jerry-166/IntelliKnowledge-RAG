"""
markdown解析器
解析图片、文字
todo: 表格、链接怎么换回去，可以记录位置吗？
todo: Markdown格式保留处理？标题、正文...
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from basic_core.llm_factory import qwen_vision
from python_services.doc_ingestion_service.app.services.parsers.base_parser import BaseParser
from python_services.doc_ingestion_service.app.services.utils.image_util import ImageUtil
from python_services.doc_ingestion_service.app.services.utils.ocr_util import OcrUtil


class ElementType(Enum):
    """元素类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    LINK = "link"
    HEADER = "header"  # Markdown格式保留处理？


@dataclass
class MarkdownElement:
    """markdown内容元素"""
    type: ElementType
    content: Any
    metadata: dict = field(default_factory=dict)


class MultimodalMarkdownParser(BaseParser):
    PATTERNS = {
        'image': r'!\[([^\]]*)\]\(([^)]+)\)',  # ![alt](url)
        'table': r'\|(.*)\|',
        'link': r'\[([^\]]+)\]\(([^)]+)\)',
        'header': r'^#+ (.+)',
    }

    def __init__(self, vision_llm=None, use_ocr: bool = True, use_vision: bool = False):
        super().__init__("Markdown解析器", "Markdown")
        self.vision_llm = vision_llm
        self.use_ocr = use_ocr
        self.use_vision = use_vision
        # 存储图片uuid和图片内容的映射关系
        self.image_mapping: dict = {}

    def parse(self, file_path: str) -> list[Document]:
        """解析Markdown的主要函数"""
        documents = []

        # 判断文件是否存在
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 使用langchain的Markdown加载器读取Markdown文件
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        if len(docs) == 0:
            return []
        raw_text_with_uuid = docs[0].page_content

        # 使用file-open获取图片Document、todo：表格、链接
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        base_dir = file_path.parent
        image_elements = self._extract_image_elements(original_content, base_dir)

        # 建立图片映射关系（uuid---image）
        self._build_image_mapping(image_elements)

        # 将图片信息映射到原始文本中
        final_text = self._replace_uuid_with_image_content(raw_text_with_uuid)
        # 构建文本的Document
        text_document = Document(
            page_content=final_text,
            metadata={
                "source": str(file_path),
                "type": "text",
                "file_name": file_path.name,
                "total_images": len(image_elements),
                "text_len": len(final_text),
            }
        )
        documents.append(text_document)

        # 构建图片的Document
        image_documents = self._build_image_documents(image_elements, base_dir)
        documents.extend(image_documents)

        return documents

    def _extract_image_elements(self, original_content: str, base_dir: Path) -> list[MarkdownElement]:
        """提取Markdown中的图片信息"""
        image_elements = []

        for match in re.finditer(self.PATTERNS['image'], original_content):
            alt_text, image_url = match.groups()
            # 处理图片
            image_element = self._process_image(alt_text, image_url, base_dir)
            if image_element:
                image_elements.append(image_element)

        return image_elements

    def _process_image(self, alt_text: str, img_path: str, base_dir: Path) -> MarkdownElement | None:
        """处理图片：OCR / 视觉模型识别"""

        try:
            content = ""
            process_type = ""
            image_bytes = ImageUtil.load_image(img_path, base_dir)
            if not image_bytes:
                return None

            # 视觉模型/OCR识别
            if self.use_vision and self.vision_llm:
                content = OcrUtil.vision_ocr(vision_llm=self.vision_llm, image_bytes=image_bytes)
                process_type = self.vision_llm.model_name
            elif self.use_ocr:
                content = OcrUtil.tesseract_ocr(image_bytes)
                process_type = "tesseracts OCR"
            else:
                return None

        except Exception as e:
            print(f"处理图片失败 {img_path}: {e}")
            return None

        element = MarkdownElement(
            type=ElementType.IMAGE,
            content=content,
            metadata={
                'source': img_path,
                'alt_text': alt_text,
                'raw': f"![{alt_text}]({img_path})",
                'base_dir': str(base_dir),
                'ocr_': {process_type},
            }
        )
        return element

    def _build_image_mapping(self, image_elements: list[MarkdownElement]):
        """
        构建图片映射：
        - Key：Loader生成的UUID(与读取的是匹配的)
        - Value：图片OCR/视觉模型内容 + 元数据
        """
        for image_element in image_elements:
            self.image_mapping[image_element.metadata['alt_text']] = {
                'content': image_element.content,
                'metadata': image_element.metadata
            }

    def _replace_uuid_with_image_content(self, raw_text_with_uuid: str) -> str:
        """替换LangChain Loader文本中的图片UUID为OCR/视觉模型内容"""
        # 匹配Loader生成的图片UUID占位符（格式：xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx）
        uuid_pattern = r'([0-9a-f\-]{36})'

        # 替换uuid的方法
        def replace_uuid(match):
            uuid_str = match.group(1)
            # 精准匹配UUID
            if self.image_mapping:
                dict_ = self.image_mapping[uuid_str]
                img_content = dict_.get('content')
                img_path = dict_.get('metadata')['image_path']
                return f"【图片内容】：\n{img_content}\n{img_path}"  # 明确标记图片语义
            return f"【图片解析失败】：UUID_{uuid_str}"

        final_text = re.sub(uuid_pattern, replace_uuid, raw_text_with_uuid)
        return final_text

    def _build_image_documents(self, image_elements: list[MarkdownElement], base_dir:  Path) -> list[Document]:
        """构建images的Document列表"""
        image_documents = []
        for image_element in image_elements:
            image_document = Document(
                page_content=image_element.content,
                metadata={
                    "type": "image",
                    **image_element.metadata,
                }
            )
            image_documents.append(image_document)

        return image_documents


if __name__ == '__main__':
    # 初始化解析器
    parser = MultimodalMarkdownParser(vision_llm=qwen_vision, use_ocr=True, use_vision=True)

    # 解析文件：返回Document列表
    documents = parser.parse(r"C:\Users\ASUS\Desktop\makedown\deepAgent.md")

    # 分离文本Document和图片Documents
    text_documents = [doc for doc in documents if doc.metadata.get("type") == "markdown_text"]
    image_documents = [doc for doc in documents if doc.metadata.get("type") == "markdown_image"]

    text_doc = text_documents[0] if text_documents else None
    image_docs = image_documents

    # 打印结果
    print("=" * 50)
    if text_doc:
        print(f"✅ 完整文本Document（语义完整）：")
        print(f"文本长度：{len(text_doc.page_content)} 字符")
        print(f"文本预览：{text_doc.page_content[:500]}...")
        print(f"文本元数据：{text_doc.metadata}")

    print("\n" + "=" * 50)
    print(f"✅ 图片Document列表（共{len(image_docs)}张）：")
    for i, img_doc in enumerate(image_docs):
        print(f"\n--- 图片{i + 1} ---")
        print(f"图片路径：{img_doc.metadata['image_path']}")
        print(f"图片内容预览：{img_doc.page_content[:200]}...")
        print(f"图片元数据：{img_doc.metadata}")
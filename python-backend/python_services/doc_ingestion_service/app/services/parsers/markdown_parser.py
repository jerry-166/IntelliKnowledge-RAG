"""
Markdown解析器
用手解析吗...
"""
from langchain_core.documents import Document

from basic_core.llm_factory import qwen_vision
from python_services.doc_ingestion_service.app.services.parsers.base_parser import base_parser
from python_services.doc_ingestion_service.app.services.utils.image_util import ImageUtil
from python_services.doc_ingestion_service.app.services.utils.ocr_util import OcrUtil
import re
import os
import base64
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from enum import Enum

import PIL.Image as PIL_Image
from io import BytesIO


class ElementType(Enum):
    """元素类型枚举"""
    TEXT = "text"
    HEADING = "heading"
    IMAGE = "image"
    CODE = "code"
    BLOCKQUOTE = "blockquote"


@dataclass
class MarkdownElement:
    """解析后的元素基类"""
    type: ElementType
    content: Any
    metadata: dict = field(default_factory=dict)


class MultimodalMarkdownParser(base_parser):
    """多模态Markdown解析器"""

    # 正则表达式模式
    PATTERNS = {
        'image': r'!\[([^\]]*)\]\(([^)]+)\)',  # ![alt](url)
        'link': r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',  # [text](url)
        'table': r'(\|[^\n]+\|(?:\n\|[-:| ]+\|)?(?:\n\|[^\n]+\|)*)',
        'code_block': r'```(\w*)\n([\s\S]*?)```',
        'heading': r'^(#{1,6})\s+(.+)$',
        'blockquote': r'^>\s+(.+)$',
        'list_item': r'^[\s]*[-*+]\s+(.+)$',
    }

    def __init__(self, vision_llm, use_ocr: bool = True, use_vision: bool = True):
        super().__init__("Markdown解析器", "Markdown")
        self.vision_llm = vision_llm
        self.use_ocr = use_ocr
        self.use_vision = use_vision

    def parse(self, file_path: str) -> List[Document]:
        """解析Markdown文件，返回langchain中的Document对象"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        base_dir = file_path.parent
        # 解析文本、图片内容（剔除一些字符、图片语义生成）
        elements = self._parse_content(content, base_dir)

        return self._elements_to_documents(elements)

    def _parse_content(self, content: str, base_dir: Path) -> List[MarkdownElement]:
        """解析Markdown内容"""
        elements = []

        code_blocks = {}
        def replace_code(match):
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks[placeholder] = (match.group(1), match.group(2))
            return placeholder

        # todo?
        content = re.sub(self.PATTERNS['code_block'], replace_code, content)

        # 按行处理
        lines = content.split('\n')
        current_text = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # 检查是否是代码块占位符
            if line.strip() in code_blocks:
                # 保存之前的文本
                if current_text:
                    elements.append(self._create_text_element('\n'.join(current_text)))
                    current_text = []

                lang, code = code_blocks[line.strip()]
                elements.append(MarkdownElement(
                    type=ElementType.CODE,
                    content=code,
                    raw=f"```{lang}\n{code}```",
                    metadata={'language': lang}
                ))
                i += 1
                continue

            # 检查图片
            img_match = re.search(self.PATTERNS['image'], line)
            if img_match:
                if current_text:
                    elements.append(self._create_text_element('\n'.join(current_text)))
                    current_text = []

                alt_text, img_path = img_match.groups()
                img_element = self._process_image(alt_text, img_path, base_dir)
                elements.append(img_element)

                # 处理图片后面可能的文本
                remaining = line[img_match.end():]
                if remaining.strip():
                    current_text.append(remaining)
                i += 1
                continue

            # 检查标题
            heading_match = re.match(self.PATTERNS['heading'], line)
            if heading_match:
                if current_text:
                    elements.append(self._create_text_element('\n'.join(current_text)))
                    current_text = []

                level = len(heading_match.group(1))
                text = heading_match.group(2)
                elements.append(MarkdownElement(
                    type=ElementType.HEADING,
                    content=text,
                    raw=line,
                    metadata={'level': level}
                ))
                i += 1
                continue

            # 普通文本
            current_text.append(line)
            i += 1

        return elements

    def _process_image(self, alt_text: str, img_path: str, base_dir: Path) -> MarkdownElement | None:
        """处理图片：OCR / 视觉模型识别"""
        element = MarkdownElement(
            type=ElementType.IMAGE,
            content="",
            metadata={
                'image_path': img_path,
                'raw': f"![{alt_text}]({img_path})"
            }
        )
        # 加载图片
        try:
            image_bytes = ImageUtil.load_image(img_path, base_dir)

            if not image_bytes:
                return None

            # 视觉模型识别（默认使用ocr识别）
            if self.use_vision and self.vision_llm:
                element.content = OcrUtil.vision_ocr(vision_llm=self.vision_llm, image_bytes=image_bytes)
            elif self.use_ocr:
                element.content = OcrUtil.tesseract_ocr(image_bytes)
            else:
                return None

        except Exception as e:
            print(f"处理图片失败 {img_path}: {e}")

        return element

    def _elements_to_documents(self, elements: List[MarkdownElement]) -> List[Document]:
        """转化为langchain的Document"""
        documents: List[Document] = []
        for element in elements:
            document = Document(
                page_content=element.content,
                metadata={
                    "type": element.type.value,
                    **element.metadata,
                },
            )
            documents.append(document)

        return documents


# ============ 使用示例 ============

if __name__ == '__main__':
    # 初始化解析器
    parser = MultimodalMarkdownParser(vision_llm=qwen_vision)

    # 解析文件
    documents = parser.parse(r"C:\Users\ASUS\Desktop\makedown\deepAgent.md")

    # 遍历 Documents 列表
    print(f"共解析出 {len(documents)} 个文档块")
    for i, doc in enumerate(documents):
        print(f"\n--- 文档 {i + 1} ---")
        print(f"类型: {doc.metadata.get('content_type')}")
        print(f"内容预览: {doc.page_content[:200]}...")

"""
多模态 Markdown 解析器
支持：文本、图片(OCR+视觉模型)
todo:表格！链接(怎么替换到相应位置？？？)
我认为没有必要去单独获取：代码块，所以去除了相应的代码
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader  # LangChain Markdown加载器
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from basic_core.llm_factory import qwen_vision
from python_services.doc_ingestion_service.app.services.parsers.base_parser import base_parser
from python_services.doc_ingestion_service.app.services.utils.image_util import ImageUtil
from python_services.doc_ingestion_service.app.services.utils.ocr_util import OcrUtil

import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any
from enum import Enum


class ElementType(Enum):
    """元素类型枚举"""
    TEXT = "text"
    IMAGE = "image"


@dataclass
class MarkdownElement:
    """解析后的元素基类"""
    type: ElementType
    content: Any
    metadata: dict = field(default_factory=dict)


class MultimodalMarkdownParser(base_parser):
    """多模态Markdown解析器（整合LangChain Loader + 自定义图片解析）"""

    # 正则表达式模式
    PATTERNS = {
        'image': r'!\[([^\]]*)\]\(([^)]+)\)',  # ![alt](url)
        'table': r'\|(.*)\|',
        'link': r'\[([^\]]+)\]\(([^)]+)\)',
    }

    def __init__(self, vision_llm, use_ocr: bool = True, use_vision: bool = True):
        super().__init__("Markdown解析器", "Markdown")
        self.vision_llm = vision_llm
        self.use_ocr = use_ocr
        self.use_vision = use_vision
        # 存储「图片路径/UUID」->「图片信息（OCR/路径/语义）」的映射
        self.image_mapping: dict = {}

    def parse(self, file_path: str) -> tuple[Document, List[Document]]:
        """
        解析Markdown文件，返回：
        - 完整文本Document（替换图片UUID为语义文本）
        - 图片Document列表（每张图片1个）
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 1：用LangChain Loader加载文本（会生成图片UUID）
        loader = UnstructuredMarkdownLoader(str(file_path))
        loader_docs = loader.load()  # 加载出1个完整文本Document（图片被替换为UUID）
        raw_text_with_uuid = loader_docs[0].page_content  # 带UUID的原始文本

        # 2：提取图片路径+处理OCR/视觉模型
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        base_dir = file_path.parent
        image_elements = self._extract_image_elements(original_content, base_dir)  # 解析出元素（含图片）

        # 3：建立「UUID-图片信息」映射 + 替换文本中的UUID
        # 提取解析的图片元素，生成映射
        self._build_image_mapping(image_elements)
        # 替换Loader文本中的UUID为图片语义文本（保留完整语义）
        final_text = self._replace_uuid_with_image_content(raw_text_with_uuid)

        # ========== 步骤4：构建最终的文本Document + 图片Document列表 ==========
        # 1. 完整文本Document（替换了图片UUID，保留所有文本语义）
        text_document = Document(
            page_content=final_text,
            metadata={
                "source": str(file_path),
                "type": "markdown_text",
                "file_name": file_path.name,
                "total_images": len(self.image_mapping)
            }
        )

        # 2. 图片Document列表（每张图片1个）
        image_documents = self._build_image_documents(image_elements, base_dir)

        return text_document, image_documents

    def _extract_image_elements(self, content: str, base_dir: Path) -> List[MarkdownElement]:
        """提取Markdown图片信息"""
        elements = []

        # 匹配所有的图片
        for match in re.finditer(self.PATTERNS['image'], content):
            alt_text, img_path = match.groups()
            element = self._process_image(alt_text, img_path, base_dir)
            if element:
                elements.append(element)

        return elements

    def _process_image(self, alt_text: str, img_path: str, base_dir: Path) -> MarkdownElement | None:
        """处理图片：OCR / 视觉模型识别"""
        element = MarkdownElement(
            type=ElementType.IMAGE,
            content="",
            metadata={
                'image_path': img_path,
                'alt_text': alt_text,
                'raw': f"![{alt_text}]({img_path})",
                'base_dir': str(base_dir)
            }
        )
        try:
            image_bytes = ImageUtil.load_image(img_path, base_dir)
            if not image_bytes:
                return None

            # 视觉模型/OCR识别
            if self.use_vision and self.vision_llm:
                element.content = OcrUtil.vision_ocr(vision_llm=self.vision_llm, image_bytes=image_bytes)
            elif self.use_ocr:
                element.content = OcrUtil.tesseract_ocr(image_bytes)
            else:
                return None

        except Exception as e:
            print(f"处理图片失败 {img_path}: {e}")
            return None

        return element

    def _build_image_mapping(self, image_elements: List[MarkdownElement]) -> None:
        """
        构建图片映射：
        - Key：Loader生成的UUID(与读取的是匹配的)
        - Value：图片OCR/视觉模型内容 + 元数据
        """
        for elem in image_elements:
            self.image_mapping[elem.metadata['alt_text']] = {
                'content': elem.content,
                'metadata': elem.metadata
            }

    def _replace_uuid_with_image_content(self, text_with_uuid: str) -> str:
        """
        替换LangChain Loader文本中的图片UUID为OCR/视觉模型内容
        原理：Loader会把![alt](path)替换为类似「<IMAGE: uuid4>」的占位符，这里反向替换
        """
        # 匹配Loader生成的图片UUID占位符（格式：<IMAGE: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx>）
        uuid_pattern = r'<IMAGE: ([0-9a-f-]+)>'

        # 替换逻辑：如果有图片映射，替换为OCR内容；否则保留UUID
        def replace_match(match):
            uuid_str = match.group(1)
            # 注：如果需要精准匹配UUID和图片路径，可补充Loader的图片UUID生成逻辑
            # 这里简化：按顺序替换（因为解析顺序和Loader替换顺序一致）
            if self.image_mapping:
                img_path = list(self.image_mapping.keys())[0]
                img_content = self.image_mapping.pop(img_path)['content']
                return f"【图片内容】：{img_content}"  # 明确标记图片语义
            return f"【图片解析失败】：UUID_{uuid_str}"

        final_text = re.sub(uuid_pattern, replace_match, text_with_uuid)
        return final_text

    def _build_image_documents(self, elements: List[MarkdownElement], base_dir: Path) -> List[Document]:
        """
        构建图片Document列表（每张图片1个）
        """
        image_docs = []
        for elem in elements:
            if elem.type == ElementType.IMAGE:
                img_doc = Document(
                    page_content=elem.content,  # OCR/视觉模型内容
                    metadata={
                        "type": "markdown_image",
                        "image_path": elem.metadata['image_path'],
                        "absolute_path": str(Path(base_dir) / elem.metadata['image_path']),
                        "alt_text": elem.metadata['alt_text'],
                        "source_file": base_dir.name,
                        # 关联文本Document的UUID（可选）
                        "text_document_uuid": str(uuid.uuid4())
                    }
                )
                image_docs.append(img_doc)
        return image_docs

    def split_text_document(self, text_doc: Document, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """
        备用：切分文本Document（保留语义的前提下）
        使用LangChain的Markdown切分器，按标题/段落切分
        """
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents([text_doc])
        # 补充元数据
        for doc in split_docs:
            doc.metadata.update({
                "split_chunk": True,
                "chunk_size": chunk_size,
                "original_source": text_doc.metadata['source']
            })
        return split_docs


# ============ 使用示例 ============
if __name__ == '__main__':
    # 初始化解析器
    parser = MultimodalMarkdownParser(vision_llm=qwen_vision, use_ocr=True, use_vision=True)

    # 解析文件：返回「1个完整文本Document」+「图片Document列表」
    text_doc, image_docs = parser.parse(r"C:\Users\ASUS\Desktop\makedown\deepAgent.md")

    # 打印结果
    print("=" * 50)
    print(f"✅ 完整文本Document（语义完整）：")
    print(f"文本长度：{len(text_doc.page_content)} 字符")
    print(f"文本预览：{text_doc.page_content[:500]}...")
    print(f"文本元数据：{text_doc.metadata}")

    print("\n" + "=" * 50)
    print(f"✅ 图片Document列表（共{len(image_docs)}张）：")
    for i, img_doc in enumerate(image_docs):
        print(f"\n--- 图片{i+1} ---")
        print(f"图片路径：{img_doc.metadata['image_path']}")
        print(f"图片内容预览：{img_doc.page_content[:200]}...")
        print(f"图片元数据：{img_doc.metadata}")

    # （可选）如果后续需要切分文本Document
    split_text_docs = parser.split_text_document(text_doc, chunk_size=1000, chunk_overlap=100)
    print(f"\n✅ 切分后文本Document数量：{len(split_text_docs)}")

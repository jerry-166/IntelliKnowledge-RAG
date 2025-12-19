"""
pdf解析器（实现parser基类）
todo: 普通的pdf对于视觉模型和ocr的选择
todo: 处理的图片大小
todo: metadata的处理（page等...）
todo: 检查循环逻辑正确与否，是否有重复处理page中的图片操作
"""
import base64
import datetime
import io
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List

import fitz
from PIL import Image
from zoneinfo import ZoneInfo
from langchain_core.documents import Document

from basic_core.llm_factory import qwen_vision
from python_services.core.parser_element_type import ElementType
from python_services.parsers.base_parser import BaseParser
from python_services.utils.ocr_util import OcrUtil


@dataclass
class PDFElement:
    """PDF中的单个元素"""
    type: ElementType
    content: Any
    page_num: int
    bbox: tuple
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFParser(BaseParser):
    """
    复杂PDF处理器 - 处理多种元素类型

    处理策略：
    1. 结构化提取：使用PyMuPDF提取
    2. 视觉理解：对复杂页面使用多模态LLM理解
    3. OCR增强：对扫描PDF使用OCR
    """

    def __init__(
            self,
            vision_llm=None,
            use_ocr: bool = True,
            extract_images: bool = True,
            extract_tables: bool = True,
            min_image_size: int = 10,
    ):
        super().__init__("pdf解析器", ["pdf"])
        self.vision_llm = vision_llm
        self.use_ocr = use_ocr
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.min_image_size = min_image_size

    def parse_impl(self, file_path_or_url: str, is_scanned: bool = False) -> List[Document]:
        """
        解析pdf文件，返回langchain的Document列表
        """
        pdf_path = Path(file_path_or_url)
        if not pdf_path.exists():
            raise FileNotFoundError(f"文件不存在：{pdf_path}")

        docs = fitz.open(pdf_path)  # Document()
        total_pages = len(docs)

        # 结构列表
        page_elements_map: Dict[int, List[PDFElement]] = {}

        for page_num in range(total_pages):
            page = docs[page_num]
            page_elements: List[PDFElement] = []

            if is_scanned:
                # 扫描件：直接使用视觉LLM描述就行（大多扫描件是文和图一起）
                scanned_page_elements = self._scanned_page_ocr(page, page_num)
                if scanned_page_elements:
                    print(f"处理扫描件成功")
                    page_elements.append(scanned_page_elements)
            else:
                # 普通页面：结构化提取
                # 1. 提取文本块
                text_elements = self._extract_texts(page, page_num)
                if text_elements:
                    print(f"提取PDF文本成功")
                    page_elements.extend(text_elements)

                # 2. 提取图片
                if self.extract_images:
                    image_elements = self._extract_images(page, page_num, docs)
                    if image_elements:
                        print(f"提取PDF图片成功")
                        page_elements.extend(image_elements)

                # 3. 提取表格
                if self.extract_tables:
                    table_elements = self._extract_tables(page, page_num)
                    if table_elements:
                        print(f"提取PDF表格成功")
                        page_elements.extend(table_elements)

                # 4. 提取链接
                link_elements = self._extract_links(page, page_num)
                if link_elements:
                    print(f"提取PDF链接成功")
                    page_elements.extend(link_elements)

            page_elements_map[page_num] = page_elements

        docs.close()

        # 6. 转换为langchain的Document列表并返回
        return self._elements_to_documents(
            page_elements_map,
            file_path=str(pdf_path.absolute()),
            total_pages=total_pages,
        )

    def _scanned_page_ocr(self, page, page_num: int) -> PDFElement:
        """使用视觉模型/ocr识别扫描件"""
        ocr_method = ""
        ocr_result = ""
        if self.extract_images:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            if self.vision_llm:
                # 既需要提取图片，也传入了视觉LLM，则使用视觉LLM进行描述
                ocr_result = OcrUtil.vision_ocr(self.vision_llm, img_bytes, 'png')
                ocr_method = self.vision_llm.model_name
            elif not self.vision_llm:
                # 没有视觉LLM，判断是否有tesseract_ocr
                if self.use_ocr:
                    ocr_result = OcrUtil.tesseract_ocr(img_bytes)
                    ocr_method = "tesseract_ocr"

        return PDFElement(
                type=ElementType.IMAGE,
                content=ocr_result.strip(),
                page_num=page_num,
                bbox=(0, 0, page.rect.width, page.rect.height),
                metadata={
                    "source": "ocr_page",
                    "method": ocr_method,
                }
            )

    def _extract_texts(self, page, page_num: int) -> List[PDFElement]:
        """提取文本，保持结构"""
        elements = []

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # 文本块
                text_content = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_content += span["text"]
                    text_content += "\n"

                if text_content.strip():
                    # 检查是否是标题（基于字体大小）
                    is_header = self._detect_header(block)

                    elements.append(
                        PDFElement(
                            type=ElementType.TEXT,
                            content=text_content.strip(),
                            page_num=page_num,
                            bbox=tuple(block["bbox"]),
                            # todo 对于metadata
                            metadata={
                                # "font_size": self._get_dominant_font_size(block),
                                "is_header": is_header,
                            }
                        )
                    )

        return elements

    def _extract_images(self, page, page_num: int, doc) -> List[PDFElement]:
        """提取图片"""
        elements = []

        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # 检查图片大小(去除一些小型图表元素)
                if base_image["width"] < self.min_image_size or base_image["height"] < self.min_image_size:
                    continue

                # 获取图片位置
                bbox = self._get_image_bbox(page, xref)

                # 视觉LLM，提取图片描述 todo:ocr OR 视觉LLM
                """
                图片描述放在metadata中吗？如果检索不到该图片，那它的metadata不是就没用了?
                ===> 放入content中便于检索
                """
                image_description = ""
                if self.vision_llm:
                    image_description = self._describe_image(image_bytes, base_image["ext"])
                    if image_description:
                        print(f"视觉模型处理pdf图片，获得图片描述：{image_description[:100]}")

                elements.append(
                    PDFElement(
                        type=ElementType.IMAGE,
                        content=image_description,
                        page_num=page_num,
                        bbox=bbox,
                        metadata={
                            # "width": pil_image.width,
                            # "height": pil_image.height,
                            "format": base_image.get("ext", "unknown"),
                            "image_index": img_index,
                            "base64": base64.b64encode(image_bytes).decode()  # 用于在前端渲染图片
                        }
                    )
                )
            except Exception as e:
                print(f"提取图片失败：{e}")
                continue

        return elements

    def _extract_tables(self, page, page_num: int) -> List[PDFElement]:
        """提取表格,保证结构"""
        elements = []

        try:
            tables = page.find_tables()
            for table in tables:
                # 提取表格数据
                table_data = table.extract()
                if not table_data or not any(any(cell for cell in row) for row in table_data):
                    continue

                # 转化为markdown表格
                markdown_table = self._table_to_markdown(table_data)

                elements.append(
                    PDFElement(
                        type=ElementType.TABLE,
                        content=markdown_table,
                        page_num=page_num,
                        bbox=table.bbox,
                        metadata={
                            # "rows": len(table_data),
                            # "cols": len(table_data[0]) if table_data else 0,
                            "raw_data": table_data,
                        }
                    )
                )
        except Exception as e:
            print(f"提取表格失败(页{page_num + 1}): {e}")

        return elements

    def _extract_links(self, page, page_num: int) -> List[PDFElement]:
        """提取链接"""
        elements = []

        links = page.get_links()
        for link in links:
            if link.get("uri"):
                elements.append(
                    PDFElement(
                        type=ElementType.LINK,
                        content=link.get("uri"),
                        page_num=page_num,
                        bbox=tuple(link.get("from")) if "from" in link else (0, 0, 0, 0),
                        metadata={
                            "link_type": "external",
                            "text": self._get_link_text(page, link),
                        }
                    )
                )

        return elements

    def _detect_header(self, block: dict) -> bool:
        """检测文本块是否是标题"""
        font_size = self._get_dominant_font_size(block)
        text_len = sum(
            len(span.get("text", ""))
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )

        return font_size > 14 and text_len < 20

    def _get_dominant_font_size(self, block: dict) -> int:
        """获取文本块的主要字体大小"""
        sizes = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span["size"])

        return max(sizes) if sizes else 12

    def _describe_image(self, image_bytes: bytes, image_format: str = "png") -> str:
        """使用视觉模型模数图片"""
        if not self.vision_llm:
            return ""

        try:
            base64_image = base64.b64encode(image_bytes).decode()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请详细描述这张图片的内容，包含图表、图形、文字等所有信息。如果是图表，请提取其中的数据。不包含背景颜色，文字颜色，字体及大小等无关信息"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            response = self.vision_llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"调用视觉模型描述图片失败：{e}")
            return ""

    def _get_image_bbox(self, page, xref: int) -> tuple:
        """获取图片的边界框"""
        for item in page.get_images(full=True):
            # todo: 这里是干什么？
            if item[0] == xref:
                # 获取图片的边界框
                rects = page.get_image_rects(item)
                if rects:
                    r = rects[0]
                    return (r.x0, r.y0, r.x1, r.y1)
        return (0, 0, 0, 0)

    def _table_to_markdown(self, table_data: List[List]) -> str:
        """将表格数据转换为Markdown格式"""
        if not table_data:
            return ""

        lines = []

        # 表头
        header = table_data[0]
        lines.append("| " + " | ".join((str(cell or "") for cell in header)) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")

        # 数据行
        for row in table_data[1:]:
            lines.append("| " + " | ".join((str(cell or "") for cell in row)) + " |")

        return "\n".join(lines)

    def _get_link_text(self, page, link: str) -> str:
        """获取链接的文字"""
        if "from" in link:
            rect = fitz.Rect(link["from"])
            return page.get_text("text", clip=rect).strip()
        return ""

    def _elements_to_documents(
            self,
            page_elements_map: Dict[int, List[PDFElement]],
            file_path: str,
            total_pages: int
    ) -> List[Document]:
        """返回langchain的Document对象，以页为单位"""
        documents = []
        file_name = Path(file_path).name

        # 按页进行document创建
        for page_num in sorted(page_elements_map.keys()):
            elements = page_elements_map[page_num]

            if not elements:
                continue

            # 按位置元素排序元素
            sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))

            # 分离不同类型的元素
            text_parts: List[PDFElement] = []
            image_parts: List[PDFElement] = []
            for element in sorted_elements:
                if element.type == ElementType.IMAGE:
                    image_parts.append(element)
                else:
                    text_parts.append(element)

            # 创建页面文本的Document
            page_contents: List[str] = []
            for text_part in text_parts:
                if text_part.type == ElementType.TABLE:
                    page_contents.append(f"\n{text_part.content}\n")
                elif text_part.type == ElementType.LINK:
                    link_text = text_part.metadata.get("text", "")
                    if link_text:
                        page_contents.append(f"[{link_text}]({text_part.content})")

                elif text_part.type in (ElementType.TEXT, ElementType.HEADER):
                    # 标题可以添加markdown格式
                    if text_part.type == ElementType.HEADER:
                        page_contents.append(f"## {text_part.content}")
                    else:
                        page_contents.append(text_part.content)

            page_content = "\n".join(page_contents)
            documents.append(Document(
                page_content=page_content,
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "page_num": page_num + 1,
                    "total_pages": total_pages,
                    "type": "text",
                    "ext": "pdf",
                    "create_time": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
                }
            ))

            # 为每一个有描述的图片创建单独的Document
            for img_idx, image_part in enumerate(image_parts):
                if image_part.content:  # 有图片描述
                    documents.append(Document(
                        page_content=image_part.content,
                        metadata={
                            "source": file_path,
                            "file_name": file_name,
                            "page_num": page_num + 1,
                            "type": "image",
                            "ext": "pdf",
                            # "bbox": image_part.bbox,
                            "create_time": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                            **image_part.metadata,
                        }
                    ))

        return documents


if __name__ == '__main__':
    file0 = "C:/Users/ASUS/OneDrive/OneDrive 入门.pdf"
    file1 = "C:/Users/ASUS/Desktop/IntelliKnowledge-RAG/docs/ceshi-pdf.pdf"
    pdf_parser = PDFParser(vision_llm=qwen_vision)
    documents = pdf_parser.parse(file1)

    print(f"共解析出 {len(documents)} 个文档块")
    for i, doc in enumerate(documents):
        print(f"\n--- 文档 {i + 1} ---")
        print(f"页码: {doc.metadata.get('page_num')}/{doc.metadata.get('total_pages')}")
        print(f"类型: {doc.metadata.get('content_type')}")
        print(f"内容预览: {doc.page_content[:200]}...")

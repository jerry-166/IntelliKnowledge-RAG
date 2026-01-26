"""
pdf解析器（实现parser基类）
ok: 普通的pdf对于视觉模型和ocr的选择 --- 优先使用视觉模型
ok: 处理的图片大小 --- 没办法全考虑，只对小的图标跳过
todo: metadata的处理（page等...）
ok: 检查循环逻辑正确与否，是否有重复处理page中的图片操作 --- 没有吧，可能是一个是去构建md，一个去存图片向量库了
"""
import base64
import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

import fitz
from langchain_core.documents import Document
from python_services.core.parser_element_type import ElementType
from python_services.parsers.base_parser import BaseParser
from python_services.utils.image_util import ImageUtil
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

    def _save_page_image(self, page, path: str):
        """保存每一页图片"""
        # 确保目录存在
        dir_path = Path(path).parent
        dir_path.mkdir(parents=True, exist_ok=True)

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        bytes = pix.tobytes()
        with open(path, "wb") as f:
            f.write(bytes)

    def _save_visualized_page_image(self, page, path: str, elements: List[PDFElement]):
        """保存带有元素边框的可视化页面图片"""
        # 创建一个临时PDF文档用于绘制
        temp_doc = fitz.open()
        draw_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)

        # 复制原页面内容
        mat = fitz.Matrix(1, 1)  # 单位矩阵，不缩放
        pix = page.get_pixmap(matrix=mat)
        draw_page.insert_image(
            fitz.Rect(0, 0, draw_page.rect.width, draw_page.rect.height),
            pixmap=pix
        )

        # 为每个元素绘制边框
        for element in elements:
            # 根据元素类型设置不同的边框样式
            border_color = (0, 0, 0)  # 默认黑色
            border_width = 1.0

            # 不同类型元素使用不同颜色
            if element.type == ElementType.TEXT:
                border_color = (0, 0, 1)  # 蓝色 - 普通文本
            elif element.type == ElementType.HEADER:
                border_color = (1, 0, 0)  # 红色 - 标题
            elif element.type == ElementType.IMAGE:
                border_color = (0, 1, 0)  # 绿色 - 图片
            elif element.type == ElementType.TABLE:
                border_color = (1, 0, 1)  # 紫色 - 表格
            elif element.type == ElementType.LINK:
                border_color = (1, 0.5, 0)  # 橙色 - 链接

            # 绘制边框
            rect = fitz.Rect(*element.bbox)
            draw_page.draw_rect(rect, color=border_color, width=border_width)

        # 生成带有边框的图片
        self._save_page_image(draw_page, path)

        # 关闭临时文档
        temp_doc.close()

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
                # 1. 先提取表格，用于后续排除表格区域的文本提取
                table_elements = []
                excluded_areas = []
                if self.extract_tables:
                    table_elements = self._extract_tables(page, page_num)
                    if table_elements:
                        print(f"提取PDF表格成功")
                        # 收集表格边界框，用于排除文本提取
                        excluded_areas = [table.bbox for table in table_elements]

                # 2. 提取文本块，排除表格区域
                text_elements = self._extract_texts(page, page_num, excluded_areas)
                if text_elements:
                    print(f"提取PDF文本成功")
                    page_elements.extend(text_elements)

                # 3. 添加表格元素
                if table_elements:
                    page_elements.extend(table_elements)

                # 4. 提取图片
                if self.extract_images:
                    image_elements = self._extract_images(page, page_num, docs)
                    if image_elements:
                        print(f"提取PDF图片成功")
                        page_elements.extend(image_elements)

                # 5. 提取链接
                link_elements = self._extract_links(page, page_num)
                if link_elements:
                    print(f"提取PDF链接成功")
                    page_elements.extend(link_elements)

            # 保存每页原始图片
            original_path = Path("./output") / pdf_path.stem / "image" / "original" / f"{page_num}.png"
            self._save_page_image(page, str(original_path))

            # 保存带有元素边框的可视化页面图片
            if page_elements:  # 只对有提取元素的页面生成可视化
                augment_path = Path("./output") / pdf_path.stem / "image" / "augment" / f"{page_num}.png"
                self._save_visualized_page_image(page, str(augment_path), page_elements)

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
                ocr_result = OcrUtil.vision_ocr(self.vision_llm, img_bytes)
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

    def _extract_texts(self, page, page_num: int, excluded_areas: List[tuple] = None) -> List[PDFElement]:
        """提取文本，保持结构，排除指定区域"""
        if excluded_areas is None:
            excluded_areas = []

        elements = []

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # 文本块
                # 检查文本块是否与排除区域重叠
                block_bbox = block["bbox"]
                should_exclude = False

                for excl_bbox in excluded_areas:
                    if self._check_overlap(block_bbox, excl_bbox):
                        should_exclude = True
                        break

                if should_exclude:
                    continue

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
                            type=ElementType.TEXT if not is_header else ElementType.HEADER,
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

                # 视觉LLM，提取图片描述
                image_description = OcrUtil.describe_image(image_bytes, self.vision_llm)
                if image_description:
                    print(f"视觉模型处理pdf图片，获得图片描述：{image_description[:100]}")

                elements.append(
                    PDFElement(
                        type=ElementType.IMAGE,
                        content=image_description or "",
                        page_num=page_num,
                        bbox=bbox,
                        metadata={
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
                # 根据table_data的位置信息，删除text提取出的table

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

    def _check_overlap(self, bbox1: tuple, bbox2: tuple, threshold: float = 0.5) -> bool:
        """检查两个边界框是否重叠，超过阈值则认为重叠"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 计算交集面积
        left = max(x1_1, x1_2)
        top = max(y1_1, y1_2)  # 修复：使用正确的顶部坐标
        right = min(x2_1, x2_2)
        bottom = min(y2_1, y2_2)

        if left < right and top < bottom:
            intersection_area = (right - left) * (bottom - top)
            bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)

            # 如果交集面积占原区域面积的比例超过阈值，则认为重叠
            overlap_ratio = intersection_area / bbox1_area if bbox1_area > 0 else 0
            return overlap_ratio >= threshold

        return False

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
        file_stem = Path(file_path).stem  # 不带扩展名的文件名

        output_dir = Path(".") / "output" / file_stem / "image"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_md = Path(".") / "output" / file_stem
        output_md.mkdir(parents=True, exist_ok=True)

        md_lines = []
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

            # 提取并保存md
            inserted_image = set()  # 记录已经加入的图片
            for ele in sorted_elements:
                type = ele.type
                content = ele.content
                if type == ElementType.HEADER:
                    md_lines.append(f"# {content}\n")
                elif type == ElementType.IMAGE:
                    # 下载到本地，并保证不会重复
                    img_path = os.path.join(output_dir, f"page{page_num}_img{ele.metadata['image_index']}.png")
                    if img_path not in inserted_image:
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(ele.metadata["base64"]))
                            img_bytes = ImageUtil.base64_to_bytes(ele.metadata["base64"])
                            inserted_image.add(img_path)
                            md_lines.append(f"![Image]({img_path})\n")
                            image_content = OcrUtil.describe_image(img_bytes, self.vision_llm)
                        md_lines.append("." + image_content)  # 加入图片描述
                else:
                    md_lines.append(content + "\n")

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

        # 保存到md中
        with open(os.path.join(output_md, "output.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"处理pdf文档{file_name}，保存图片到{output_dir}，保存处理后的md文档到{output_md}")
        return documents


if __name__ == '__main__':
    file0 = r"C:\Users\ASUS\Desktop\IntelliKnowledge-RAG\python_backend\a_other_rag\unstructured_fitz_multi\MCP实战课件【合集】.pdf"
    file1 = "C:/Users/ASUS/Desktop/IntelliKnowledge-RAG/docs/ceshi-pdf.pdf"
    pdf_parser = PDFParser()  # vision_llm=qwen_vision
    documents = pdf_parser.parse(file1)

    print(f"共解析出 {len(documents)} 个文档块")
    for i, doc in enumerate(documents):
        print(f"\n--- 文档 {i + 1} ---")
        print(f"页码: {doc.metadata.get('page_num')}/{doc.metadata.get('total_pages')}")
        print(f"类型: {doc.metadata.get('content_type')}")
        print(f"内容预览: {doc.page_content[:200]}...")

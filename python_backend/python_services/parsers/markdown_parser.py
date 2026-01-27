"""
markdown解析器
解析图片、文字
todo: 表格、链接怎么换回去，可以记录位置吗？
todo: Markdown格式保留处理？标题、正文...
"""
import datetime
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from langchain_core.documents import Document
from python_services.core.parser_element_type import ElementType
from python_services.parsers.base_parser import BaseParser
from python_services.utils.image_util import ImageUtil
from python_services.utils.ocr_util import OcrUtil


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

    def __init__(
            self,
            vision_llm=None,
            max_workers: int = 4
    ):
        super().__init__("Markdown解析器", ["md"], max_workers)
        self.vision_llm = vision_llm
        # 存储图片uuid和图片内容的映射关系
        self.image_mapping: dict = {}

    def parse_impl(self, file_path_or_url: str) -> list[Document]:
        """解析Markdown的主要函数"""
        documents = []

        # 判断文件是否存在
        file_path = Path(file_path_or_url)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 直接读取原始内容
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # 提取所有元素
        image_elements = self._extract_image_elements(original_content, file_path)

        # 清理并处理原始内容：保留结构，去除多余空白符号
        cleaned_content = self._clean_markdown_content(original_content, image_elements)

        # 构建文本的Document
        text_document = Document(
            page_content=cleaned_content,
            metadata={
                "source": str(file_path),
                "type": "text",
                "file_name": file_path.name,
                "ext": "md",
                'create_time': datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "total_images": len(image_elements)
            }
        )
        documents.append(text_document)

        # 构建图片的Document
        image_documents = self._build_image_documents(image_elements)
        documents.extend(image_documents)

        # 下载原来的Md文件
        path = Path("./output") / f"{Path(file_path_or_url).stem}" / "original.md"
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(original_content)  # 保存原本的内容（通过读取）
        # 下载加强的Md文件
        path = Path("./output") / f"{Path(file_path_or_url).stem}" / "augment.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)  # 保存清理后的内容
        return documents

    def _clean_markdown_content(self, content: str, image_elements: list[MarkdownElement]) -> str:
        """清理Markdown内容：保留结构，去除多余空白符号
        
        Args:
            content: 原始Markdown内容
            image_elements: 提取到的图片元素列表
            
        Returns:
            清理后的Markdown内容
        """

        # 1. 处理图片：将图片替换为OCR/视觉模型内容
        def replace_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            # 查找对应的图片元素
            for img_ele in image_elements:
                if img_ele.metadata['source'] == img_path:
                    return f"\n【图片内容】：\n{img_ele.content}\n【图片路径】：{img_path}\n"
            return match.group(0)  # 如果没找到，保留原样

        content = re.sub(self.PATTERNS['image'], replace_image, content)

        # 2. 清理多余的空白符号
        lines = []
        for line in content.split('\n'):
            # 去除行首行尾的空白
            cleaned_line = line.strip()
            # 去除行内多余的空格和tab（连续的空白替换为单个空格）
            cleaned_line = re.sub(r"\s+", ' ', cleaned_line)
            # 保留表格、标题、链接的结构
            lines.append(cleaned_line)

        # 3. 合并连续的空行（最多保留一个空行）
        cleaned_content = ''
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned_content += '\n'
                    prev_empty = True
            else:
                cleaned_content += line + '\n'
                prev_empty = False

        return cleaned_content.strip()

    def _extract_image_elements(self, original_content: str, path: Path) -> list[MarkdownElement]:
        """提取Markdown中的图片信息"""
        image_elements = []

        for match in re.finditer(self.PATTERNS['image'], original_content):
            alt_text, image_url = match.groups()
            # 处理图片
            image_element = self._process_image(alt_text, image_url, path)
            if image_element:
                image_elements.append(image_element)

        return image_elements

    def _process_image(self, alt_text: str, img_path: str, path: Path) -> MarkdownElement | None:
        """处理图片：OCR / 视觉模型识别"""
        try:
            image = ImageUtil.load_image(img_path, path.parent)
            if not image:
                print(f"图片加载失败，返回None")
                return None
            # 视觉模型/OCR识别
            img_bytes = ImageUtil.image_to_bytes(image=image)[0]
            content = OcrUtil.describe_image(img_bytes, self.vision_llm)
            # 如果所有方法都失败了
            if not content:
                print(f"所有OCR方法都失败了，返回None")
                return None
            # 处理图片内容(防止代码注释的‘#’变成md的标题格式)
            # 在行首的 # 前添加点，防止被识别为Markdown标题
            content = re.sub(r'^(#+)', r'\1.', content, flags=re.MULTILINE)

            # 保存图片到本地
            try:
                save_dir = Path("./output") / f"{path.stem}" / "images"
                # 创建目录（exist_ok=True确保目录存在）
                save_dir.mkdir(parents=True, exist_ok=True)
                # 生成唯一文件名
                save_path = save_dir / f"{path.stem}_{uuid.uuid4()}.png"
                # 保存图片
                with open(save_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"图片已保存到: {save_path}")
            except Exception as save_e:
                print(f"保存图片失败: {save_e}")
        except Exception as e:
            print(f"处理图片失败 {img_path}: {e}")
            return None

        print(f"处理MD图片成功，{content[:100]}")
        element = MarkdownElement(
            type=ElementType.IMAGE,
            content=content,
            metadata={
                'type': 'image',
                'source': img_path,
                'alt_text': alt_text,
                # 'raw': f"![{alt_text}]({img_path})",
                # 'base_dir': str(base_dir),
                'ext': "md",
            }
        )
        return element

    """
        def _build_image_mapping(self, image_elements: list[MarkdownElement]):
            构建图片映射：
            - Key：Loader生成的UUID(与读取的是匹配的)
            - Value：图片OCR/视觉模型内容 + 元数据
            
            for image_element in image_elements:
                self.image_mapping[image_element.metadata['alt_text']] = {
                    'content': image_element.content,
                    'metadata': image_element.metadata
                }
        
        def _replace_uuid_with_image_content(self, raw_text_with_uuid: str) -> str:
            # 替换LangChain Loader文本中的图片UUID为OCR/视觉模型内容
            # 匹配Loader生成的图片UUID占位符（格式：xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx）
            uuid_pattern = r'([0-9a-f\-]{36})'
    
            # 替换uuid的方法
            def replace_uuid(match):
                uuid_str = match.group(1)
                # 精准匹配UUID
                if self.image_mapping:
                    dict_ = self.image_mapping[uuid_str]
                    img_content = dict_.get('content')
                    img_path = dict_.get('metadata')['source']
                    return f"【图片内容】：\n{img_content}\n{img_path}"  # 明确标记图片语义
                return f"【图片解析失败】：UUID_{uuid_str}"
    
            final_text = re.sub(uuid_pattern, replace_uuid, raw_text_with_uuid)
            return final_text
    """

    def _build_image_documents(self, image_elements: list[MarkdownElement]) -> list[Document]:
        """构建images的Document列表"""
        image_documents = []
        for image_element in image_elements:
            image_document = Document(
                page_content=image_element.content,
                metadata={
                    # "type": "image",
                    **image_element.metadata,
                    "create_time": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
                }
            )
            image_documents.append(image_document)

        return image_documents


if __name__ == '__main__':
    # 初始化解析器
    parser = MultimodalMarkdownParser()  # vision_llm=qwen_vision)

    # 解析文件：返回Document列表
    documents = parser.parse(r"C:\Users\ASUS\Desktop\makedown\deepAgent.md")

    # 分离文本Document和图片Documents
    text_documents = [doc for doc in documents if doc.metadata.get("type") == "text"]
    image_documents = [doc for doc in documents if doc.metadata.get("type") == "image"]

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
        print(f"图片路径：{img_doc.metadata['source']}")
        print(f"图片内容预览：{img_doc.page_content[:200]}...")
        print(f"图片元数据：{img_doc.metadata}")

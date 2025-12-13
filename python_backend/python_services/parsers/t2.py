# services/parsers/base_parser.py
"""
解析器基类 - 增强版
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ElementType(Enum):
    """元素类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    LINK = "link"
    HEADER = "header"
    CODE = "code"
    LIST = "list"


@dataclass
class ParsedElement:
    """解析元素"""
    type: ElementType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 位置信息
    page_num: Optional[int] = None
    bbox: Optional[tuple] = None

    # 原始信息
    raw: Optional[str] = None


@dataclass
class ParseResult:
    """解析结果"""
    elements: List[ParsedElement]
    metadata: Dict[str, Any]

    # 统计信息
    text_count: int = 0
    image_count: int = 0
    table_count: int = 0
    parse_time_ms: float = 0.0

    def to_documents(self) -> List[Document]:
        """转换为Document列表"""
        documents = []

        for element in self.elements:
            doc = Document(
                page_content=element.content,
                metadata={
                    "type": element.type.value,
                    "page_num": element.page_num,
                    "bbox": element.bbox,
                    **element.metadata,
                    **self.metadata,
                }
            )
            documents.append(doc)

        return documents


class BaseParser(ABC):
    """
    解析器基类

    Features:
    - 统一接口
    - 异常处理
    - 性能监控
    - 并发支持
    """

    def __init__(
            self,
            name: str,
            supported_formats: List[str],
            max_workers: int = 4,
    ):
        self.name = name
        self.supported_formats = [fmt.lower() for fmt in supported_formats]
        self.max_workers = max_workers

        # 统计信息
        self._parse_count = 0
        self._total_time_ms = 0.0
        self._error_count = 0

    def parse(self, file_path: str, **kwargs) -> List[Document]:
        """
        解析文件

        Args:
            file_path: 文件路径
            **kwargs: 额外参数

        Returns:
            Document列表
        """
        path = Path(file_path)

        # 验证文件
        self._validate_file(path)

        start_time = time.perf_counter()

        try:
            # 执行解析
            result = self._parse_impl(path, **kwargs)

            # 记录统计
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result.parse_time_ms = elapsed_ms
            self._parse_count += 1
            self._total_time_ms += elapsed_ms

            logger.info(
                f"[{self.name}] Parsed {path.name}: "
                f"{result.text_count} texts, {result.image_count} images, "
                f"{result.table_count} tables in {elapsed_ms:.1f}ms"
            )

            return result.to_documents()

        except Exception as e:
            self._error_count += 1
            logger.error(f"[{self.name}] Parse failed for {path.name}: {e}")
            raise

    def _validate_file(self, path: Path):
        """验证文件"""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower().lstrip('.')
        if suffix not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {self.supported_formats}"
            )

    @abstractmethod
    def _parse_impl(self, path: Path, **kwargs) -> ParseResult:
        """解析实现 (子类实现)"""
        pass

    def parse_batch(
            self,
            file_paths: List[str],
            **kwargs
    ) -> Dict[str, List[Document]]:
        """
        批量解析

        Args:
            file_paths: 文件路径列表

        Returns:
            {文件路径: Document列表}
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.parse, fp, **kwargs): fp
                for fp in file_paths
            }

            for future in as_completed(futures):
                fp = futures[future]
                try:
                    docs = future.result()
                    results[fp] = docs
                except Exception as e:
                    logger.error(f"Batch parse failed for {fp}: {e}")
                    results[fp] = []

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "supported_formats": self.supported_formats,
            "parse_count": self._parse_count,
            "error_count": self._error_count,
            "avg_parse_time_ms": (
                self._total_time_ms / self._parse_count
                if self._parse_count > 0 else 0
            ),
        }
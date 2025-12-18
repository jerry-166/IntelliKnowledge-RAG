"""
parser的基类
"""
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Any, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    解析器基类

    Features:
    - 统一接口
    - 异常处理
    - 性能监控
    - 并发支持
    """

    def __init__(self, name, file_type: List[str], max_workers: int = 4):
        self.name = name
        self.file_type = file_type
        self.max_workers = max_workers

        # 统计信息
        self._parse_count = 0
        self._total_time_ms = 0.0
        self._error_count = 0

    def parse(self, path: str) -> list[Document]:
        """解析文件"""
        # 检查文件
        self._check_path(path)

        # 计时
        start = time.perf_counter()

        try:
            # 开始解析
            documents = self.parse_impl(file_path_or_url=path)

            # 统计信息
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._parse_count += 1
            self._total_time_ms += elapsed_ms

            logger.info(
                f"✅️[{self.name}] Parsed {path}: "
                f"{len(documents)} documents in {elapsed_ms:.2f}ms"
            )
            return documents
        except Exception as e:
            self._error_count += 1
            logger.error(f"❌️Parse failed for {path}: {e}")
            raise e

    def _check_path(self, path: str) -> bool:
        """检查文件"""
        if path.startswith(("http://", "https://")):
            return True

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower().lstrip('.')
        if suffix not in self.file_type:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {self.file_type}"
            )
        return True

    @abstractmethod
    def parse_impl(self, file_path_or_url: str):
        """解析过程的实现"""
        pass

    def parse_batch(
            self,
            file_list: list[str],
    ) -> Dict[str, list[Document]]:
        # 批量解析方法
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.parse, file): file
                for file in file_list
            }

            for future in as_completed(futures):
                file = futures[future]
                try:
                    docs = future.result()
                    results[file] = docs
                except Exception as e:
                    print(f"Batch parse failed for {file}: {e}")
                    results[file] = []

        return results

    def get_stats(self) -> Dict[str, Any]:
        """统计解析信息"""
        return {
            "name": self.name,
            "file_type": self.file_type,
            "parse_count": self._parse_count,
            "total_time_ms": self._total_time_ms,
            "error_count": self._error_count,
            "avg_parse_time_ms": (
                self._total_time_ms / self._parse_count
                if self._parse_count > 0 else 0
            )
        }

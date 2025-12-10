"""
parser的基类
"""
from abc import ABC, abstractmethod


class BaseParser(ABC):
    def __init__(self, name, file_type):
        self.name = name
        self.file_type = file_type

    @abstractmethod
    def parse(self, file: str):
        # 解析方法（子类需重写）
        pass

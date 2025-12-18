"""
解析器解析的元素格式类型
"""
from enum import Enum


class ElementType(Enum):
    """元素类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    LINK = "link"
    HEADER = "header"  # Markdown格式保留处理？
    FOOTER = "footer"
    CODE = "code"

"""
测试Markdown解析
"""
from app.services.parsers.markdown_parser import MarkdownParser

if __name__ == '__main__':
    markdown_parser = MarkdownParser()
    file_path = "C:/Users/ASUS/Desktop/IntelliKnowledge-RAG/README.md"
    docs = markdown_parser.parse(file_path=file_path)
    print(docs)
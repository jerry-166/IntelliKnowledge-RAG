"""
测试pdf解析器
"""

from app.services.parsers.pdf_parser import PDFParser

if __name__ == '__main__':
    pdf_parser = PDFParser()
    # 纯文字还好
    file_path = "C:/Users/ASUS/Desktop/面试/byteDance1.pdf"
    # 含图片、页面跳转、表格的pdf(图片、跳转连接没有)
    file_path2 = "C:/Users/ASUS/Desktop/IntelliKnowledge-RAG/docs/ceshi-pdf.pdf"

    docs = pdf_parser.parse(file_path=file_path2)
    print(docs)

"""
保存读取到的信息
"""


class RawContent:
    def __init__(self):
        self.docs = [
            {"page": {"text": "", "image": "", "table": "", "others": ""}}
        ]

    def add_text(self, text, page):
        self.docs[page]["text"] = text

from typing import Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class Message(BaseModel):
    id: str
    role: str


a = HumanMessage(content="aaa", message=Message(id="123", role=""))
# print(a.additional_kwargs["message"]["id"])


from langchain_core.messages import HumanMessage

# 初始化时传自定义属性
a = HumanMessage(content="aaa", conversation_id="123", user_tag="vip")

# 访问自定义属性
print(a.additional_kwargs["conversation_id"])  # 输出: 123
print(a.additional_kwargs["user_tag"])         # 输出: vip
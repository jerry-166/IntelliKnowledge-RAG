import datetime
import json
import uuid
from fastapi import APIRouter, HTTPException, Path, Query, Response
from typing import List, Optional
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

# 创建路由实例
router = APIRouter()

# 延迟初始化
user_id = None
rag_workflow = None


def get_user_id():
    global user_id
    if user_id is None:
        from python_services.core.context_login import get_user
        user_id = get_user()
    return user_id


def get_rag_workflow():
    global rag_workflow
    if rag_workflow is None:
        from agent1 import rag_workflow
    return rag_workflow


# 消息模型
class Message(BaseModel):
    id: str
    conversation_id: str
    content: str
    role: str  # user or assistant
    created_at: str
    metadata: Optional[dict] = None


# 对话模型
class Conversation(BaseModel):  # 会话标识用户的话，关联的消息就不用标识了
    id: str
    title: str  # 标题
    created_at: str
    updated_at: str
    metadata: Optional[dict] = None


# 聊天请求模型
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    collection_id: Optional[int] = None  # 标识哪个知识库
    stream: bool = False


# 聊天响应模型
class ChatResponse(BaseModel):
    message_id: str  # 消息会被保存，并返回message_id
    conversation_id: str
    content: str
    role: str
    created_at: str
    references: Optional[List[dict]] = None  # 引用信息


# 模拟对话数据
mock_conversations = [
    Conversation(
        id="conv_1",
        title="测试对话1",
        created_at="2024-01-01T10:00:00",
        updated_at="2024-01-01T10:30:00",
        metadata={'user_id': "1"}
    ),
    Conversation(
        id="conv_2",
        title="测试对话2",
        created_at="2024-01-02T15:00:00",
        updated_at="2024-01-02T15:15:00",
        metadata={'user_id': "2"}
    )
]

# 模拟消息数据
mock_messages = [
    Message(
        id="msg_1",
        conversation_id="conv_1",
        content="你好，能帮我解答一个问题吗？",
        role="user",
        created_at="2024-01-01T10:00:00",
        metadata={}
    ),
    Message(
        id="msg_2",
        conversation_id="conv_1",
        content="当然可以，请问您有什么问题？",
        role="assistant",
        created_at="2024-01-01T10:01:00",
        metadata={}
    )
]


# 获取对话列表
@router.get("/conversations", response_model=List[Conversation])
def get_conversations():
    """获取对话列表"""
    # 从mock_conversations中获取，注意metadata中user_id的比较
    user_id = get_user_id()
    if user_id:
        conversations = [conv for conv in mock_conversations if conv.metadata.get("user_id", None) == user_id]
    else:
        conversations = mock_conversations
    return conversations


# 获取对话历史
@router.get("/conversations/{conv_id}/messages", response_model=dict)
def get_conversation_history(
        conv_id: str = Path(..., description="会话id"),
        page: int = 1,
        page_size: int = 20
):
    """获取对话历史"""
    user_id = get_user_id()
    conversation = next(
        (conv for conv in mock_conversations if conv.id == conv_id and conv.metadata.get('user_id', None) == user_id),
        None)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    messages = [msg for msg in mock_messages if msg.conversation_id == conv_id]

    # 分页处理
    start = (page - 1) * page_size
    end = start + page_size
    paginated_messages = messages[start:end]

    return {
        "conversation_id": conv_id,
        "title": conversation.title,
        "messages": paginated_messages,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_messages": len(messages)
        }
    }


# 创建对话请求模型
class CreateConversationRequest(BaseModel):
    title: Optional[str] = "新对话"
    metadata: Optional[dict] = None


# 创建新对话
@router.post("/conversations", response_model=Conversation)
def create_conversation(data: CreateConversationRequest):
    """创建新对话"""
    user_id = get_user_id()
    new_conv = Conversation(
        id=f"conv_{len(mock_conversations) + 1}",
        title=data.title,
        created_at=datetime.datetime.now().isoformat(),
        updated_at=datetime.datetime.now().isoformat(),
        metadata={
            'user_id': user_id,
            **data.metadata
        }
    )

    mock_conversations.append(new_conv)
    return new_conv


def delete_message(conv_id: str):
    global mock_messages
    mock_messages = [msg for msg in mock_messages if msg.conversation_id != conv_id]


# 删除对话
@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    """删除对话"""
    global mock_conversations
    mock_conversations = [conv for conv in mock_conversations if conv.id != conv_id]

    # 删除关联的消息
    delete_message(conv_id)
    return {
        "success": True,
        "message": "对话删除成功"
    }


# 清空会话
@router.post("/clear")
def clear_conversation(conv_id: str):
    """清空会话"""
    delete_message(conv_id)  # 删除指定id的全部messages
    return {
        "success": True,
        "message": "会话清空成功"
    }


# 非流式聊天
@router.post("/invoke", response_model=ChatResponse)
def chat(request: ChatRequest):
    """非流式聊天"""
    # 对话ID
    conversation_id = request.conversation_id or f"conv_{len(mock_conversations) + 1}"
    # 获取历史信息+query
    input = save_message_get_input(conversation_id, request)

    # 调用工作流
    config = RunnableConfig(configurable={
        "user_id": uuid.uuid4(),
        "thread_id": str(uuid.uuid4()) + conversation_id,
        "search_type": "hybrid",
    })
    response = get_rag_workflow().invoke(
        input=input,
        config=config,
    )

    # 封装结果
    message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        content=response["message"].content,
        role="assistant",
        created_at=datetime.datetime.now()
    )
    mock_messages.append(message)

    # 创建模拟响应
    return ChatResponse(
        message_id=f"msg_{len(mock_messages) + 1}",
        conversation_id=conversation_id,
        content=response["message"].content,
        role="assistant",
        created_at=datetime.datetime.now(),
        references=response["references"]
    )


def save_message_get_input(conversation_id, request):
    # 构建历史信息
    input = [message for message in mock_messages if message.conversation_id == conversation_id]
    message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        content=request.message,
        role="user",
        created_at=datetime.datetime.now().isoformat()
    )
    mock_messages.append(message)
    input.append(message)
    return input


# SSE 响应
@router.post("/stream")
async def stream_chat(request: ChatRequest):
    """流式聊天"""
    # 会话id
    conversation_id = request.conversation_id or f"conv_{len(mock_conversations) + 1}"

    async def event_generator(conversation_id):
        # 获取input
        input = save_message_get_input(conversation_id, request)
        config = RunnableConfig(configurable={
            "user_id": uuid.uuid4(),
            "thread_id": str(uuid.uuid4()) + conversation_id,
            "search_type": "hybrid",
        })

        async for chunk in get_rag_workflow().astream(
                input=input,
                config=config,
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return Response(
        content=event_generator(conversation_id),
        media_type="text/event-stream",  # 声明为SSE格式
        headers={
            "Cache-Control": "no-cache",  # 禁止缓存
            "Connection": "keep-alive",  # 保持长连接
            "Access-Control-Allow-Origin": "*",  # 允许跨域，生产环境需要指定
        }
    )

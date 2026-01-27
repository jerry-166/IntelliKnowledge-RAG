"""
检查检索必要性（模型是否有该方面知识）
 |  评分机制（判断是否有关，无关去重写），分解呢？
\|/
重写/分解问题（语义指代、复杂多问分解）
RAG检索/调用模型
生成答案
---
运行时上下文Context（记录用户id，token等）
保存/获取用户信息的工具（streamWriter）
结构化输出
@wrap_model_call ✅️
@wrap_tool_call
@dynamic_prompt
通过中间件定义状态
"""
import getpass
import os
import uuid
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition, ToolRuntime
from langgraph.store.memory import InMemoryStore
from basic_core.llm_factory import local_qwen, qwen_vision, claude
from basic_core.prompts import GRADE_PROMPT, RESPONSE_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT
from python_services.core.settings import get_config
from python_services.rag_pipeline import RAGPipeline
from python_services.utils.search_kwargs_util import SearchKwargsUtil

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)
pipeline = RAGPipeline(config=get_config(), vision_llm=qwen_vision)


def generate_or_search(state: MessagesState):
    """生成答案或者检索向量库，网络搜索"""
    print(f"进入generate_or_search节点")
    query = state.get("messages", [])[0]
    # 先检索知识库生成上下文
    # print(local_qwen.invoke("你是谁？").content)
    input1 = f"根据问题{query}，如果需要使用retriever_tool工具去查询有关的上下文，则返回一段有关问题的完整的上下文信息，如果不需要，则直接输出答案"
    input2 = f"根据问题{query}，使用retriever_tool工具去查询有关的上下文，则返回一段有关问题的完整的上下文信息"
    outputs = local_qwen.bind_tools([retriever_tool]).invoke(
        input=input2
    )
    print(f"generate_or_search节点输出结果如下：\n{outputs.content}")
    return {
        "messages": AIMessage(outputs.content),
        "references": [{}],
    }


def grade_(state: MessagesState) -> Literal["rewrite_query", "response"]:
    """评判检索到的上下文与需要回答的问题是否一致"""
    print(f"进入grade_节点")
    messages = state.get("messages", [])
    if not messages:
        raise
    query = messages[0]
    context = messages[-1]
    grade_input = GRADE_PROMPT.format({"query": query, "context": context})
    outputs = local_qwen.invoke(grade_input)
    print(f"grade_节点输出结果为：{outputs.content}")
    if hasattr(outputs, "content") and outputs.content == "yes":
        return "response"
    elif hasattr(outputs, "content") and outputs.content == "no":
        return "rewrite_query"


def rewrite_query(state: MessagesState):
    """重写问题，以历史为基准，重写"""
    print("进入rewrite_query节点")
    print("rewrite_query节点输出：")
    pass


def small_query(state: MessagesState):
    """分解问题为小问题，需要融合答案"""
    pass


def response_(state: MessagesState):
    """生成答案"""
    print("进入response_节点")
    query = state.get("messages")[0]
    context = state.get("messages")[-1]
    prompt = (RESPONSE_SYSTEM_PROMPT.format({"query": query, "context": context}))
    output = claude.invoke(input=prompt)
    print(f"response_节点输出结果如下：\n{output.content}")
    return {"messages": AIMessage(output.content)}


@wrap_model_call
def dynamic_model(request: ModelRequest, handler) -> ModelResponse:
    """根据问题上下文长度，来选择不同上下文长度的模型"""
    query = request.state["messages"][-1].content
    query_len = len(query)
    if query_len > 1000:
        print("问题长度过长，选择更长上下文模型")
        request.model = claude
    else:
        print("问题长度适合，选择本地qwen模型")
        request.model = local_qwen
    return handler(request)


@tool()
def retriever_tool(runtime: ToolRuntime):
    """检索知识库，获取与query有关的上下文信息

    Args:
        runtime: ToolRuntime，工具运行时的上下文信息
    """
    print("进入retriever_tool工具")
    user_id = runtime.context.get("user_id")
    filter_dict = SearchKwargsUtil.build_search_kwargs(
        vector_store="faiss",
        filter_conditions={"user_id": user_id}
    )

    outputs = pipeline.search(
        query=runtime.state["messages"][0],
        top_k=5,
        filter_dict=filter_dict,
        search_type=runtime.context.get("search_type", "hybrid"),
        use_reranker=True,
        use_cache=True,
    )
    print(f"retriever_tool工具生成{len(outputs)}条结果")
    return outputs


role = "善解人意，乐于助人的小助手喵喵"
rules = ["回答不涉及黄赌毒，要正能量", "了解用户知识水平，提供详略不同的解释"]
tools = ["tavily_tool", "retriever_tool"]
prompt_template = PromptTemplate.from_template(template=AGENT_SYSTEM_PROMPT,
                                               partial_variables={"rules": "\n".join(rules)})
system_prompt = prompt_template.invoke(input={"role": role, "tools": "\n".join(tools)}).to_string()
agent = create_agent(
    model=local_qwen,
    tools=[tavily_tool, retriever_tool],
    system_prompt=system_prompt,
    middleware=[
        dynamic_model,
        SummarizationMiddleware(
            model=claude,
            max_tokens_before_summary=4000,
            messages_to_keep=10,
        )
    ]
)


def invoke_agent(state: MessagesState):
    """调用Agent"""
    pass


# 构建流图结构
graph = StateGraph(MessagesState)
graph.add_node(generate_or_search)
graph.add_node("retriever", ToolNode(tools=[retriever_tool]))
graph.add_node(grade_)
graph.add_node(rewrite_query)
graph.add_node("response", response_)
graph.add_edge(START, "generate_or_search")
graph.add_conditional_edges(
    "generate_or_search",
    tools_condition,
    {
        "tools": "retriever",
        END: END
    }
)
graph.add_conditional_edges(
    "retriever",
    grade_
)
# graph.add_conditional_edges(
#     "grade_",
#     [rewrite_query, response_],
#     {
#         "rewrite_query": rewrite_query,
#         "response": response_
#     }
# )

graph.add_edge("rewrite_query", "generate_or_search")
graph.add_edge("response", END)

rag_workflow = graph.compile(
    name="知识增强检索工作流",
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)
config = RunnableConfig(configurable={
    "user_id": uuid.uuid4(),
    "thread_id": "u1",
    "search_type": "hybrid",
})
output = rag_workflow.invoke(
    input=MessagesState(messages=[HumanMessage("如何使用LangChain进行知识增强检索？")]),
    config=config
)

print(output)

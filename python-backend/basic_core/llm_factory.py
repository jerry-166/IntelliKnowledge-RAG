"""
llm工厂
"""
import os

from langchain_community.embeddings import DashScopeEmbeddings, ZhipuAIEmbeddings

from zai import ZhipuAiClient

from langchain_ollama import ChatOllama

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

zhipu_embedding = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
)

qwen_embedding = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

claude = ChatOpenAI(
    model="claude-haiku-4-5-20251001",
    base_url="https://apic1.ohmycdn.com/api/v1/ai/openai/cc-omg/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=1024,
    timeout=60,
    max_retries=3,
)

zhipu = ChatOpenAI(
    model="glm-4.6",
    temperature=0.7,
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key=os.getenv("ZHIPU_API_KEY"),
    max_tokens=1024, timeout=60, max_retries=3, )

# 不能识别图像

qwen = ChatOpenAI(
    model="qwen3-max-preview",
    temperature=0.7,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    max_tokens=1024,
    timeout=60,
    max_retries=3,
)

qwen_vision = ChatOpenAI(
    model="qwen3-vl-plus",
    temperature=0.7,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    max_tokens=1024,
    timeout=60,
    max_retries=3,
)

# qwen_vision_vllm = ChatOpenAI(
#     model="Qwen3-VL-2B-Instruct",
#     temperature=0.7,
#     base_url="http://121.40.237.89:8000/v1",
#     api_key='vllm',
#     max_tokens=1024,
#     timeout=60,
#     max_retries=3,
# )

deepseek = ChatOpenAI(
    model="deepseek-v3",
    temperature=0.7,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    max_tokens=1024,
    timeout=60,
    max_retries=3,
)

# 可以识别文字、图像

doubao = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    temperature=0.7,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("ARK_API_KEY"),
    max_tokens=1024, timeout=60, max_retries=3,
)

local_qwen = ChatOllama(
    model="qwen3:0.6b",
    reasoning=True,
    temperature=0.7,
    num_predict=1024,
)


class LLMFactory:
    """
    LLM工厂
    """

    def __init__(
            self,
            model_name: str,
            base_url: str,
            api_key: str,
            temperature: float,
            max_tokens: int,
            timeout: int,
            max_retries: int
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def create_chat_llm(self):
        if self.model_name == "claude":
            return claude
        elif self.model_name == "zhipu":
            return zhipu
        elif self.model_name == "qwen":
            return qwen
        elif self.model_name == "qwen_vision":
            return qwen_vision
        elif self.model_name == "deepseek":
            return deepseek
        elif self.model_name == "doubao":
            return doubao
        elif self.model_name == "local_qwen":
            return local_qwen

    def create_ollama_llm(self, reasoning: bool = False):
        return ChatOllama(
            model=self.model_name,
            reasoning=True,
            temperature=self.temperature,
            num_predict=self.max_tokens,
        )


if __name__ == '__main__':
    invoke = qwen_vision_vllm.invoke("你好，你是谁？")
    print(invoke)

"""Default prompts."""

RESPONSE_SYSTEM_PROMPT = """你是一个乐于助人的小助手，请根据上下文信息回答用户的问题，
上下文：
{context}
要求：
为确保一致性，上下文中最多参考3个句子，
若你不能确定答案，请直接输出‘我不知道’
"""
QUERY_SYSTEM_PROMPT = """Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:

<previous_queries/>
{queries}
</previous_queries>

System time: {system_time}"""
GRADE_PROMPT = """
你是一名评判句子相关性的专家，擅长比较问句和上下文之间的相关性
问句：
{query}
上下文：
{context}
最终你只需要输出：yes（代表相关）或no（代表不相关）
返回json格式
"""
AGENT_SYSTEM_PROMPT = """
你是{role},擅长使用各种工具来完成任务
你有工具：
{tools}
请按要求回答问题：
{rules}
"""

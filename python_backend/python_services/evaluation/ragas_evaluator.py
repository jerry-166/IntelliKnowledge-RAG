from langchain_core.messages import HumanMessage
from ragas import evaluate
from ragas.metrics import (
    _faithfulness,  # 置信度？
    _answer_relevancy,  # 回答相关性
    _context_precision,  # 准确率
    _context_recall,  # 召回率
)
from datasets import Dataset
import pandas as pd

from agent1 import rag_workflow
from python_services.rag_pipeline import RAGPipeline
from python_services.core.settings import get_config


def create_evaluation_dataset(queries_and_ground_truths):
    """创建评估数据集，包含问题，真实答案，上下文，模型回复"""
    data = {
        "question": [],
        "answer": [],
        "contexts": [],  # 检索到的上下文
        "ground_truth": []  # 真实答案
    }
    pipeline = RAGPipeline(config=get_config())

    for item in queries_and_ground_truths:
        question = item['question']
        ground_truth = item['ground_truth']

        contexts = pipeline.call(question)
        answer = rag_workflow.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": f"eval_{hash(question)}"}},
            context={"user_id": "user_111"},
        )

        data['question'].append(question)
        data['answer'].append(answer.content)
        data['contexts'].append([c.page_content for c in contexts])
        data['ground_truth'].append(ground_truth)

    return Dataset.from_dict(data)


def run_ragas_evaluation(dataset: Dataset):
    """执行RAGAS评估"""
    # 定义评估指标
    metrics = [
        _faithfulness,
        _answer_relevancy,
        _context_precision,
        _context_recall,
    ]

    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    # 输出结果
    print(result)
    print(result.to_pandas())  # 转化为 Pandas DataFrame分析
    return result


def generate_evaluation_report(results):
    """生成评估报告"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = results.to_pandas()

    # 创建评估指标图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 创建Faithfulness 评分分布
    sns.histplot(data=df, x='faithfulness', ax=axes[0, 0])
    axes[0, 0].set_title('Faithfulness Distribution')

    # Answer Relevancy 评分分布
    sns.histplot(data=df, x='answer_relevancy', ax=axes[0, 1])
    axes[0, 1].set_title('Answer Relevancy Distribution')

    # Context Precision 评分分布
    sns.histplot(data=df, x='context_precision', ax=axes[1, 0])
    axes[1, 0].set_title('Context Precision Distribution')

    # Context Recall 评分分布
    sns.histplot(data=df, x='context_recall', ax=axes[1, 1])
    axes[1, 1].set_title('Context Recall Distribution')

    plt.tight_layout()
    plt.savefig('rag_evaluation_report.png')


if __name__ == '__main__':
    # 示例测试数据
    test_data = [
        {
            "question": "关于项目的技术架构是什么？",
            "ground_truth": "项目采用RAG架构，包含解析器、分割器、向量存储等组件..."
        },
        {
            "question": "如何配置分块参数？",
            "ground_truth": "可以在配置文件中设置chunk_size和chunk_overlap参数..."
        }
    ]

    # 运行评估
    dataset = create_evaluation_dataset(test_data)
    results = run_ragas_evaluation(dataset)
    generate_evaluation_report(results)
    print(f"平均评分: {results}")

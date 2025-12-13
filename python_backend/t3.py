# main.py
"""
企业级RAG系统 - 使用示例
"""
import logging
from pathlib import Path

from config.settings import RAGConfig, get_config
from services.vector_store.multimodal_store import MultimodalVectorStore
from services.parsers.pdf_parser import PDFParser
from services.parsers.markdown_parser import MultimodalMarkdownParser
from services.splitter.integration_splitter import IntegrationSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG处理管道"""

    def __init__(self, config: RAGConfig = None, vision_llm=None):
        self.config = config or get_config()
        self.vision_llm = vision_llm

        # 初始化组件
        self._init_parsers()
        self._init_splitter()
        self._init_vector_store()

        logger.info("RAG Pipeline initialized")

    def _init_parsers(self):
        """初始化解析器"""
        self.parsers = {
            "pdf": PDFParser(
                vision_llm=self.vision_llm,
                use_ocr=self.config.parser.use_ocr,
                extract_images=True,
                extract_tables=True,
            ),
            "md": MultimodalMarkdownParser(
                vision_llm=self.vision_llm,
                use_ocr=self.config.parser.use_ocr,
                use_vision=self.config.parser.use_vision,
            ),
            # 添加更多解析器...
        }

    def _init_splitter(self):
        """初始化分割器"""
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding.text_model,
            model_kwargs={'device': self.config.embedding.device},
        )

        self.splitter = IntegrationSplitter(
            embeddings=embeddings,
            chunk_size=self.config.splitter.chunk_size,
            chunk_overlap=self.config.splitter.chunk_overlap,
        )

    def _init_vector_store(self):
        """初始化向量存储"""
        self.vector_store = MultimodalVectorStore(
            config=self.config,
            vision_llm=self.vision_llm,
        )

    def ingest(self, file_path: str) -> int:
        """
        摄入文档

        Args:
            file_path: 文件路径

        Returns:
            添加的文档数量
        """
        path = Path(file_path)
        suffix = path.suffix.lower().lstrip('.')

        # 选择解析器
        parser = self.parsers.get(suffix)
        if not parser:
            raise ValueError(f"No parser for format: {suffix}")

        # 解析
        logger.info(f"Parsing: {file_path}")
        documents = parser.parse(file_path)

        # 分割
        logger.info(f"Splitting {len(documents)} documents")
        split_docs = self.splitter.split_documents_(documents, file_type=suffix)

        # 存储
        logger.info(f"Storing {len(split_docs)} chunks")
        self.vector_store.add_documents(split_docs)

        return len(split_docs)

    def ingest_directory(self, dir_path: str) -> int:
        """摄入目录下所有文件"""
        total = 0
        dir_path = Path(dir_path)

        for ext in self.parsers.keys():
            for file in dir_path.glob(f"**/*.{ext}"):
                try:
                    count = self.ingest(str(file))
                    total += count
                except Exception as e:
                    logger.error(f"Failed to ingest {file}: {e}")

        return total

    def search(
            self,
            query: str,
            top_k: int = 10,
            use_reranker: bool = True,
    ) -> list:
        """
        检索

        Args:
            query: 查询文本
            top_k: 返回数量
            use_reranker: 是否使用重排序

        Returns:
            检索结果列表
        """
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            use_reranker=use_reranker,
        )

        return [r.to_dict() for r in results]

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "parsers": {
                name: parser.get_stats()
                for name, parser in self.parsers.items()
            },
        }


# 使用示例
if __name__ == '__main__':
    from basic_core.llm_factory import qwen_vision

    # 创建配置
    config = RAGConfig(
        vector_store=VectorStoreConfig(
            provider="chroma",
            persist_directory="./data/vector_db",
        ),
        reranker=RerankerConfig(
            enabled=True,
            model="BAAI/bge-reranker-base",
        ),
        cache=CacheConfig(
            enabled=True,
            backend="memory",
        ),
    )

    # 初始化管道
    pipeline = RAGPipeline(config=config, vision_llm=qwen_vision)

    # 摄入文档
    pipeline.ingest("./docs/example.pdf")
    pipeline.ingest("./docs/example.md")

    # 检索
    results = pipeline.search(
        query="什么是深度学习？",
        top_k=5,
        use_reranker=True,
    )

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']:.4f}) ---")
        print(f"Content: {result['content'][:200]}...")

    # 查看统计
    print("\n--- Stats ---")
    print(pipeline.get_stats())
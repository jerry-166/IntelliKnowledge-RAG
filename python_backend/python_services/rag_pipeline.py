"""
RAG检索的主流程
"""
import logging
from pathlib import Path
from typing import Optional, Literal, Any
from uuid import uuid4

from basic_core.llm_factory import qwen_vision
from python_services.core.settings import RAGConfig, get_config
from python_services.parsers.image_parser import ImageParser
from python_services.parsers.markdown_parser import MultimodalMarkdownParser
from python_services.parsers.pdf_parser import PDFParser
from python_services.parsers.ppt_parser import PPtParser
from python_services.parsers.text_parser import TextParser
from python_services.parsers.web_parser import WebParser
from python_services.parsers.word_parser import WordParser
from python_services.splitter.integration_splitter import IntegrationSplitter
from python_services.vector_store.multimodal_store import MultimodalVectorStore
import python_services.core.logger_config


logger = logging.getLogger(__name__)


"""
1. suffix = file_path.suffix.lower().lstrip('.')
    移除字符串开头的所有点号字符
    strip(): 移除字符串两端（开头和结尾）的指定字符
    lstrip(): 只移除字符串开头的指定字符
    rstrip(): 只移除字符串结尾的指定字符
    
2. for i, result in enumerate(results, 1)
    i从1开始
"""


class RAGPipeline:
    """RAG处理管道"""

    def __init__(self, config: RAGConfig = None, vision_llm=None):
        self.config = config or get_config()
        self.vision_llm = vision_llm

        self._init_parsers()
        self._init_splitter()
        self._init_vector_store()

        logger.info("✅️RAG Pipeline initialized")

    def _init_parsers(self):
        """初始化不同的解析器"""
        image_parser = ImageParser(vision_llm=self.vision_llm, use_ocr=self.config.parser.use_ocr,
                                   use_vision=self.config.parser.use_vision, )
        text_parser = TextParser()
        self.parsers = {
            "pdf": PDFParser(
                vision_llm=self.vision_llm,
                use_ocr=self.config.parser.use_ocr,
                extract_images=self.config.parser.extract_images,
                extract_tables=self.config.parser.extract_tables,
                min_image_size=self.config.parser.min_image_size,
            ),
            "md": MultimodalMarkdownParser(
                vision_llm=self.vision_llm,
                use_ocr=self.config.parser.use_ocr,
                use_vision=self.config.parser.use_vision,
            ),
            "png": image_parser,
            "jpg": image_parser,
            "jpeg": image_parser,
            "txt": text_parser,
            "docx": WordParser(),
            "ppt": PPtParser(
                vision_llm=self.vision_llm,
                use_ocr=self.config.parser.use_ocr,
                use_vision=self.config.parser.use_vision,
            ),
            "url": WebParser(
                use_ocr=self.config.parser.use_ocr,
                use_vision=self.config.parser.use_vision,
            )
            # 添加别的解析器...
        }

        logger.info(f"✅️Parsers initialized: {self.parsers.keys()}")

    def _init_splitter(self):
        """初始化切分器"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            model_kwargs = {
                'device': self.config.splitter.device,
                'trust_remote_code': self.config.splitter.trust_remote_code,
            }
            semantic_embedding = HuggingFaceEmbeddings(
                model_name=self.config.splitter.semantic_embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': self.config.splitter.normalize_embeddings}
            )
            self.splitter = IntegrationSplitter(
                semantic_embedding=semantic_embedding,
                chunk_size=self.config.splitter.chunk_size,
                chunk_overlap=self.config.splitter.chunk_overlap,
            )

            logger.info(f"✅️Splitter initialized: {self.splitter}")
        except ImportError:
            raise ImportError("❌️Please install langchain_huggingface to use HuggingFaceEmbeddings")

    def _init_vector_store(self):
        """初始化向量存储"""
        self.vector_store = MultimodalVectorStore(
            rag_config=self.config,
        )

    def ingest(self, file_path: str, show_progress: bool = False):
        """
        摄入文档

        Args:
            file_path: 文件路径
            show_progress: 是否展示进程

        Returns:
            添加的文档数量
        """
        path = Path(file_path)
        suffix = path.suffix.lower().lstrip('.')
        parser = self.parsers.get(suffix)
        if not parser:
            raise ValueError(f"❌️No parser for format: {suffix}")
        # 解析文档
        logger.info(f"Parsing file: {file_path}")
        documents = parser.parse(file_path)

        # 切分文档
        logger.info(f"Splitting {len(documents)} documents")
        split_documents = self.splitter.split_documents_(documents, file_suffix=suffix)
        # 存储向量
        logger.info(f"Storing {len(split_documents)} chunks")
        doc_ids = self.vector_store.add_documents(
            split_documents,
            batch_size=self.config.vector_store.batch_size,
            show_progress=show_progress
        )

        return len(doc_ids)

    def parse_file(self, file_path):
        path = Path(file_path)
        suffix = path.suffix.lower().lstrip('.')
        parser = self.parsers.get(suffix)
        if not parser:
            raise ValueError(f"❌️No parser for format: {suffix}")
        # 解析文档
        logger.info(f"Parsing file: {file_path}")
        documents = parser.parse(file_path)
        return documents

    def ingest_directory(self, dir_path: str, show_progress: bool = False):
        """
        批量摄入目录下的所有文件
        """
        total = 0
        dir_path = Path(dir_path)
        for ext in self.parsers.keys():
            for file in dir_path.glob(f"**/*.{ext}"):
                try:
                    count = self.ingest(str(file), show_progress=show_progress)
                    total += count
                except Exception as e:
                    logger.error(f"Failed to ingest {file}: {e}")

        return total

    def search(
            self,
            query: str,
            top_k: int = 10,
            filter_dict: Optional[dict[str, Any]] = None,
            search_type: Literal["text", "image", "hybrid"] = "text",
            use_reranker: bool = True,
            use_cache: bool = True,
    ) -> list:
        """
        检索

        Args:
            query: 检索文本
            top_k: 返回数量
            filter_dict: 过滤字典
            search_type: 搜索模式
            use_reranker: 是否使用重排器
            use_cache: 是否使用缓存

        Returns:
            检索结果列表
        """
        search_results = self.vector_store.search(
            query=query, top_k=top_k, filter_dict=filter_dict,
            search_type=search_type, use_reranker=use_reranker,
            use_cache=use_cache
        )
        return [r.to_dict() for r in search_results]

    def get_stats(self):
        """获取统计信息"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "parsers": {
                name: parser.get_stats()
                for name, parser in self.parsers.items()
            }
        }


def call(query: str):
    results = pipeline.search(
        query=query,
        filter_dict={"ext": "pdf"},
        search_type="hybrid",
        top_k=5,
        use_reranker=True,
    )
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']:.4f}) ---")
        print(f"Content: {result['content'][:200]}...")
    # 查看统计
    print("\n--- Stats ---")
    print(pipeline.get_stats())


if __name__ == '__main__':
    # 初始化管道
    pipeline = RAGPipeline(config=get_config(), vision_llm=qwen_vision)

    # 摄入文档
    # file_name = r"C:\Users\ASUS\Desktop\video.pdf"
    # pipeline.ingest(file_name, show_progress=True)
    # file_dir = r"C:\Users\ASUS\Desktop\pythonCode\LangChain\second_start\src\agent\static\pdf"
    # pipeline.ingest_directory(file_dir, show_progress=True)
    # file_name = r"C:\Users\ASUS\Desktop\makedown\deepAgent.md"
    # pipeline.ingest(file_name, show_progress=True)

    # 检索
    call("是一个由几何形状构成的抽象图形标志，不包含任何文字、数据或传统意义上的图表。其内容描述如下：- **整体结构**：图像由几个相互交叠的红色和灰色几何形状组成，整体呈现动态和现代感。")
    # call("什么事提示词工程？")
    # call("西北工业大学最高是多少？")
    # call("西北工业大学最高分是多少？")

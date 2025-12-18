from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS, DistanceStrategy

from basic_core.llm_factory import init_embedding_basic
from python_services.core.settings import CacheConfig
from python_services.utils.index_util import IndexUtil

embed = init_embedding_basic(CacheConfig.embedding)

index = IndexUtil.get_faiss_index(embed, "flat", 100)

faiss = FAISS(
    embedding_function=embed,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    distance_strategy=DistanceStrategy.COSINE,
)

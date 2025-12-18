"""
测试一下ClipEmbedding
"""
from python_services.core.settings import get_config
from python_services.embeddings.clip_embedding import ClipEmbeddings

clip_embedding = ClipEmbeddings(get_config().embedding.clip_embedding_model)
score = clip_embedding.compute_similarity(
    "a handsome man with his girlfriend sex in the bed",
    r"C:\Users\ASUS\Desktop\小狗.jpg",
)
print(score)

"""
测试一下ClipEmbedding
"""
from python_services.core.settings import get_config
from python_services.embeddings.clip_embedding import CLIPEmbeddings

clip_embedding = CLIPEmbeddings(get_config().embedding.clip_embedding_model)
file_list = [r"C:\Users\ASUS\Desktop\小狗.jpg", r"C:\Users\ASUS\Desktop\鲜花.png"]
score = clip_embedding.compute_similarity(
    "一个狼人",
    file_list,
)
print(score)

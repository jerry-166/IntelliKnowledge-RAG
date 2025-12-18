"""
索引的工具类
"""


class IndexUtil:
    @staticmethod
    def get_faiss_index(embedding, index_type: str, faiss_nlist):
        import faiss
        embedding_dim = len(embedding.embed_query("hello world"))
        # 选择索引类型
        if index_type == "flat":
            # 精确搜索（适合小数据集）
            index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "ivf":
            # 需要先训练才能使用
            # IVF 索引（适合中等数据集）
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = faiss_nlist or 100
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        elif index_type == "hnsw":
            # HNSW 索引（适合大数据集，高召回）
            index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            index = faiss.IndexFlatIP(embedding_dim)
        return index

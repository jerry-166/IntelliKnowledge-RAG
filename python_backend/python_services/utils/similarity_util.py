"""
求相似度的工具类
"""
import numpy as np
import similarities


class SimilarityUtil:
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray):
        """计算余弦相似度"""
        return float(np.dot(a, b) )

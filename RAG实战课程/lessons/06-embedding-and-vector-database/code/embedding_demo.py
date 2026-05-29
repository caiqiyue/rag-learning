"""
Embedding 模型调用示例
支持 OpenAI、本地模型等
"""

from typing import List
import numpy as np


class EmbeddingModel:
    """
    Embedding 模型统一接口
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        if self.model_name.startswith("text-embedding"):
            self._init_openai()
        elif self.model_name.startswith("BGE") or self.model_name.startswith("bge"):
            self._init_bge()
        else:
            self._init_openai()  # 默认用 OpenAI

    def _init_openai(self):
        """OpenAI Embedding"""
        from langchain.embeddings import OpenAIEmbeddings

        self.embeddings = OpenAIEmbeddings(model=self.model_name)
        self._embed_func = self._embed_openai

    def _init_bge(self):
        """BGE 开源模型"""
        from langchain.embeddings import HuggingFaceBgeEmbeddings

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._embed_func = self._embed_huggingface

    def _embed_openai(self, text: str) -> List[float]:
        """OpenAI Embedding"""
        return self.embeddings.embed_query(text)

    def _embed_huggingface(self, text: str) -> List[float]:
        """HuggingFace Embedding"""
        return self.embeddings.embed_query(text)

    def embed(self, text: str) -> List[float]:
        """向量化单个文本"""
        return self._embed_func(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        return self.embeddings.embed_documents(texts)

    def get_dimension(self) -> int:
        """获取向量维度"""
        # 实际应用中可以通过 API 或模型配置获取
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "BAAI/bge-large-zh-v1.5": 1024,
        }
        return dimensions.get(self.model_name, 1536)


def calculate_similarity(v1: List[float], v2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    v1 = np.array(v1)
    v2 = np.array(v2)

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 初始化模型
    model = EmbeddingModel("text-embedding-3-small")
    print(f"模型: {model.model_name}")
    print(f"维度: {model.get_dimension()}")

    # 向量化
    texts = [
        "RAG 是检索增强生成技术",
        "RAG 的全称是 Retrieval-Augmented Generation",
        "深度学习是机器学习的分支",
    ]

    vectors = model.embed_batch(texts)

    print(f"\n向量数量: {len(vectors)}")
    print(f"每个向量维度: {len(vectors[0])}")

    # 计算相似度
    print("\n=== 相似度矩阵 ===")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            sim = calculate_similarity(vectors[i], vectors[j])
            print(f"{t1[:15]}... vs {t2[:15]}... : {sim:.4f}")

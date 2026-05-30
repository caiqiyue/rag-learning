"""
检索器实现
支持混合检索、Reranker、上下文压缩
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """检索结果"""

    content: str
    metadata: Dict
    score: float
    rank: int = 0


class BaseRetriever:
    """检索器基类"""

    def search(self, query_vector: List[float], k: int = 10) -> List[RetrievalResult]:
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    """向量检索器"""

    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    def search(self, query_vector: List[float], k: int = 10) -> List[RetrievalResult]:
        results = self.vectorstore.similarity_search_by_vector(query_vector, k=k)

        return [
            RetrievalResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=1.0 - doc.metadata.get("distance", 0),  # Chroma 用 distance
                rank=i,
            )
            for i, doc in enumerate(results)
        ]


class KeywordRetriever(BaseRetriever):
    """关键词检索器 (BM25)"""

    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.index = self._build_bm25_index()

    def _build_bm25_index(self):
        """构建 BM25 索引"""
        # 简化实现，实际可用 rank_bm25 库
        return {}

    def search(self, query_vector: List[float], k: int = 10) -> List[RetrievalResult]:
        # 简化实现
        return []


class HybridRetriever:
    """混合检索器"""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: Optional[KeywordRetriever] = None,
        weights: List[float] = [0.5, 0.5],
    ):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.weights = weights

    def search(
        self, query: str, query_vector: List[float], k: int = 10
    ) -> List[RetrievalResult]:
        # 1. 向量检索
        vector_results = self.vector_retriever.search(query_vector, k=k * 2)

        # 2. 关键词检索（如果有）
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.search(query_vector, k=k * 2)
        else:
            keyword_results = []

        # 3. RRF 融合
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results, k=60)

        # 4. 截取 top-k
        return fused[:k]

    def _reciprocal_rank_fusion(
        self,
        results1: List[RetrievalResult],
        results2: List[RetrievalResult],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """RRF 融合"""
        scores = {}

        # 向量检索得分
        for i, r in enumerate(results1):
            key = r.content[:100]
            scores[key] = scores.get(key, 0) + self.weights[0] / (k + i)

        # 关键词检索得分
        for i, r in enumerate(results2):
            key = r.content[:100]
            scores[key] = scores.get(key, 0) + self.weights[1] / (k + i)

        # 排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回完整结果
        doc_map = {r.content[:100]: r for r in results1 + results2}

        return [
            RetrievalResult(
                content=doc_map[key].content,
                metadata=doc_map[key].metadata,
                score=score,
                rank=i,
            )
            for i, (key, score) in enumerate(sorted_items)
        ]


class Reranker:
    """重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """加载模型"""
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, candidates: List[RetrievalResult], top_k: int = 10
    ) -> List[RetrievalResult]:
        """重排序"""
        self._load_model()

        pairs = [(query, r.content) for r in candidates]
        scores = self.model.predict(pairs)

        # 排序
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                content=r.content, metadata=r.metadata, score=float(s), rank=i
            )
            for i, (r, s) in enumerate(scored[:top_k])
        ]


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建检索器
    retriever = VectorRetriever(
        vectorstore=None,  # 需要传入真实的 vectorstore
        embeddings=None,  # 需要传入真实的 embeddings
    )

    # 搜索
    # results = retriever.search(query_vector=[...], k=10)
    # print(f"检索到 {len(results)} 个结果")

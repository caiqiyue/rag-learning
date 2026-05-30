"""
向量数据库使用示例
支持 Chroma、FAISS 等主流向量库
"""

from typing import List, Dict, Optional
import numpy as np


class VectorStoreManager:
    """
    向量数据库管理器
    支持 Chroma、FAISS 等
    """

    def __init__(self, db_type: str = "chroma", persist_dir: str = "./vector_db"):
        self.db_type = db_type
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.embeddings = None
        self._initialize()

    def _initialize(self):
        """初始化"""
        from langchain.embeddings import OpenAIEmbeddings

        self.embeddings = OpenAIEmbeddings()

        if self.db_type == "chroma":
            self._init_chroma()
        elif self.db_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")

    def _init_chroma(self):
        """初始化 Chroma"""
        from langchain.vectorstores import Chroma

        self.vectorstore = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def _init_faiss(self):
        """初始化 FAISS"""
        from langchain.vectorstores import FAISS

        # FAISS 需要先有文档才能初始化
        self.vectorstore = None
        self.faiss_type = "faiss"

    def add_texts(
        self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None
    ):
        """添加文档"""
        if self.db_type == "chroma":
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            self.vectorstore.persist()

        elif self.db_type == "faiss":
            if self.vectorstore is None:
                # 第一次：直接创建
                self.vectorstore = FAISS.from_texts(
                    texts=texts, embedding=self.embeddings, metadatas=metadatas, ids=ids
                )
            else:
                # 后续：追加
                self.vectorstore.add_texts(
                    texts=texts, embedding=self.embeddings, metadatas=metadatas, ids=ids
                )

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Dict = None,
        return_scores: bool = True,
    ) -> List[Dict]:
        """
        搜索

        Args:
            query: 查询文本
            k: 返回数量
            filter_dict: 元数据过滤条件
            return_scores: 是否返回相似度分数

        Returns:
            搜索结果列表
        """
        if self.vectorstore is None:
            return []

        if return_scores:
            results = self.vectorstore.similarity_search_with_score(
                query=query, k=k, filter=filter_dict
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": score,
                }
                for doc, score in results
            ]
        else:
            results = self.vectorstore.similarity_search(
                query=query, k=k, filter=filter_dict
            )
            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]

    def search_by_vector(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """用向量搜索"""
        if self.vectorstore is None:
            return []

        if self.db_type == "chroma":
            results = self.vectorstore.similarity_search_by_vector(
                embedding=query_vector, k=k
            )
        elif self.db_type == "faiss":
            results = self.vectorstore.similarity_search_by_vector(query_vector, k=k)

        return [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in results
        ]

    def save(self):
        """保存"""
        if self.db_type == "chroma":
            self.vectorstore.persist()
        elif self.db_type == "faiss":
            if self.vectorstore:
                self.vectorstore.save(self.persist_dir)

    def load(self):
        """加载"""
        if self.db_type == "chroma":
            self.vectorstore = Chroma(
                collection_name="documents",
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        elif self.db_type == "faiss":
            self.vectorstore = FAISS.load(self.persist_dir, self.embeddings)

    def delete(self, ids: List[str]):
        """删除"""
        if self.vectorstore:
            self.vectorstore.delete(ids=ids)
            self.save()

    def get_embedding(self, text: str) -> List[float]:
        """获取向量"""
        return self.embeddings.embed_query(text)


# ========== 相似度计算 ==========


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """余弦相似度"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """欧氏距离"""
    return float(np.linalg.norm(v1 - v2))


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """点积"""
    return float(np.dot(v1, v2))


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 初始化
    manager = VectorStoreManager(db_type="chroma", persist_dir="./chroma_db")

    # 添加文档
    docs = [
        "RAG 是检索增强生成技术",
        "Embedding 将文本转换为向量",
        "向量数据库用于存储和检索",
    ]
    manager.add_texts(docs)

    # 搜索
    results = manager.search("RAG 是什么", k=2)
    for r in results:
        print(f"[{r['distance']:.4f}] {r['content']}")

# 第六课：Embedding 与向量数据库

## 本节概述

本节课将学习 RAG 系统的核心：如何将文本转换为向量（Embedding），以及如何存储和检索这些向量。你将掌握：

- Embedding 的工作原理
- 主流 Embedding 模型对比
- 向量数据库的原理和选型
- 相似度计算方法
- Embedding 实战代码

**学习时长**：约 3-4 小时

---

## 6.1 Embedding 是什么？

### 6.1.1 从文字到数字

**Embedding（嵌入）**是将离散的文字转换为连续的数字向量（Vector）的技术。

```
文字 → 向量

"Apple"     → [0.92, -0.31, 0.15, ...]  # 1536维
"苹果公司"  → [0.89, -0.35, 0.18, ...]  # 语义相近
"手机"      → [0.45, 0.72, -0.22, ...]   # 语义不同
```

**为什么需要 Embedding？**

```
传统方法：关键词匹配
问题："苹果"能匹配"Apple"，但匹配不了"水果"或"乔布斯"

Embedding 方法：语义相似度
"苹果"和"Apple" → 向量相近
"苹果"和"水果" → 向量相近
"苹果"和"手机" → 向量中等相近
"苹果"和"汽车" → 向量相远
```

### 6.1.2 Embedding 的直观理解

**生活中的类比**：书架与书

```
想象一个图书馆：

                    科学区
                       ↓
    ┌─────────────────────────────────────┐
    │                                     │
文学区 ← Apple/苹果/水果 → 技术区
    │                                     │
    └─────────────────────────────────────┘
                       ↑
                    商业区

Embedding 就像把书分配到书架上：
- 语义相近的书放在相邻位置
- 我们可以用"位置"（向量）来描述每本书
- 找书时，找"位置相近"的书就是"语义相近"的书
```

### 6.1.3 Embedding 的数学原理

**核心思想**：将语义相近的内容映射到向量空间中相近的位置

```
向量空间示例（简化到 2 维）：

                    Y (语义轴)
                    ↑
                    |
    苹果            |           香蕉
    [0.9, 0.3]     |         [0.85, 0.35]
                    |
    ────────────────┼──────────────→ X (形式轴)
                    |
    文档A           |           文档B
    [0.2, 0.1]     |
                    |
                    |

向量相似度计算：
- 余弦相似度：cos(θ) = (A·B) / (|A| × |B|)
- 值越接近 1，表示越相似
```

---

## 6.2 主流 Embedding 模型

### 6.2.1 OpenAI Embedding

```python
from langchain.embeddings import OpenAIEmbeddings

# 初始化
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 最新版
)

# 向量化单个文本
query = "什么是 RAG？"
query_vector = embeddings.embed_query(query)

# 向量化多个文本
texts = ["RAG 是检索增强生成", "Embedding 是向量化技术"]
doc_vectors = embeddings.embed_documents(texts)

print(f"向量维度: {len(query_vector)}")
print(f"向量前5维: {query_vector[:5]}")
```

### 6.2.2 模型对比

| 模型 | 维度 | 中文支持 | 特点 | 价格 |
|------|------|---------|------|------|
| text-embedding-3-small | 1536 | 良好 | 最新版，效果好 | ¥0.15/1M tokens |
| text-embedding-ada-002 | 1536 | 良好 | 稳定版 | ¥0.4/1M tokens |
| BGE-large-zh | 1024 | 优秀 | 开源，中文优化 | 免费（本地） |
| M3E | 768 | 优秀 | 开源，轻量 | 免费（本地） |
| GTE-large-zh | 1024 | 优秀 | 阿里开源 | 免费（本地） |

### 6.2.3 开源模型使用

```python
# 使用 BGE（大模型国产开源）
from langchain.embeddings import HuggingFaceBgeEmbeddings

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 本地运行，不需要 API Key
query_vector = embeddings.embed_query("你好，世界")
```

---

## 6.3 向量数据库

### 6.3.1 为什么需要向量数据库？

```
普通数据库 vs 向量数据库

普通数据库（MySQL/PostgreSQL）：
├── 存储：结构化数据（表格）
├── 查询：精确匹配 WHERE name = 'Apple'
└── 索引：B-tree、Hash

向量数据库：
├── 存储：向量 + 元数据
├── 查询：相似度搜索 "找出最相似的内容"
└── 索引：HNSW、IVF、PQ 等向量索引
```

### 6.3.2 主流向量数据库对比

| 数据库 | 特点 | 适用场景 | 部署方式 |
|--------|------|---------|----------|
| **Chroma** | 轻量、本地优先 | 快速原型、单机 | 嵌入/本地 |
| **Milvus** | 分布式、高可用 | 企业级生产 | 云/私有 |
| **Pinecone** | 云原生、免运维 | 云服务 | 全托管 |
| **Weaviate** | 混合检索 | 多模态 | 云/私有 |
| **Faiss** | Facebook开源 | 离线分析 | 本地 |
| **Qdrant** | Rust开发、性能高 | 生产级 | 云/私有 |
| **pgvector** | PostgreSQL扩展 | 已有PG场景 | 集成部署 |

### 6.3.3 Chroma（推荐入门）

```python
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 初始化
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # 持久化路径
)

# 添加文档
texts = [
    "RAG 是检索增强生成技术",
    "Embedding 将文本转换为向量",
    "向量数据库用于存储和检索向量"
]

# 方法1：直接添加
vectorstore.add_texts(texts=texts)

# 方法2：带元数据添加
metadatas = [
    {"source": "doc1", "topic": "RAG"},
    {"source": "doc2", "topic": "Embedding"},
    {"source": "doc3", "topic": "向量库"}
]
vectorstore.add_texts(texts=texts, metadatas=metadatas)

# 相似度搜索
query = "什么是检索增强生成？"
results = vectorstore.similarity_search(query=query, k=3)

# 带相似度分数的搜索
results_with_scores = vectorstore.similarity_search_with_score(
    query=query, k=3
)

for doc, score in results_with_scores:
    print(f"[{score:.4f}] {doc.page_content}")
```

### 6.3.4 带过滤的搜索

```python
# 元数据过滤
results = vectorstore.similarity_search(
    query="RAG 技术",
    k=5,
    filter={"topic": "RAG"}  # 只搜索 topic=RAG 的文档
)

# 复杂过滤
results = vectorstore.similarity_search(
    query="产品介绍",
    k=5,
    filter={
        "$and": [
            {"source": {"$eq": "product"}},
            {"category": {"$eq": "electronics"}}
        ]
    }
)
```

---

## 6.4 相似度计算

### 6.4.1 三种相似度度量

```python
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """余弦相似度：[-1, 1]，越接近1越相似"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2)

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """点积：值越大越相似（不归一化）"""
    return np.dot(v1, v2)

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """欧氏距离：越小越相似"""
    return np.linalg.norm(v1 - v2)

# 示例
v1 = np.array([0.9, 0.3])
v2 = np.array([0.85, 0.35])

print(f"余弦相似度: {cosine_similarity(v1, v2):.4f}")  # 接近 1
print(f"点积: {dot_product(v1, v2):.4f}")
print(f"欧氏距离: {euclidean_distance(v1, v2):.4f}")  # 接近 0
```

### 6.4.2 Chroma 中的相似度

```python
# Chroma 默认使用余弦相似度
# 也可以指定其他度量方式

from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# 创建集合时指定度量方式
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}  # cosine / l2 / ip
)
```

---

## 6.5 实战：完整的 Embedding 和向量库流程

```python
"""
完整的 Embedding 和向量数据库操作流程
"""

from typing import List, Dict
import numpy as np


class EmbeddingPipeline:
    """
    Embedding 和向量库完整流程
    """
    
    def __init__(self, embedding_model="text-embedding-3-small"):
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
    
    def initialize_vectorstore(self, persist_directory: str):
        """初始化向量数据库"""
        from langchain.vectorstores import Chroma
        
        self.vectorstore = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ):
        """添加文档到向量库"""
        if self.vectorstore is None:
            raise ValueError("请先调用 initialize_vectorstore")
        
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # 持久化
        self.vectorstore.persist()
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        搜索最相似的文档
        
        Returns:
            [{"content": str, "metadata": dict, "distance": float}, ...]
        """
        if self.vectorstore is None:
            raise ValueError("请先调用 initialize_vectorstore")
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "distance": score  # 距离（越小越相似）
            }
            for doc, score in results
        ]
    
    def search_by_vector(
        self,
        query_vector: List[float],
        k: int = 5
    ) -> List[Dict]:
        """用向量搜索"""
        if self.vectorstore is None:
            raise ValueError("请先调用 initialize_vectorstore")
        
        results = self.vectorstore.similarity_search_by_vector(
            embedding=query_vector,
            k=k
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量"""
        return self.embeddings.embed_query(text)
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        return self.embeddings.embed_documents(texts)
    
    def delete(self, ids: List[str]):
        """删除文档"""
        if self.vectorstore is None:
            raise ValueError("请先调用 initialize_vectorstore")
        
        self.vectorstore.delete(ids=ids)
        self.vectorstore.persist()


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 初始化
    pipeline = EmbeddingPipeline()
    pipeline.initialize_vectorstore("./chroma_db")
    
    # 添加文档
    docs = [
        "RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写。",
        "RAG 由三个步骤组成：检索、增强、生成。",
        "Embedding 是将文本转换为向量的技术。",
        "向量数据库用于存储和检索向量。",
        "Chroma 是一个轻量级的向量数据库。"
    ]
    
    metadatas = [
        {"source": "doc1", "topic": "RAG"},
        {"source": "doc2", "topic": "RAG"},
        {"source": "doc3", "topic": "Embedding"},
        {"source": "doc4", "topic": "向量库"},
        {"source": "doc5", "topic": "向量库"}
    ]
    
    pipeline.add_documents(texts=docs, metadatas=metadatas)
    
    # 搜索
    print("=== 搜索：RAG 相关内容 ===")
    results = pipeline.search("RAG 是什么", k=3)
    for r in results:
        print(f"[{r['distance']:.4f}] {r['content']}")
        print(f"   元数据: {r['metadata']}")
        print()
    
    # 带过滤的搜索
    print("=== 搜索：仅向量库相关内容 ===")
    results = pipeline.search("数据库", k=3, filter_dict={"topic": "向量库"})
    for r in results:
        print(f"[{r['distance']:.4f}] {r['content']}")
```

---

## 6.6 向量索引算法

### 6.6.1 为什么要索引？

```
暴力搜索 vs 索引搜索

暴力搜索（Brute Force）：
- 计算查询向量与所有向量的相似度
- 100万向量 = 100万次计算
- 精确但慢

索引搜索（Approximate Nearest Neighbor - ANN）：
- 预先构建索引结构
- 查询时只需计算部分向量
- 近似但快（99%+ 准确率）
```

### 6.6.2 主流索引算法

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **HNSW** | 速度快、内存占用高 | 通用场景，推荐 |
| **IVF** | 聚类加速 | 大规模数据 |
| **PQ** | 压缩存储 | 内存受限场景 |
| **LSH** | 局部敏感哈希 | 精确匹配 |

### 6.6.3 Chroma 的 HNSW

```python
# Chroma 默认使用 HNSW 索引
# 可以通过 metadata 配置参数

from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# 创建集合时配置 HNSW 参数
collection = client.create_collection(
    name="my_collection",
    metadata={
        "hnsw:space": "cosine",      # 度量方式
        "hnsw:construction_ef": 100, # 构建参数（越高越精确越慢）
        "hnsw:search_ef": 100         # 搜索参数（越高越精确越慢）
    }
)
```

---

## 本节总结

### 核心要点

1. **Embedding**：将文本转换为语义向量，相似文本 → 相近向量
2. **相似度度量**：余弦相似度（常用）、点积、欧氏距离
3. **向量数据库**：Chroma（原型）、Milvus（企业）、Pinecone（云）
4. **索引算法**：HNSW（推荐）、IVF、PQ

### 代码文件

- `code/vector_store.py`: 向量数据库操作
- `code/embedding_demo.py`: Embedding 调用示例

### 思考题

1. 为什么 Embedding 能捕捉语义相似性，但关键词匹配不能？
2. 向量数据库和普通数据库的本质区别是什么？
3. HNSW 索引的 `search_ef` 参数越大越好吗？为什么？

### 下节预告

下一节课我们将学习高级检索策略，掌握从初步召回到最后重排序的完整检索流程。

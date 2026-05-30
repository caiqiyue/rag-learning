"""
第06节 检索器类型与相似度计算 - 代码示例

本文件演示RAG系统中检索器的核心概念和实现方法，包括：
1. 相似度度量：余弦相似度、点积、欧氏距离
2. 检索器类型：VectorIndexRetriever、BM25Retriever、QueryFusionRetriever
3. Top-K检索与相似度阈值检索

依赖安装：
    pip install llama-index llama-index-retrievers-bm25 rank-bm25 sentence-transformers
"""

# ============================================================================
# 第一部分：相似度度量原理解释
# ============================================================================
"""
相似度度量是检索系统的核心，它决定了如何判断查询与文档之间的相关性。

三种主要的相似度度量方式：
1. 余弦相似度（Cosine Similarity）：衡量两个向量方向的相似程度，取值范围[-1, 1]
   - 值越接近1表示两个向量越相似
   - 值越接近-1表示两个向量越相反
   - 值接近0表示两个向量正交（无相关性）

2. 点积（Dot Product）：向量对应元素相乘后求和，取值范围无限制
   - 值越大表示两个向量越相似
   - 对于归一化向量，点积等价于余弦相似度

3. 欧氏距离（Euclidean Distance）：两点之间的直线距离，取值范围[0, +∞)
   - 值越小表示两个向量越相似
   - 通常通过距离转相似度公式：similarity = 1 / (1 + distance)
"""

# ============================================================================
# 第二部分：相似度计算函数实现
# ============================================================================

import numpy as np
from typing import List, Tuple


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度

    原理：余弦相似度通过计算两个向量夹角的余弦值来衡量它们的相似程度。
    公式：cosine_similarity = (A · B) / (||A|| × ||B||)

    适用场景：
    - 文本嵌入向量的相似度计算
    - 当需要考虑向量方向而非 magnitude 时
    - 文档检索和语义匹配场景

    参数:
        vec1: 第一个向量（numpy数组）
        vec2: 第二个向量（numpy数组）

    返回:
        float: 余弦相似度值，范围[-1, 1]
    """
    # 处理零向量情况
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # 计算余弦相似度：向量点积除以两个向量的模长乘积
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)

    return float(similarity)


def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的点积相似度

    原理：点积是向量对应元素相乘后的累加和。
    公式：dot_product = Σ(Ai × Bi)

    适用场景：
    - 向量已经归一化时，点积等价于余弦相似度
    - 需要考虑向量 magnitude 的场景
    - 大规模向量检索（如ANN算法）

    参数:
        vec1: 第一个向量（numpy数组）
        vec2: 第二个向量（numpy数组）

    返回:
        float: 点积值，无取值范围限制
    """
    return float(np.dot(vec1, vec2))


def euclidean_distance_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    通过欧氏距离计算相似度

    原理：欧氏距离是向量空间中两点之间的直线距离。
    公式：distance = √(Σ(Ai - Bi)²)

    转换公式：similarity = 1 / (1 + distance)
    - 距离为0时，相似度为1（完全相同）
    - 距离越大，相似度越接近0

    适用场景：
    - 需要衡量向量绝对距离的场景
    - 聚类分析
    - 推荐系统中的协同过滤

    参数:
        vec1: 第一个向量（numpy数组）
        vec2: 第二个向量（numpy数组）

    返回:
        float: 相似度值，范围(0, 1]
    """
    distance = np.linalg.norm(vec1 - vec2)
    # 使用转换公式将距离转换为相似度
    similarity = 1.0 / (1.0 + distance)
    return float(similarity)


def similarity_demo():
    """
    演示三种相似度计算方法的使用
    """
    print("=" * 70)
    print("相似度计算演示")
    print("=" * 70)

    # 示例：文本嵌入向量（简化示例，实际应用中通过Embedding模型生成）
    # 假设有三个文档的向量表示
    doc_vectors = {
        "文档A-人工智能": np.array([0.8, 0.6, 0.2, 0.1]),
        "文档B-机器学习": np.array([0.7, 0.8, 0.3, 0.2]),
        "文档C-食谱烹饪": np.array([0.1, 0.2, 0.9, 0.8]),
    }

    # 用户查询向量："AI技术发展"
    query_vector = np.array([0.9, 0.7, 0.1, 0.1])

    print("\n查询向量:", query_vector)
    print("-" * 70)

    # 计算查询向量与每个文档的相似度
    for doc_name, doc_vec in doc_vectors.items():
        cosine = cosine_similarity(query_vector, doc_vec)
        dot = dot_product_similarity(query_vector, doc_vec)
        euclid = euclidean_distance_similarity(query_vector, doc_vec)

        print(f"\n{doc_name}:")
        print(f"  向量: {doc_vec}")
        print(f"  余弦相似度: {cosine:.4f}")
        print(f"  点积: {dot:.4f}")
        print(f"  欧氏距离相似度: {euclid:.4f}")

    # 排序结果
    print("\n" + "=" * 70)
    print("按余弦相似度排序结果:")
    similarities = []
    for doc_name, doc_vec in doc_vectors.items():
        cosine = cosine_similarity(query_vector, doc_vec)
        similarities.append((doc_name, cosine))

    # 按相似度降序排列
    similarities.sort(key=lambda x: x[1], reverse=True)
    for rank, (doc_name, score) in enumerate(similarities, 1):
        print(f"  第{rank}名: {doc_name} (相似度: {score:.4f})")


# ============================================================================
# 第三部分：Top-K检索实现
# ============================================================================

def top_k_retrieval(
    query_vector: np.ndarray,
    document_vectors: List[Tuple[str, np.ndarray]],
    k: int = 5,
    similarity_func=None
) -> List[Tuple[str, float]]:
    """
    实现Top-K检索：返回最相似的前K个文档

    原理：
    1. 计算查询向量与所有文档向量的相似度
    2. 按相似度降序排列
    3. 返回前K个文档

    参数:
        query_vector: 查询向量
        document_vectors: 文档向量列表，每项为(文档ID, 向量)的元组
        k: 需要返回的文档数量
        similarity_func: 相似度计算函数，默认使用余弦相似度

    返回:
        List[Tuple[str, float]]: 前K个最相似的文档列表，每项为(文档ID, 相似度)
    """
    if similarity_func is None:
        similarity_func = cosine_similarity

    # 计算所有文档的相似度
    similarities = []
    for doc_id, doc_vec in document_vectors:
        sim = similarity_func(query_vector, doc_vec)
        similarities.append((doc_id, sim))

    # 按相似度降序排列
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 返回前K个结果
    return similarities[:k]


def threshold_retrieval(
    query_vector: np.ndarray,
    document_vectors: List[Tuple[str, np.ndarray]],
    threshold: float = 0.5,
    similarity_func=None
) -> List[Tuple[str, float]]:
    """
    实现相似度阈值检索：返回所有超过阈值的文档

    原理：
    1. 计算查询向量与所有文档向量的相似度
    2. 过滤掉相似度低于阈值的文档
    3. 返回满足条件的文档（可能不足K个）

    参数:
        query_vector: 查询向量
        document_vectors: 文档向量列表，每项为(文档ID, 向量)的元组
        threshold: 相似度阈值，范围[0, 1]
        similarity_func: 相似度计算函数，默认使用余弦相似度

    返回:
        List[Tuple[str, float]]: 所有超过阈值的文档列表
    """
    if similarity_func is None:
        similarity_func = cosine_similarity

    # 计算所有文档的相似度
    results = []
    for doc_id, doc_vec in document_vectors:
        sim = similarity_func(query_vector, doc_vec)
        if sim >= threshold:
            results.append((doc_id, sim))

    # 按相似度降序排列
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def retrieval_demo():
    """
    演示Top-K检索和阈值检索的使用
    """
    print("\n" + "=" * 70)
    print("Top-K检索与阈值检索演示")
    print("=" * 70)

    # 模拟文档向量库（实际应用中从向量数据库获取）
    documents = [
        ("doc_1_AI", np.array([0.9, 0.8, 0.2, 0.1])),
        ("doc_2_ML", np.array([0.8, 0.9, 0.3, 0.2])),
        ("doc_3_DL", np.array([0.7, 0.8, 0.4, 0.3])),
        ("doc_4_CV", np.array([0.6, 0.5, 0.7, 0.6])),
        ("doc_5_NLP", np.array([0.85, 0.75, 0.25, 0.15])),
        ("doc_6_cooking", np.array([0.1, 0.2, 0.9, 0.8])),
        ("doc_7_travel", np.array([0.15, 0.25, 0.85, 0.75])),
        ("doc_8_sport", np.array([0.2, 0.3, 0.8, 0.7])),
    ]

    # 查询向量
    query = np.array([0.88, 0.82, 0.2, 0.1])

    print(f"\n文档库大小: {len(documents)} 个文档")
    print(f"查询向量: {query}")

    # Top-3检索
    print("\n--- Top-3 检索结果 ---")
    top3_results = top_k_retrieval(query, documents, k=3)
    for rank, (doc_id, score) in enumerate(top3_results, 1):
        print(f"  第{rank}名: {doc_id} (相似度: {score:.4f})")

    # 阈值检索（阈值=0.7）
    print("\n--- 阈值检索（阈值=0.70）---")
    threshold_results = threshold_retrieval(query, documents, threshold=0.70)
    for doc_id, score in threshold_results:
        print(f"  {doc_id}: {score:.4f}")

    # 阈值检索（阈值=0.8）
    print("\n--- 阈值检索（阈值=0.80）---")
    threshold_results = threshold_retrieval(query, documents, threshold=0.80)
    for doc_id, score in threshold_results:
        print(f"  {doc_id}: {score:.4f}")


# ============================================================================
# 第四部分：LlamaIndex检索器实现
# ============================================================================

"""
LlamaIndex提供了多种检索器实现，本节介绍三种核心检索器：

1. VectorIndexRetriever（向量检索器）
   - 基于密集向量检索（Dense Retrieval）
   - 通过计算查询向量与文档向量的余弦相似度来排序
   - 适用于语义相似度匹配，能理解语义和概念

2. BM25Retriever（关键词检索器）
   - 基于稀疏向量检索（Sparse Retrieval）
   - 依赖倒排索引，精确匹配关键词
   - 适用于专业术语、代码片段等精确匹配场景

3. QueryFusionRetriever（融合检索器）
   - 结合多种检索策略（如向量检索+BM25）
   - 通过RRF（Reciprocal Rank Fusion）算法融合结果
   - 兼顾语义召回和关键词精确匹配，是生产环境首选

检索器选择建议：
- 初步探索：从VectorIndexRetriever开始
- 专业领域：对于需要精确匹配的领域，使用BM25Retriever
- 生产环境：推荐使用QueryFusionRetriever实现混合检索
"""


def llama_index_retriever_demo():
    """
    演示LlamaIndex中三种检索器的使用方法

    注意：此函数需要LlamaIndex环境才能运行
    以下代码展示了完整的使用模式，适用于实际项目
    """
    print("\n" + "=" * 70)
    print("LlamaIndex检索器演示（需要LlamaIndex环境）")
    print("=" * 70)

    # 以下为伪代码，展示API调用模式
    """
    # ---------------------------------------------------------------
    # 1. VectorIndexRetriever - 向量检索器
    # ---------------------------------------------------------------
    from llama_index.core import VectorStoreIndex
    from llama_index.core.retrievers import VectorIndexRetriever

    # 假设已有索引和节点
    # index = VectorStoreIndex(nodes)

    # 方式一：使用as_retriever()快捷方法
    retriever = index.as_retriever(similarity_top_k=5)

    # 方式二：直接创建VectorIndexRetriever（获得更多控制）
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,  # 返回前5个最相似结果
        # filters=MetadataFilters(...)  # 可选：元数据过滤
    )

    # 执行检索
    query = "B200A芯片的存储器规格是什么？"
    nodes = vector_retriever.retrieve(query)

    # 打印结果
    for node in nodes:
        print(f"得分: {node.score:.4f}")
        print(f"内容: {node.text[:200]}...")
        print("-" * 60)


    # ---------------------------------------------------------------
    # 2. BM25Retriever - 关键词检索器
    # ---------------------------------------------------------------
    from llama_index.retrievers.bm25 import BM25Retriever

    # BM25基于词频统计，不需要向量索引
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,  # 节点列表
        similarity_top_k=5,  # 返回前5个结果
        # 可以调整BM25参数：
        # language="english"  # 指定语言
    )

    # 执行检索
    results = bm25_retriever.retrieve(query)

    # 打印结果
    for node in results:
        print(f"BM25得分: {node.score:.4f}")
        print(f"内容: {node.text[:200]}...")


    # ---------------------------------------------------------------
    # 3. QueryFusionRetriever - 融合检索器
    # ---------------------------------------------------------------
    from llama_index.core.retrievers import QueryFusionRetriever

    # 创建融合检索器，结合向量检索和BM25
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],  # 组合多个检索器
        similarity_top_k=5,  # 最终返回的文档数量
        num_queries=1,  # 生成的子查询数量（1表示不进行查询扩展）
        mode="reciprocal_rerank",  # 使用RRF融合算法
        use_async=True,  # 异步执行
        verbose=True,  # 打印详细信息
        retriever_weights=[0.6, 0.4],  # 检索器权重（向量:BM25）
    )

    # 执行检索
    query_nodes = fusion_retriever.retrieve(query)

    # 打印结果
    for node in query_nodes:
        print(f"融合得分: {node.score:.4f}")
        print(f"内容: {node.text[:200]}...")
    """
    print("""
    [提示] 以上为LlamaIndex检索器的代码示例
    如需实际运行，请确保已安装:
        pip install llama-index llama-index-retrievers-bm25 rank-bm25

    推荐在Jupyter Notebook环境中运行完整示例
    """)


# ============================================================================
# 第五部分：检索器对比与选型建议
# ============================================================================

def retriever_comparison():
    """
    展示三种检索器的对比和适用场景
    """
    print("\n" + "=" * 70)
    print("检索器类型对比与选型建议")
    print("=" * 70)

    comparison_data = [
        ("VectorIndexRetriever", "向量检索（Dense Retrieval）",
         "语义相似度匹配，泛化能力强",
         "对关键词不敏感，向量构建成本高",
         "初步探索、快速搭建语义检索系统"),

        ("BM25Retriever", "关键词匹配（Sparse Retrieval）",
         "速度快，精确匹配关键词效果好",
         "无法理解语义，同义词问题难处理",
         "专业术语、代码片段、精确匹配场景"),

        ("QueryFusionRetriever", "混合检索（Hybrid）",
         "结合两者优点，召回率和精度更高",
         "配置相对复杂，需要平衡权重",
         "生产环境首选，兼顾语义和关键词"),
    ]

    print(f"\n{'检索器类型':<25} {'原理':<20} {'优点':<25} {'缺点':<20} {'适用场景'}")
    print("-" * 120)

    for row in comparison_data:
        print(f"{row[0]:<25} {row[1]:<20} {row[2]:<25} {row[3]:<20} {row[4]}")

    print("\n" + "=" * 70)
    print("选型建议总结")
    print("=" * 70)
    print("""
    1. 初步探索阶段
       -> 从VectorIndexRetriever开始，快速搭建语义检索系统

    2. 专业领域（法律、医疗、代码等）
       -> 使用BM25Retriever或HybridRetriever
       -> 这些领域需要精确匹配术语

    3. 生产环境
       -> 强烈推荐HybridRetriever（QueryFusionRetriever）
       -> 实现最全面和最精准的召回效果

    4. 混合检索+重排序是黄金组合
       -> 粗召回（HybridRetriever）：Top-K=10~20
       -> 精排（Reranker）：最终Top-N=3~5
       -> 这种管道模式兼顾召回率和精度
    """)


# ============================================================================
# 第六部分：完整RAG检索流程示例
# ============================================================================

def full_rag_retrieval_flow():
    """
    展示完整的RAG检索流程

    完整的RAG检索流程包括：
    1. 文档加载和预处理
    2. 文档分割（Chunking）
    3. 向量化（Embedding）
    4. 索引构建
    5. 检索器创建
    6. 执行检索
    7. 结果后处理（如Rerank）
    """
    print("\n" + "=" * 70)
    print("完整RAG检索流程")
    print("=" * 70)
    print("""
    完整RAG检索流程示例代码结构：

    Step 1: 文档加载
    ----------
    from llama_index.core import Document
    documents = [Document(text="文档内容...")]

    Step 2: 文档分割
    ----------
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    Step 3: 向量化
    ----------
    from llama_index.core import VectorStoreIndex
    from llama_index.core import Settings
    # 配置Embedding模型（默认OpenAI）
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    Step 4: 索引构建
    ----------
    index = VectorStoreIndex(nodes)

    Step 5: 创建检索器
    ----------
    # 方式一：快捷方法
    retriever = index.as_retriever(similarity_top_k=5)

    # 方式二：混合检索
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever

    vector_retriever = index.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        mode="reciprocal_rerank",
        retriever_weights=[0.6, 0.4]
    )

    Step 6: 执行检索
    ----------
    query = "用户问题"
    results = retriever.retrieve(query)

    Step 7: 结果后处理（Rerank）
    ----------
    from llama_index.core.postprocessor import SentenceTransformerRerank

    reranker = SentenceTransformerRerank(
        model="/path/to/reranker/model",
        top_n=3  # 只保留前3个最相关结果
    )

    reranked_nodes = reranker.postprocess(results)
    """)


# ============================================================================
# 主函数入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("第06节 检索器类型与相似度计算 - 代码演示")
    print("=" * 70)

    # 1. 相似度计算演示
    similarity_demo()

    # 2. Top-K检索和阈值检索演示
    retrieval_demo()

    # 3. 检索器对比
    retriever_comparison()

    # 4. 完整RAG检索流程
    full_rag_retrieval_flow()

    # 5. LlamaIndex检索器说明
    llama_index_retriever_demo()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("""
    下一步学习：
    - 学习混合检索与重排序（第11节）
    - 使用真实数据运行完整的RAG流程
    - 调整similarity_top_k参数观察效果变化
    """)
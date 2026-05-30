"""
混合检索与重排序策略 - 混合检索实现
=====================================

本文件演示如何在LlamaIndex中实现三种检索策略：
1. BM25（稀疏检索）- 基于关键词的精确匹配
2. 密集检索（Dense Retrieval）- 基于向量的语义相似度
3. 混合检索（Hybrid Search）- 融合前两种方法的优势

学习目标：
- 理解向量检索与关键词检索的互补原理
- 掌握BM25Retriever、VectorIndexRetriever的使用方法
- 掌握QueryFusionRetriever进行结果融合

依赖安装：
    pip install llama-index llama-index-retrievers-bm25 rank-bm25
"""

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Node
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding

# ============================================================
# 第一部分：准备文档和节点
# ============================================================

# 示例文档集合（实际应用中应从文件/数据库加载）
sample_documents = [
    Document(text="英伟达计划在2024年下半年推出B100和B200 GPU，主要供应云端服务商客户。B200A采用4颗HBM3e存储器，总容量144GB。"),
    Document(text="特斯拉Optimus机器人正在快速迭代，二代产品已在WAIC亮相，展现了先进的运动控制和感知能力。"),
    Document(text="苹果公司正在研发桌面机器人，计划于2026年推出，配备AI芯片和智能助手功能。"),
    Document(text="OpenAI发布了GPT-5模型，在推理能力和多模态理解方面有显著提升。"),
    Document(text="谷歌发布Gemini Ultra 2.0，在长上下文理解方面达到100M tokens。"),
]

def create_nodes(documents: list[Document]) -> list[Node]:
    """
    将文档解析为节点

    Args:
        documents: 原始文档列表

    Returns:
        nodes: 解析后的节点列表
    """
    # 使用简单节点解析器，chunk_size=100（实际应用中根据需求调整）
    node_parser = SimpleNodeParser.from_defaults(chunk_size=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes

# 创建节点
nodes = create_nodes(sample_documents)
print(f"创建了 {len(nodes)} 个节点")

# ============================================================
# 第二部分：BM25检索器（稀疏检索）
# ============================================================

def create_bm25_retriever(nodes: list[Node]) -> BM25Retriever:
    """
    创建BM25检索器

    BM25（Best Matching 25）是一种经典的关键词检索算法，
    广泛应用于搜索引擎中。它基于倒排索引结构，通过统计
    词频（TF）和逆文档频率（IDF）来评估文档与查询的相关性。

    BM25的优势：
    - 对精确关键词匹配效果很好
    - 对于专业术语、型号、代码等能精确召回
    - 计算速度快，不依赖复杂的深度学习模型

    BM25的局限：
    - 无法理解语义（如同义词、多义词）
    - 对拼写变体敏感

    Args:
        nodes: 节点列表

    Returns:
        BM25Retriever实例
    """
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,           # 节点集合，用于构建倒排索引
        similarity_top_k=3,    # 返回最相关的3个节点
        # 以下参数可调整：
        # lowercase: 是否将文本转为小写（默认True）
        # w: 词频权重（默认1.0）
        # k1: 词频饱和参数（默认1.5）
        # b: 文档长度归一化参数（默认0.75）
    )
    return bm25_retriever

# 创建BM25检索器
bm25_retriever = create_bm25_retriever(nodes)
print("BM25检索器创建成功")

# 示例查询
bm25_query = "英伟达B200A的存储器规格"

# 执行BM25检索
bm25_results = bm25_retriever.retrieve(bm25_query)
print(f"\n=== BM25检索结果 (查询: '{bm25_query}') ===")
for i, node in enumerate(bm25_results):
    print(f"  [{i+1}] Score: {node.score:.4f}")
    print(f"      Content: {node.get_content()[:50]}...")

# ============================================================
# 第三部分：密集检索器（向量检索）
# ============================================================

def create_vector_retriever(nodes: list[Node]) -> any:
    """
    创建向量检索器

    密集检索（Dense Retrieval）使用嵌入模型将文本转换为向量，
    然后通过向量相似度（如余弦相似度）来评估文档与查询的相关性。

    密集检索的优势：
    - 能够理解语义相似性（如同义词、 paraphrase）
    - 对拼写错误有一定容忍度
    - 能捕捉深层的语义关联

    密集检索的局限：
    - 对精确关键词匹配不如BM25
    - 需要高效的向量索引（如FAISS、Milvus）
    - 计算成本较高

    Args:
        nodes: 节点列表

    Returns:
        VectorIndexRetriever实例
    """
    from llama_index.core import VectorStoreIndex

    # 创建向量索引
    # Settings.embed_model用于配置嵌入模型（默认OpenAIEmbedding）
    # 实际应用中可根据需求使用本地模型（如BGE、Sentence-Transformers）
    vector_index = VectorStoreIndex(nodes)

    # 创建向量检索器
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=3   # 返回最相关的3个节点
    )
    return vector_retriever

# 创建向量检索器
vector_retriever = create_vector_retriever(nodes)
print("\n向量检索器创建成功")

# 示例查询
vector_query = "英伟达B200A的存储器规格"

# 执行向量检索
vector_results = vector_retriever.retrieve(vector_query)
print(f"\n=== 向量检索结果 (查询: '{vector_query}') ===")
for i, node in enumerate(vector_results):
    print(f"  [{i+1}] Score: {node.score:.4f}")
    print(f"      Content: {node.get_content()[:50]}...")

# ============================================================
# 第四部分：混合检索（融合策略）
# ============================================================

def create_hybrid_retriever(
    vector_retriever: any,
    bm25_retriever: BM25Retriever
) -> QueryFusionRetriever:
    """
    创建混合检索器

    混合检索结合了密集检索和稀疏检索的优点：
    1. 向量检索：捕捉语义相似性
    2. BM25检索：精确匹配关键词

    QueryFusionRetriever使用"Reciprocal Rank Fusion"(RRF)算法
    对多个检索器的结果进行融合排序。

    RRF的核心思想：
    - 如果一个文档在多个检索器中都排名靠前，说明它更相关
    - 通过计算 1/(rank + k) 的和来进行融合（k为平滑参数）
    - 这种方法可以有效平衡不同检索器的得分尺度差异

    Args:
        vector_retriever: 向量检索器
        bm25_retriever: BM25检索器

    Returns:
        QueryFusionRetriever实例
    """
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[
            vector_retriever,    # 第一个检索器：向量检索
            bm25_retriever       # 第二个检索器：BM25检索
        ],
        similarity_top_k=5,      # 最终返回的文档数量
        num_queries=1,          # 生成查询变体数量（1表示不生成）
        mode="reciprocal_rerank",  # 使用RRF融合模式
        use_async=True,         # 是否异步执行
        verbose=True,           # 是否打印调试信息
        # retriever_weights: 可以为不同检索器设置权重（默认均等）
    )
    return hybrid_retriever

# 创建混合检索器
hybrid_retriever = create_hybrid_retriever(vector_retriever, bm25_retriever)
print("\n混合检索器创建成功")

# 示例查询
hybrid_query = "英伟达B200A的存储器规格"

# 执行混合检索
print(f"\n=== 混合检索结果 (查询: '{hybrid_query}') ===")
hybrid_results = hybrid_retriever.retrieve(hybrid_query)
for i, node in enumerate(hybrid_results):
    print(f"  [{i+1}] Score: {node.score:.4f}")
    print(f"      Content: {node.get_content()[:60]}...")

# ============================================================
# 第五部分：Retriever权重调整
# ============================================================

def create_weighted_hybrid_retriever(
    vector_retriever: any,
    bm25_retriever: BM25Retriever,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> QueryFusionRetriever:
    """
    创建带权重的混合检索器

    可以通过retriever_weights参数为不同检索器设置权重，
    使其在融合结果中占据更大或更小的比例。

    权重设置建议：
    - 语义复杂、意图模糊的查询：提高向量检索权重
    - 专业术语多、需要精确匹配的查询：提高BM25权重

    Args:
        vector_retriever: 向量检索器
        bm25_retriever: BM25检索器
        vector_weight: 向量检索权重
        bm25_weight: BM25检索权重

    Returns:
        加权混合检索器
    """
    weighted_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        retriever_weights=[vector_weight, bm25_weight]  # 设置检索器权重
    )
    return weighted_retriever

# ============================================================
# 第六部分：多种融合模式对比
# ============================================================

def compare_fusion_modes(vector_retriever, bm25_retriever):
    """
    对比不同的结果融合模式

    QueryFusionRetriever支持多种融合模式：
    1. reciprocal_rerank（推荐）：使用RRF算法，平衡不同检索器的排名
    2. distributed：分散融合（需要多个查询变体）
    3. re reciprocal_rerank_fusion：RRF的另一种实现

    实际应用中，reciprocal_rerank是最常用且效果稳定的模式。
    """
    query = "英伟达GPU的最新消息"

    # 模式1：reciprocal_rerank（推荐）
    retriever_rrf = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        mode="reciprocal_rerank",
        verbose=False
    )
    results_rrf = retriever_rrf.retrieve(query)
    print(f"\n=== RRF模式结果 (top-3) ===")
    for i, node in enumerate(results_rrf[:3]):
        print(f"  [{i+1}] Score: {node.score:.4f} - {node.get_content()[:40]}...")

# ============================================================
# 主函数：演示完整流程
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("混合检索演示程序")
    print("=" * 60)

    # 1. BM25检索演示
    print("\n--- BM25检索演示 ---")
    bm25_query = "英伟达B200A"
    bm25_results = bm25_retriever.retrieve(bm25_query)
    for i, node in enumerate(bm25_results):
        print(f"  [{i+1}] {node.get_content()[:50]}... (score={node.score:.4f})")

    # 2. 向量检索演示
    print("\n--- 向量检索演示 ---")
    vector_query = "英伟达B200A的存储器规格"
    vector_results = vector_retriever.retrieve(vector_query)
    for i, node in enumerate(vector_results):
        print(f"  [{i+1}] {node.get_content()[:50]}... (score={node.score:.4f})")

    # 3. 混合检索演示
    print("\n--- 混合检索演示 ---")
    hybrid_query = "英伟达B200A芯片规格"
    hybrid_results = hybrid_retriever.retrieve(hybrid_query)
    for i, node in enumerate(hybrid_results):
        print(f"  [{i+1}] {node.get_content()[:50]}... (score={node.score:.4f})")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
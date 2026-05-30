"""
混合检索与重排序策略 - 重排序（Rerank）实现
=============================================

本文件演示如何使用重排序模型（如BGE-Reranker）对检索结果进行精排。

核心概念：
- 召回阶段（Recall）：快速从大规模文档中筛选候选文档
- 精排阶段（Rerank）：对候选文档进行深度相关性评估

重排序的价值：
- 不同于向量检索的"粗排"，Reranker进行深度的一对一评估
- 能精准识别语义相关但表面不相似的文档
- 显著提升最终结果的质量

依赖安装：
    pip install llama-index llama-index-postprocessor-reranker
    # 模型下载（需要huggingface或modelscope）
    # modelscope download --model BAAI/bge-reranker-base --local_dir ./bge_reranker
"""

from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Node, TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# ============================================================
# 第一部分：重排序（Rerank）原理
# ============================================================

"""
重排序的工作机制：

1. 召回阶段（向量检索/BM25）：
   - 快速扫描大规模文档库
   - 基于向量相似度或关键词匹配筛选Top-K候选（如K=20）
   - 速度快但精度相对粗糙

2. 精排阶段（Reranker）：
   - 对召回的K个候选进行深度评估
   - 使用交叉编码器（Cross-Encoder）同时输入查询和文档
   - 计算精细的相关性分数
   - 输出最终排序的Top-N结果（如N=5）

交叉编码器 vs 双编码器：
- 双编码器（Bi-Encoder）：分别编码查询和文档，然后计算相似度
  - 优点：速度快，可预先计算文档向量
  - 缺点：无法捕捉查询和文档之间的深度交互

- 交叉编码器（Cross-Encoder）：同时输入查询和文档，通过注意力机制
  - 优点：能深度理解查询和文档的交互关系
  - 缺点：计算成本高，需要实时计算

常见的Reranker类型：
1. Cross-Encoder Reranker（如BGE-Reranker）：精度最高，推荐使用
2. LLM-as-Judge Reranker：利用大语言模型判断相关性，灵活但成本高
3. API-Based Reranker（如Cohere、Jina）：云服务，开箱即用
"""

# ============================================================
# 第二部分：准备测试数据
# ============================================================

# 示例文档（模拟实际的金融研报场景）
sample_documents = [
    Document(text="英伟达计划在2024年下半年推出B100和B200 GPU，主要供应云端服务商客户。B200A采用4颗HBM3e存储器，总容量144GB。"),
    Document(text="特斯拉Optimus机器人正在快速迭代，二代产品已在WAIC亮相，展现了先进的运动控制和感知能力。"),
    Document(text="苹果公司正在研发桌面机器人，计划于2026年推出，配备AI芯片和智能助手功能。"),
    Document(text="OpenAI发布了GPT-5模型，在推理能力和多模态理解方面有显著提升。"),
    Document(text="谷歌发布Gemini Ultra 2.0，在长上下文理解方面达到100M tokens。"),
    Document(text="英伟达Blackwell架构的B200系列是下一代AI计算平台，取代Hopper架构。"),
    Document(text="HBM3e是第五代高带宽内存，比HBM3容量更大、带宽更高。"),
]

def create_nodes(documents: list[Document]) -> list[Node]:
    """
    将文档解析为节点

    Args:
        documents: 原始文档列表

    Returns:
        nodes: 解析后的节点列表
    """
    node_parser = SimpleNodeParser.from_defaults(chunk_size=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes

# 创建节点
nodes = create_nodes(sample_documents)
print(f"创建了 {len(nodes)} 个节点")

# ============================================================
# 第三部分：创建基础检索器
# ============================================================

from llama_index.core import VectorStoreIndex

def create_base_retrievers(nodes: list[Node]):
    """
    创建基础检索器组合（用于后续重排序）

    为了演示完整的"召回->精排"流程，我们先创建：
    1. 向量检索器
    2. BM25检索器（可选）
    3. 混合检索器

    Args:
        nodes: 节点列表

    Returns:
        包含各检索器的字典
    """
    from llama_index.retrievers.bm25 import BM25Retriever

    # 创建向量索引和检索器
    vector_index = VectorStoreIndex(nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=10)  # 召回较多候选

    # 创建BM25检索器
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10
    )

    # 创建混合检索器
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=10,  # 扩大召回范围，为Reranker提供更多候选
        mode="reciprocal_rerank",
        use_async=True,
        verbose=False
    )

    return {
        "vector": vector_retriever,
        "bm25": bm25_retriever,
        "hybrid": hybrid_retriever
    }

# 创建检索器
retrievers = create_base_retrievers(nodes)
print("基础检索器创建成功")

# ============================================================
# 第四部分：创建Reranker（精排器）
# ============================================================

def create_bge_reranker(model_path: str = None) -> SentenceTransformerRerank:
    """
    创建BGE-Reranker精排器

    BGE-Reranker是北京智源人工智能研究院开发的对映模型，
    用于对检索结果进行精细排序。

    工作原理：
    - 输入：查询 + 候选文档
    - 输出：每个文档的相关性分数（0-1之间）
    - 分数越高表示越相关

    参数说明：
    - model: 模型路径或模型名称
    - top_n: 最终返回的文档数量
    - score_threshold: 分数阈值，低于该阈值的文档会被过滤

    Args:
        model_path: 模型本地路径，如 "/path/to/bge-reranker-base"

    Returns:
        SentenceTransformerRerank实例
    """
    # 如果没有本地模型路径，可以使用模型名称从HF加载
    # 实际应用中，请先下载模型：
    # modelscope download --model BAAI/bge-reranker-base --local_dir ./bge_reranker

    if model_path is None:
        # 使用默认模型（需要网络连接下载）
        model_path = "BAAI/bge-reranker-base"

    reranker = SentenceTransformerRerank(
        model=model_path,           # 模型路径或名称
        top_n=3,                     # 精排后返回3个最相关文档
        score_threshold=0.3,         # 分数阈值，低于0.3的文档不返回
        # device: "cpu" 或 "cuda"（根据硬件选择）
    )

    return reranker

# 创建Reranker（请确保模型已下载或网络可用）
# 实际使用时，将model_path替换为本地模型路径
try:
    # 尝试创建Reranker，如果模型不存在会失败
    reranker = create_bge_reranker()
    print("BGE-Reranker创建成功")
except Exception as e:
    print(f"Reranker创建提示: {e}")
    print("注意：请先下载BGE-Reranker模型或配置正确的模型路径")
    reranker = None

# ============================================================
# 第五部分：构建带Reranker的QueryEngine
# ============================================================

def build_reranked_query_engine(retriever, reranker) -> RetrieverQueryEngine:
    """
    构建带重排序功能的QueryEngine

    QueryEngine是LlamaIndex的核心组件，它封装了：
    1. 检索器（Retriever）：从索引中召回候选文档
    2. 后处理器（Postprocessor）：对候选进行精排、过滤等
    3. 响应合成器（Response Synthesizer）：整合上下文生成答案

    典型的RAG流程：
    Query -> Retriever(召回Top-K) -> Postprocessor(Rerank精排Top-N) -> LLM生成答案

    Args:
        retriever: 检索器
        reranker: 精排器

    Returns:
        配置好的QueryEngine
    """
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker]  # 注入Reranker
    )
    return query_engine

# ============================================================
# 第六部分：演示无Reranker vs 有Reranker的对比
# ============================================================

def compare_with_without_reranker(query: str, retriever, reranker=None):
    """
    对比有Reranker和无Reranker的检索效果

    这个对比展示了Reranker的价值：
    - 无Reranker：直接使用向量相似度排序
    - 有Reranker：对候选进行深度评估后重新排序

    Args:
        query: 查询字符串
        retriever: 检索器
        reranker: 精排器（可选）
    """
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")

    # 1. 无Reranker的检索结果
    print("\n--- 无Reranker（直接召回）---")
    raw_results = retriever.retrieve(query)
    for i, node in enumerate(raw_results[:5]):
        print(f"  [{i+1}] Score={node.score:.4f} | {node.get_content()[:50]}...")

    # 2. 有Reranker的检索结果
    if reranker:
        print("\n--- 有Reranker（精排后）---")
        # 使用postprocess_nodes方法对结果进行精排
        reranked_results = reranker.postprocess_nodes(raw_results, query_string=query)
        for i, node in enumerate(reranked_results):
            print(f"  [{i+1}] Score={node.score:.4f} | {node.get_content()[:50]}...")

# ============================================================
# 第七部分：完整流程演示
# ============================================================

def demonstrate_reranking_flow():
    """
    演示完整的"混合检索 + 重排序"流程

    流程说明：
    1. 使用混合检索器召回Top-20候选
    2. 使用Reranker对候选进行精排
    3. 返回Top-5最相关结果
    """
    print("\n" + "=" * 60)
    print("完整流程演示：混合检索 + Reranker精排")
    print("=" * 60)

    # 使用混合检索器
    hybrid_retriever = retrievers["hybrid"]

    # 演示查询
    test_queries = [
        "英伟达B200A的存储器规格是什么？",
        "AI芯片的最新进展",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")

        # 步骤1：混合检索（召回阶段）
        print("  [步骤1] 混合检索召回候选...")
        candidates = hybrid_retriever.retrieve(query)
        print(f"         召回 {len(candidates)} 个候选文档")

        # 步骤2：Reranker精排
        if reranker:
            print("  [步骤2] Reranker精排...")
            refined = reranker.postprocess_nodes(candidates, query_string=query)
            print(f"         精排后保留 {len(refined)} 个文档")

            # 打印最终结果
            print("\n  最终结果（Top-3）：")
            for i, node in enumerate(refined[:3]):
                print(f"    {i+1}. {node.get_content()[:60]}...")
        else:
            print("  [步骤2] Reranker未配置，跳过精排")
            print("\n  最终结果（Top-3）：")
            for i, node in enumerate(candidates[:3]):
                print(f"    {i+1}. {node.get_content()[:60]}...")

# ============================================================
# 第八部分：其他Reranker类型简介
# ============================================================

def introduce_other_rerankers():
    """
    介绍其他类型的Reranker，帮助理解不同方案的特点
    """
    print("\n" + "=" * 60)
    print("Reranker类型对比")
    print("=" * 60)

    reranker_types = """
    1. Cross-Encoder Reranker（如BGE-Reranker）
       - 原理：同时输入查询和文档，通过深度注意力交互评估相关性
       - 优点：精度最高，能深刻理解语义关联
       - 缺点：计算成本高，需要GPU支持
       - 适用场景：对精度要求高的生产环境

    2. LLM-as-Judge Reranker
       - 原理：使用大语言模型判断文档与查询的相关性
       - 优点：灵活性强，可利用大模型的推理能力
       - 缺点：延迟高、成本高、结果可能不一致
       - 适用场景：需要复杂推理的相关性判断

    3. API-Based Reranker（如Cohere Rerank、Jina Reranker）
       - 原理：调用云服务提供的重排序API
       - 优点：开箱即用，无需本地计算资源
       - 缺点：产生API费用，依赖网络连接
       - 适用场景：追求精度与性能平衡的生产环境

    4. ColBERT Reranker
       - 原理：基于延迟交互的向量相似度匹配
       - 优点：比Cross-Encoder快，精度也不错
       - 缺点：需要专门的模型支持
       - 适用场景：对延迟有一定要求的场景
    """
    print(reranker_types)

# ============================================================
# 第九部分：最佳实践建议
# ============================================================

def best_practices():
    """
    使用Reranker的最佳实践建议
    """
    practices = """
    ============================================================
    Reranker最佳实践
    ============================================================

    1. 召回数量（similarity_top_k）与精排数量（top_n）的比例
       - 推荐比例：10:3 或 20:5
       - 召回数太少可能漏掉关键信息
       - 太多会增加Reranker的计算负担

    2. 模型选择
       - BGE-Reranker-base：效果与速度的平衡选择
       - BGE-Reranker-large：精度更高，速度更慢
       - 本地部署 vs 云服务：根据资源情况选择

    3. 与混合检索的结合
       - 最佳模式：HybridRetriever + Reranker
       - 先通过混合检索召回多路结果（向量+关键词）
       - 再通过Reranker对混合结果进行精排

    4. 分数阈值调整
       - 根据实际效果调整 score_threshold
       - 过低会引入不相关的文档
       - 过高可能过滤掉有用的信息

    5. 性能优化
       - Reranker是性能瓶颈，尽量减少调用频率
       - 可以考虑缓存频繁查询的结果
       - 批处理多个查询提高吞吐量
    """
    print(practices)

# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("重排序（Rerank）演示程序")
    print("=" * 60)

    # 演示基本对比
    query = "英伟达B200A的存储器规格"
    if reranker:
        compare_with_without_reranker(query, retrievers["hybrid"], reranker)
    else:
        compare_with_without_reranker(query, retrievers["hybrid"], None)

    # 演示完整流程
    demonstrate_reranking_flow()

    # 介绍其他Reranker类型
    introduce_other_rerankers()

    # 最佳实践
    best_practices()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
"""
模块化RAG架构设计 - 模块化RAG核心组件演示
=============================================

本文件演示模块化RAG架构的三大核心组件：
1. Dataflow（数据流）- 文档的加载、解析、索引构建流程
2. Router（路由器）- 根据查询类型选择不同的检索路径
3. Query Pipeline（查询管道）- 完整的检索-生成流程编排

学习目标：
- 理解模块化RAG将系统拆分为独立模块的设计思想
- 掌握Router、QueryPipeline组件的使用方法
- 能够设计灵活组合的模块化RAG架构

依赖安装：
    pip install llama-index llama-index-retrievers-bm25 llama-index-postprocessor-rerank

模块化RAG架构优势：
1. 独立模块可单独优化和替换
2. 支持多种检索器灵活组合
3. 便于针对不同场景定制流程
4. 提高系统可维护性和可扩展性
"""

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Node, QueryBundle
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.query_pipeline import (
    QueryPipeline,
    InputComponent,
    FnComponent,
    RetrieverComponent,
    QueryComponent,
    CustomQueryComponent
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import CompactResponseSynthesizer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.storage import StorageContext
from llama_index.core import VectorStoreIndex
from typing import List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# ============================================================
# 第一部分：示例数据和工具函数
# ============================================================

# 示例文档集合（模拟企业知识库）
SAMPLE_DOCUMENTS = [
    Document(text="英伟达计划在2024年下半年推出B100和B200 GPU，主要供应云端服务商客户。B200A采用4颗HBM3e存储器，总容量144GB，传输速度达8TB/s。"),
    Document(text="特斯拉Optimus机器人正在快速迭代，二代产品已在WAIC亮相，展现了先进的运动控制和感知能力，预计2025年实现量产。"),
    Document(text="苹果公司正在研发桌面机器人，计划于2026年推出，配备AI芯片和智能助手功能，面向家庭和办公场景。"),
    Document(text="OpenAI发布了GPT-5模型，在推理能力和多模态理解方面有显著提升，支持128K上下文窗口。"),
    Document(text="谷歌发布Gemini Ultra 2.0，在长上下文理解方面达到100M tokens，刷新了业界纪录。"),
    Document(text="RAG技术结合了检索系统和生成模型的优势，能够在保证答案准确性的同时提供流畅的自然语言输出。"),
    Document(text="向量数据库是RAG系统的核心组件，负责存储文档的向量表示并提供高效的相似度检索能力。"),
    Document(text="常见的向量数据库包括ChromaDB、FAISS、Milvus、Pinecone等，各有其适用场景和优势。"),
]

def create_nodes(documents: List[Document], chunk_size: int = 100) -> List[Node]:
    """
    将文档解析为节点（Chunk）

    Args:
        documents: 原始文档列表
        chunk_size: 每个节点的最大字符数

    Returns:
        nodes: 解析后的节点列表
    """
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes


def create_vector_index(nodes: List[Node]) -> VectorStoreIndex:
    """
    创建向量索引

    Args:
        nodes: 节点列表

    Returns:
        VectorStoreIndex: 向量索引对象
    """
    index = VectorStoreIndex(nodes)
    return index


# 创建节点和索引
print("=" * 60)
print("初始化：创建节点和向量索引")
print("=" * 60)
nodes = create_nodes(SAMPLE_DOCUMENTS)
print(f"创建了 {len(nodes)} 个节点")

vector_index = create_vector_index(nodes)
print("向量索引创建成功")

# ============================================================
# 第二部分：Dataflow（数据流）
# ============================================================

class DocumentDataflow:
    """
    文档数据流处理器

    Dataflow负责管理文档从原始输入到可检索状态的完整处理流程。
    在模块化RAG中，Dataflow是一个独立的可配置管道，支持：
    - 多种文档格式的加载和解析
    - 文档清洗和预处理
    - 节点分割策略配置
    - 向量化处理
    - 索引构建和更新

    设计原则：
    1. 单一职责：每个组件只负责一个明确的任务
    2. 可插拔：组件可以灵活替换和组合
    3. 可观测：每个步骤都有日志和状态跟踪
    """

    def __init__(self, chunk_size: int = 100):
        """
        初始化数据流处理器

        Args:
            chunk_size: 节点分割大小
        """
        self.chunk_size = chunk_size
        self.documents: List[Document] = []
        self.nodes: List[Node] = []
        self.index: Optional[VectorStoreIndex] = None

    def load_documents(self, docs: List[Document]) -> "DocumentDataflow":
        """
        加载文档

        Args:
            docs: 文档列表

        Returns:
            self: 支持链式调用
        """
        self.documents = docs
        print(f"[Dataflow] 加载了 {len(docs)} 个文档")
        return self

    def parse_documents(self) -> "DocumentDataflow":
        """
        解析文档为节点

        将文档分割成较小的Chunk，以便于检索和精准匹配。

        Returns:
            self: 支持链式调用
        """
        parser = SimpleNodeParser.from_defaults(chunk_size=self.chunk_size)
        self.nodes = parser.get_nodes_from_documents(self.documents)
        print(f"[Dataflow] 解析得到 {len(self.nodes)} 个节点 (chunk_size={self.chunk_size})")
        return self

    def build_index(self) -> "DocumentDataflow":
        """
        构建向量索引

        将节点转换为向量并存储在索引中，支持高效的相似度检索。

        Returns:
            self: 支持链式调用
        """
        self.index = VectorStoreIndex(self.nodes)
        print(f"[Dataflow] 向量索引构建完成，包含 {len(self.nodes)} 个向量")
        return self

    def run(self, docs: List[Document]) -> VectorStoreIndex:
        """
        执行完整的数据流管道

        Args:
            docs: 文档列表

        Returns:
            VectorStoreIndex: 构建好的向量索引
        """
        return self.load_documents(docs).parse_documents().build_index().index


# 演示Dataflow的使用
print("\n" + "=" * 60)
print("Dataflow演示：文档数据流处理")
print("=" * 60)

dataflow = DocumentDataflow(chunk_size=100)
index = dataflow.run(SAMPLE_DOCUMENTS)
print(f"Dataflow执行完成，索引对象类型: {type(index)}")


# ============================================================
# 第三部分：Router（路由器）
# ============================================================

class QueryType(Enum):
    """查询类型枚举"""
    SEMANTIC = "semantic"           # 语义理解类查询
    EXACT_MATCH = "exact_match"     # 精确匹配类查询
    HYBRID = "hybrid"               # 混合类型查询
    GENERAL = "general"             # 通用/闲聊类查询


@dataclass
class RetrievalPath:
    """检索路径配置"""
    name: str                       # 路径名称
    retriever: BaseRetriever         # 检索器
    description: str                 # 路径描述
   适用场景: str                      # 适用场景描述


class Router:
    """
    查询路由器

    Router是模块化RAG的核心组件，负责根据查询特征选择最合适的检索路径。
    它实现了"智能路由"的思想：不同类型的查询需要不同的检索策略。

    路由器的工作原理：
    1. 接收用户查询
    2. 分析查询特征（关键词密度、语义复杂度、问题类型等）
    3. 选择最优的检索路径或路径组合
    4. 将查询传递给选定的检索器

    路由策略类型：
    - 基于规则：关键词匹配、查询长度等
    - 基于模型：使用小模型进行零样本分类
    - 基于向量：计算查询与各路径描述的相似度
    """

    def __init__(self):
        """初始化路由器"""
        self.paths: List[RetrievalPath] = []

    def add_path(self, path: RetrievalPath) -> "Router":
        """
        添加检索路径

        Args:
            path: 检索路径配置

        Returns:
            self: 支持链式调用
        """
        self.paths.append(path)
        print(f"[Router] 添加检索路径: {path.name} - {path.description}")
        return self

    def route(self, query: str) -> QueryType:
        """
        根据查询内容判断查询类型

        简单的基于规则的路由实现。实际应用中可使用更复杂的
        基于模型或向量的路由策略。

        Args:
            query: 用户查询

        Returns:
            QueryType: 查询类型
        """
        # 统计查询特征
        query_lower = query.lower()
        has_exact_terms = any(term in query_lower for term in [
            "型号", "规格", "版本", "参数", "多少", "如何", "怎么"
        ])
        has_technical_terms = any(term in query_lower for term in [
            "GPU", "CPU", "AI", "RAG", "向量", "检索"
        ])

        # 判断查询类型
        if has_exact_terms or has_technical_terms:
            if has_exact_terms:
                return QueryType.EXACT_MATCH
            else:
                return QueryType.HYBRID
        else:
            return QueryType.GENERAL

    def select_retriever(self, query: str) -> BaseRetriever:
        """
        根据查询类型选择检索器

        Args:
            query: 用户查询

        Returns:
            BaseRetriever: 选定的检索器
        """
        query_type = self.route(query)
        print(f"[Router] 查询类型: {query_type.value}")

        if query_type == QueryType.EXACT_MATCH:
            # 精确匹配类查询使用混合检索
            for path in self.paths:
                if path.name == "hybrid":
                    return path.retriever
        elif query_type == QueryType.HYBRID:
            # 混合类查询使用混合检索
            for path in self.paths:
                if path.name == "hybrid":
                    return path.retriever
        else:
            # 通用类查询使用语义检索
            for path in self.paths:
                if path.name == "semantic":
                    return path.retriever

        # 默认返回第一个路径
        return self.paths[0].retriever if self.paths else None


# 创建检索器
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)
vector_retriever = vector_index.as_retriever(similarity_top_k=3)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    mode="reciprocal_rerank"
)

# 创建路由器
print("\n" + "=" * 60)
print("Router演示：查询路由")
print("=" * 60)

router = Router()
router.add_path(RetrievalPath(
    name="semantic",
    retriever=vector_retriever,
    description="语义检索路径",
    适用场景="通用问答、闲聊类查询"
))
router.add_path(RetrievalPath(
    name="hybrid",
    retriever=hybrid_retriever,
    description="混合检索路径",
    适用场景="需要精确匹配的技术问题"
))

# 测试路由
test_queries = [
    "英伟达B200的存储器规格是什么？",
    "介绍一下最新的AI模型进展",
    "特斯拉机器人有什么新消息",
]

for query in test_queries:
    print(f"\n查询: '{query}'")
    retriever = router.select_retriever(query)
    results = retriever.retrieve(query)
    print(f"检索到 {len(results)} 个结果")


# ============================================================
# 第四部分：Query Pipeline（查询管道）
# ============================================================

class QueryPipelineBuilder:
    """
    查询管道构建器

    Query Pipeline是模块化RAG的执行引擎，负责将多个组件
    （检索器、后处理器、合成器等）串联成完整的查询流程。

    Pipeline的优势：
    1. 声明式配置：通过代码或配置文件定义流程
    2. 组件重用：同一组件可在多个Pipeline中使用
    3. 流程可视化：便于调试和优化
    4. 条件分支：支持基于查询特征的动态路由

    典型Pipeline结构：
    Input -> Retriever -> Postprocessor -> Synthesizer -> Output
    """

    def __init__(self):
        """初始化Pipeline构建器"""
        self.components = {}
        self.edges = []

    def add_input(self, name: str = "input") -> "QueryPipelineBuilder":
        """
        添加输入组件

        Args:
            name: 组件名称

        Returns:
            self: 支持链式调用
        """
        self.components[name] = InputComponent()
        return self

    def add_retriever(
        self,
        name: str,
        retriever: BaseRetriever
    ) -> "QueryPipelineBuilder":
        """
        添加检索器组件

        Args:
            name: 组件名称
            retriever: 检索器实例

        Returns:
            self: 支持链式调用
        """
        self.components[name] = RetrieverComponent(retriever)
        return self

    def add_synthesizer(
        self,
        name: str,
        synthesizer: CompactResponseSynthesizer
    ) -> "QueryPipelineBuilder":
        """
        添加响应合成器组件

        Args:
            name: 组件名称
            synthesizer: 合成器实例

        Returns:
            self: 支持链式调用
        """
        self.components[name] = synthesizer
        return self

    def add_custom_component(
        self,
        name: str,
        func: Callable,
        component_type: str = "query"
    ) -> "QueryPipelineBuilder":
        """
        添加自定义组件

        Args:
            name: 组件名称
            func: 处理函数
            component_type: 组件类型 ("query", "retriever", "postprocessor")

        Returns:
            self: 支持链式调用
        """
        if component_type == "query":
            self.components[name] = FnComponent(fn=func)
        return self

    def add_edge(
        self,
        from_component: str,
        to_component: str
    ) -> "QueryPipelineBuilder":
        """
        添加连接边

        Args:
            from_component: 起始组件名称
            to_component: 目标组件名称

        Returns:
            self: 支持链式调用
        """
        self.edges.append((from_component, to_component))
        return self

    def build(self) -> QueryPipeline:
        """
        构建查询管道

        Returns:
            QueryPipeline: 配置好的查询管道
        """
        pipeline = QueryPipeline()

        # 添加所有组件
        for name, component in self.components.items():
            pipeline.add_component(name, component)

        # 添加所有连接边
        for from_c, to_c in self.edges:
            pipeline.link(from_c, to_c)

        return pipeline


# 演示Query Pipeline的构建和使用
print("\n" + "=" * 60)
print("Query Pipeline演示：构建检索-生成管道")
print("=" * 60)

# 定义自定义查询转换函数
def transform_query(query: str) -> QueryBundle:
    """
    查询转换函数

    可以在此对用户查询进行预处理，如：
    - 添加搜索提示
    - 扩展同义词
    - 重写查询为更检索友好的形式

    Args:
        query: 原始查询

    Returns:
        QueryBundle: 转换后的查询包
    """
    # 例如：添加搜索提示来优化检索
    enhanced_query = f"根据以下内容回答问题：{query}"
    return QueryBundle(query_str=enhanced_query)

# 定义自定义后处理函数
def filter_results(nodes: List[Node]) -> List[Node]:
    """
    结果过滤函数

    可以在此对检索结果进行后处理，如：
    - 基于分数阈值过滤
    - 去重处理
    - 优先级排序

    Args:
        nodes: 检索结果节点列表

    Returns:
        List[Node]: 过滤后的节点列表
    """
    # 只保留分数大于0.5的结果
    threshold = 0.5
    filtered = [node for node in nodes if node.score >= threshold]
    print(f"[Postprocess] 过滤后保留 {len(filtered)}/{len(nodes)} 个结果")
    return filtered

# 构建Pipeline
builder = QueryPipelineBuilder()
pipeline = (
    builder
    .add_input("input")
    .add_retriever("retriever", hybrid_retriever)
    .add_custom_component("query_transformer", transform_query, "query")
    .add_custom_component("result_filter", filter_results, "query")
    .add_synthesizer("synthesizer", CompactResponseSynthesizer())
    .add_edge("input", "query_transformer")
    .add_edge("query_transformer", "retriever")
    .add_edge("retriever", "result_filter")
    .add_edge("result_filter", "synthesizer")
    .build()
)

# 执行Pipeline
test_query = "英伟达B200的存储器规格"
print(f"\n执行Pipeline，查询: '{test_query}'")
response = pipeline.run(input=test_query)
print(f"Pipeline执行完成，响应类型: {type(response)}")


# ============================================================
# 第五部分：模块化RAG完整架构演示
# ============================================================

class ModularRAGSystem:
    """
    模块化RAG系统

    这是一个完整的模块化RAG架构实现，集成了：
    1. Dataflow：文档数据处理流程
    2. Router：查询路由选择
    3. QueryPipeline：检索执行管道
    4. 后处理器：结果过滤和精排
    5. 合成器：答案生成

    设计模式：
    - 依赖注入：组件通过构造函数注入，便于测试和替换
    - 策略模式：不同的检索策略可以灵活切换
    - 管道模式：各组件通过Pipeline串联
    """

    def __init__(
        self,
        documents: List[Document],
        chunk_size: int = 100
    ):
        """
        初始化模块化RAG系统

        Args:
            documents: 文档列表
            chunk_size: 节点分割大小
        """
        print("[ModularRAG] 初始化模块化RAG系统...")

        # 1. 构建Dataflow
        print("[ModularRAG] 步骤1/4：构建数据流...")
        self.dataflow = DocumentDataflow(chunk_size=chunk_size)
        self.index = self.dataflow.run(documents)

        # 2. 创建检索器
        print("[ModularRAG] 步骤2/4：创建检索器...")
        self._create_retrievers()

        # 3. 配置Router
        print("[ModularRAG] 步骤3/4：配置路由器...")
        self.router = Router()
        self.router.add_path(RetrievalPath(
            name="semantic",
            retriever=self.vector_retriever,
            description="语义检索路径",
            适用场景="通用问答"
        ))
        self.router.add_path(RetrievalPath(
            name="hybrid",
            retriever=self.hybrid_retriever,
            description="混合检索路径",
            适用场景="技术问题"
        ))

        # 4. 构建Pipeline
        print("[ModularRAG] 步骤4/4：构建查询管道...")
        self._build_pipeline()

        print("[ModularRAG] 模块化RAG系统初始化完成！")

    def _create_retrievers(self):
        """创建各类检索器"""
        nodes = self.dataflow.nodes

        # BM25检索器（精确匹配）
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=5
        )

        # 向量检索器（语义匹配）
        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=5
        )

        # 混合检索器（融合两者）
        self.hybrid_retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            similarity_top_k=5,
            mode="reciprocal_rerank"
        )

    def _build_pipeline(self):
        """构建查询管道"""
        # 定义查询转换
        def enhance_query(query: str) -> QueryBundle:
            return QueryBundle(query_str=query)

        # 定义结果后处理
        def postprocess_results(results: List[Node]) -> List[Node]:
            # 简单过滤：移除低分结果
            return [r for r in results if r.score > 0.3]

        # 构建管道
        builder = QueryPipelineBuilder()
        self.pipeline = (
            builder
            .add_input("input")
            .add_custom_component("query_enhancer", enhance_query, "query")
            .add_retriever("retriever", self.hybrid_retriever)
            .add_custom_component("result_postprocessor", postprocess_results, "query")
            .add_edge("input", "query_enhancer")
            .add_edge("query_enhancer", "retriever")
            .add_edge("retriever", "result_postprocessor")
            .build()
        )

    def query(self, user_query: str) -> dict:
        """
        执行查询

        Args:
            user_query: 用户查询

        Returns:
            dict: 包含检索结果和元信息的字典
        """
        print(f"\n[ModularRAG] 处理查询: '{user_query}'")

        # 1. 路由选择
        retriever = self.router.select_retriever(user_query)

        # 2. 执行检索
        retrieval_results = retriever.retrieve(user_query)

        # 3. 返回结果
        return {
            "query": user_query,
            "retriever_used": retriever.__class__.__name__,
            "num_results": len(retrieval_results),
            "results": [
                {
                    "content": node.get_content(),
                    "score": node.score
                }
                for node in retrieval_results
            ]
        }


# 完整系统演示
print("\n" + "=" * 60)
print("完整模块化RAG系统演示")
print("=" * 60)

# 创建并运行系统
modular_rag = ModularRAGSystem(SAMPLE_DOCUMENTS)

# 执行测试查询
test_queries = [
    "英伟达B200的存储器规格是什么？",
    "特斯拉机器人有什么新进展？",
    "介绍一下RAG技术",
]

print("\n--- 模块化RAG查询测试 ---")
for query in test_queries:
    result = modular_rag.query(query)
    print(f"\n查询: {result['query']}")
    print(f"使用检索器: {result['retriever_used']}")
    print(f"检索结果数量: {result['num_results']}")
    for i, r in enumerate(result['results'][:2]):
        print(f"  [{i+1}] {r['content'][:50]}... (score={r['score']:.4f})")

print("\n" + "=" * 60)
print("模块化RAG架构演示完成")
print("=" * 60)
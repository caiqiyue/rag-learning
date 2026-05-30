"""
第15节 复杂查询处理与上下文压缩 - 代码示例

本文件演示了复杂查询处理的核心技术实现：
1. Query Decomposition 查询分解 - 将复杂问题拆解为多个简单子问题
2. 多跳查询处理 - 迭代检索和逐步推理
3. CRAG 可纠正检索增强 - 评估检索质量并动态调整策略

作者：RAG学习课程
日期：2026/05/30
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

# ============================================================================
# 第一部分：核心数据结构和基础类定义
# ============================================================================

@dataclass
class Document:
    """
    文档对象，用于存储文档内容和元数据

    Attributes:
        content: 文档的文本内容
        doc_id: 文档唯一标识符
        metadata: 文档的元数据信息（如来源、页码等）
        embedding: 文档的向量表示（可选）
    """
    content: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """
    检索结果对象，包含文档及其相关性信息

    Attributes:
        doc: 检索到的文档对象
        score: 相关性得分（0-1之间，越高越相关）
        node_id: 节点标识符
    """
    doc: Document
    score: float
    node_id: Optional[str] = None


@dataclass
class SubQuery:
    """
    子问题对象，用于Query Decomposition

    Attributes:
        query_id: 子问题唯一标识
        query_text: 子问题文本
        parent_query: 父问题（原始复杂问题）
        depends_on: 依赖的子问题ID列表（用于顺序分解）
        answer: 子问题的答案（检索后填充）
        is_answered: 是否已回答
    """
    query_id: str
    query_text: str
    parent_query: str
    depends_on: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    is_answered: bool = False


class QueryType(Enum):
    """查询类型枚举"""
    SIMPLE = "simple"                      # 简单查询
    SEQUENTIAL = "sequential"              # 顺序分解查询（后续问题依赖前序答案）
    PARALLEL = "parallel"                  # 并行分解查询（子问题独立）
    HIERARCHICAL = "hierarchical"          # 层次分解查询（多个子主题）


# ============================================================================
# 第二部分：模拟向量数据库和检索器
# ============================================================================

class MockVectorStore:
    """
    模拟向量数据库，用于演示检索过程

    在实际应用中，这会被替换为真实的向量数据库（如FAISS、ChromaDB等）
    """

    def __init__(self):
        """初始化向量数据库"""
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量数据库

        Args:
            documents: 要添加的文档列表
        """
        for doc in documents:
            self.documents[doc.doc_id] = doc
            self.embeddings[doc.doc_id] = self._simple_embed(doc.content)

    def _simple_embed(self, text: str) -> List[float]:
        """
        简单的文本向量化方法（演示用）

        实际应用中应使用专业的Embedding模型（如BGE、text-embedding-ada-002等）

        Args:
            text: 输入文本
        Returns:
            文本的向量表示
        """
        words = text.lower().split()
        vector = [0.0] * 10
        for i, word in enumerate(words[:10]):
            vector[i] = len(word) / 10.0
        return vector

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量
        Returns:
            相似度得分（0-1之间）
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(a * a for a in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.0
    ) -> List[RetrievalResult]:
        """
        基于向量相似度的检索方法

        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量
            score_threshold: 相似度阈值
        Returns:
            按相关性排序的检索结果列表
        """
        query_embedding = self._simple_embed(query)
        similarities = []
        for doc_id, embedding in self.embeddings.items():
            score = self.similarity(query_embedding, embedding)
            if score >= score_threshold:
                similarities.append((doc_id, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in similarities[:top_k]:
            results.append(RetrievalResult(
                doc=self.documents[doc_id],
                score=score,
                node_id=doc_id
            ))
        return results


class BaseRetriever(ABC):
    """检索器的抽象基类"""
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[RetrievalResult]:
        """执行检索的抽象方法"""
        pass


# ============================================================================
# 第三部分：Query Decomposition 查询分解
# ============================================================================

class QueryDecomposer:
    """
    查询分解器（Query Decomposition）

    核心思想：将复杂问题拆解为多个简单子问题，分别检索后再综合回答

    分解策略：
    1. 顺序分解（Sequential）：子问题按依赖顺序执行，后续问题依赖前序答案
    2. 并行分解（Parallel）：子问题相互独立，可并行检索
    3. 层次分解（Hierarchical）：将问题按子主题层次分解

    适用场景：
    - 多跳推理问题："谁写的《星际穿越》？他还导演了什么电影？"
    - 比较类问题："比较Python和Java的优劣"
    - 复合类问题："解释人工智能，包括定义、历史和应用"
    """

    def __init__(self, llm_simulator: Any = None):
        """
        初始化查询分解器

        Args:
            llm_simulator: LLM模拟器（用于实际应用中的智能分解）
        """
        self.llm_simulator = llm_simulator

    def _identify_query_type(self, query: str) -> QueryType:
        """
        判断查询类型，决定分解策略

        Args:
            query: 用户查询
        Returns:
            查询类型
        """
        # 启发式判断（实际应用中可使用LLM）
        sequential_indicators = ["谁", "什么", "哪部", "哪个"]  # 简单判断

        # 检查是否有多个实体或比较词
        has_comparison = any(word in query for word in ["比较", "对比", "vs", "或者"])
        has_list = "和" in query or "以及" in query or "还有" in query

        if has_comparison or has_list:
            return QueryType.PARALLEL

        # 检查是否为顺序依赖问题
        question_words = query.count("？")
        if question_words > 1:
            return QueryType.SEQUENTIAL

        # 简单检查包含多个子问题
        if "包括" in query or "包括" in query:
            return QueryType.HIERARCHICAL

        return QueryType.SIMPLE

    def _split_by_keywords(self, query: str) -> List[str]:
        """
        按关键词简单分割查询（演示用）

        实际应用中应使用LLM进行智能分割

        Args:
            query: 原始查询
        Returns:
            分割后的子问题列表
        """
        # 简单的分割策略
        sub_queries = []

        # 按"和"、"以及"、"还有"分割列表类问题
        list_split = re.split(r'[和以及还有]', query)
        if len(list_split) > 1:
            return [s.strip() for s in list_split if s.strip()]

        # 按问号分割多问题
        question_split = query.split("？")
        if len(question_split) > 1:
            return [s.strip() + "？" for s in question_split if s.strip()]

        return [query]

    def decompose(self, query: str) -> List[SubQuery]:
        """
        分解复杂查询为多个子问题

        Args:
            query: 原始复杂查询
        Returns:
            子问题列表
        """
        query_type = self._identify_query_type(query)

        if query_type == QueryType.SIMPLE:
            # 简单查询不需要分解
            return [SubQuery(
                query_id="q_0",
                query_text=query,
                parent_query=query
            )]

        elif query_type == QueryType.PARALLEL:
            # 并行分解：分割为独立的子问题
            sub_texts = self._split_by_keywords(query)
            return [
                SubQuery(
                    query_id=f"q_{i}",
                    query_text=sub_text,
                    parent_query=query,
                    depends_on=[]  # 并行查询，无依赖
                )
                for i, sub_text in enumerate(sub_texts)
            ]

        elif query_type == QueryType.SEQUENTIAL:
            # 顺序分解：需要按依赖关系排序
            sub_texts = self._split_by_keywords(query)
            sub_queries = []
            for i, sub_text in enumerate(sub_texts):
                depends_on = [] if i == 0 else [f"q_{i-1}"]  # 依赖前一个问题
                sub_queries.append(SubQuery(
                    query_id=f"q_{i}",
                    query_text=sub_text,
                    parent_query=query,
                    depends_on=depends_on
                ))
            return sub_queries

        else:
            # 层次分解：按子主题分割
            sub_texts = self._split_by_keywords(query)
            return [
                SubQuery(
                    query_id=f"q_{i}",
                    query_text=sub_text,
                    parent_query=query,
                    depends_on=[]
                )
                for i, sub_text in enumerate(sub_texts)
            ]

    def _generate_sub_queries_with_llm(self, query: str) -> List[SubQuery]:
        """
        使用LLM生成子问题（实际应用版本）

        Args:
            query: 原始查询
        Returns:
            子问题列表
        """
        if not self.llm_simulator:
            return self.decompose(query)

        prompt = f"""将以下复杂问题分解为多个简单的子问题。

原始问题：{query}

要求：
1. 每个子问题应该足够简单，可以单独回答
2. 如果子问题有依赖关系，标注出来
3. 确保子问题合起来能完整回答原始问题

请按以下格式输出：
子问题1：[问题内容]
子问题2：[问题内容]
..."""

        response = self.llm_simulator.generate(prompt)
        # 解析LLM输出生成SubQuery列表
        # 这里省略解析逻辑，实际应用中需要实现
        return self.decompose(query)


class QueryDecompositionRetriever(BaseRetriever):
    """
    基于查询分解的检索器

    使用Query Decomposition技术处理复杂查询：
    1. 将复杂查询分解为多个简单子问题
    2. 分别检索各子问题
    3. 综合各子问题的答案形成最终回答
    """

    def __init__(
        self,
        vector_store: MockVectorStore,
        decomposer: Optional[QueryDecomposer] = None
    ):
        """
        初始化基于查询分解的检索器

        Args:
            vector_store: 向量数据库
            decomposer: 查询分解器
        """
        self.vector_store = vector_store
        self.decomposer = decomposer or QueryDecomposer()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行基于查询分解的检索

        Args:
            query: 用户查询
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        # 步骤1：分解查询
        sub_queries = self.decomposer.decompose(query)

        if len(sub_queries) == 1:
            # 简单查询，直接检索
            return self.vector_store.similarity_search(query, top_k=top_k)

        # 步骤2：按依赖顺序检索子问题
        results_map: Dict[str, List[RetrievalResult]] = {}
        context_parts = []

        for sub_query in sub_queries:
            # 检查依赖是否满足
            if sub_query.depends_on:
                deps_answered = all(
                    results_map.get(dep) is not None
                    for dep in sub_query.depends_on
                )
                if not deps_answered:
                    continue

            # 构建带有上下文的查询（如果是顺序依赖）
            search_query = sub_query.query_text
            if sub_query.depends_on and context_parts:
                # 将前序答案加入上下文
                search_query = f"基于以下信息回答问题：{' '.join(context_parts)}。问题：{search_query}"

            # 执行检索
            results = self.vector_store.similarity_search(search_query, top_k=top_k)
            results_map[sub_query.query_id] = results

            # 记录答案
            if results:
                context_parts.append(results[0].doc.content)

        # 步骤3：合并所有结果
        all_results = []
        for results in results_map.values():
            all_results.extend(results)

        # 按相关性排序
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]


# ============================================================================
# 第四部分：多跳查询处理（迭代检索）
# ============================================================================

class MultiHopRetriever:
    """
    多跳检索器（Multi-Hop Retriever）

    核心思想：通过迭代检索逐步深入，满足多跳推理需求

    工作流程：
    1. 第一跳：基于初始查询检索相关实体
    2. 后续跳：基于当前结果构建新查询继续检索
    3. 终止条件：达到最大跳数或找到足够信息

    适用场景：
    - "谁写的《盗梦空间》？他还导演了什么电影？"
    - "特斯拉的CEO是谁？他创立的其他公司有哪些？"
    - "2024年诺贝尔物理学奖得主是谁？他的主要贡献是什么？"
    """

    def __init__(
        self,
        vector_store: MockVectorStore,
        max_hops: int = 3
    ):
        """
        初始化多跳检索器

        Args:
            vector_store: 向量数据库
            max_hops: 最大跳数限制
        """
        self.vector_store = vector_store
        self.max_hops = max_hops

    def _extract_entity_from_result(
        self,
        result: RetrievalResult,
        query: str
    ) -> Optional[str]:
        """
        从检索结果中提取关键实体

        这是一个简化版本，实际应用中应使用NER或LLM提取

        Args:
            result: 检索结果
            query: 原始查询
        Returns:
            提取的实体名称
        """
        # 简单启发式：取检索结果的前几个词作为实体
        words = result.doc.content.split()
        if len(words) > 3:
            return " ".join(words[:3])
        return result.doc.content[:20]

    def _build_next_hop_query(
        self,
        current_result: RetrievalResult,
        original_query: str
    ) -> str:
        """
        基于当前结果构建下一跳查询

        Args:
            current_result: 当前检索结果
            original_query: 原始查询
        Returns:
            下一跳的查询
        """
        # 简化版本：直接使用当前结果的内容扩展查询
        entity = self._extract_entity_from_result(current_result, original_query)

        # 提取原始查询中的意图
        if "谁" in original_query:
            return f"{entity}还做了什么？"
        elif "什么" in original_query or "哪些" in original_query:
            return f"{entity}的相关信息"
        else:
            return f"{entity}和{original_query}"

    def retrieve_with_hops(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        执行多跳检索

        Args:
            query: 初始查询
            top_k: 每跳返回的结果数

        Returns:
            包含完整检索过程的字典：
            - hops: 每跳的检索结果
            - final_results: 最终合并的结果
            - reasoning_path: 推理路径
        """
        result = {
            "original_query": query,
            "hops": [],
            "reasoning_path": [],
            "final_results": []
        }

        current_results = []
        all_results = []

        for hop in range(self.max_hops):
            # 确定当前跳的查询
            if hop == 0:
                current_query = query
            else:
                # 基于上一跳的最佳结果构建新查询
                if current_results:
                    best_result = current_results[0]
                    current_query = self._build_next_hop_query(
                        best_result,
                        query
                    )
                    result["reasoning_path"].append(
                        f"第{hop}跳：基于'{best_result.doc.content[:30]}...'构建查询"
                    )
                else:
                    break

            # 执行检索
            results = self.vector_store.similarity_search(current_query, top_k=top_k)

            hop_record = {
                "hop_number": hop + 1,
                "query": current_query,
                "results": results,
                "result_count": len(results)
            }
            result["hops"].append(hop_record)

            current_results = results
            all_results.extend(results)

            # 如果结果不理想，可以提前终止
            if not results or results[0].score < 0.3:
                result["reasoning_path"].append(
                    f"第{hop+1}跳结果质量低，终止检索"
                )
                break

        # 合并所有跳的结果
        all_results.sort(key=lambda x: x.score, reverse=True)
        result["final_results"] = all_results[:top_k]

        return result


# ============================================================================
# 第五部分：CRAG 可纠正检索增强
# ============================================================================

class RetrievalQuality(Enum):
    """检索结果质量等级"""
    HIGH = "high"    # 高质量：直接用于生成
    MEDIUM = "medium" # 中等质量：需要知识去污
    LOW = "low"      # 低质量：考虑网络搜索或依赖自身知识


class CorrectiveRAG:
    """
    CRAG - 纠正式检索增强（Corrective Retrieval Augmented Generation）

    核心思想：通过评估检索结果的相关性，对低质量检索进行主动降级

    决策机制：
    1. 检索结果质量高 → 直接用于生成
    2. 检索结果质量中 → 知识去污（Knowledge Refinement）
    3. 检索结果质量低 → 网络搜索替代 / 依赖自身知识

    优势：
    - 避免低质量检索结果误导生成
    - 自适应选择最佳的知识来源
    - 提高系统的鲁棒性
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        web_search_func: Any = None
    ):
        """
        初始化CRAG

        Args:
            retriever: 基础检索器
            web_search_func: 网络搜索函数（可选，用于低质量时替代）
        """
        self.retriever = retriever
        self.web_search_func = web_search_func

    def _calculate_relevance_score(
        self,
        query: str,
        doc: Document
    ) -> float:
        """
        计算文档与查询的相关性分数

        这是一个简化版本，实际应用中可使用更复杂的评估方法：
        - LLM判断
        - 关键词覆盖率
        - 语义相似度

        Args:
            query: 查询文本
            doc: 文档对象
        Returns:
            相关性分数（0-1之间）
        """
        query_words = set(query.lower().split())
        doc_words = set(doc.content.lower().split())

        # 计算词汇重叠度
        overlap = query_words & doc_words
        if not query_words:
            return 0.0

        # Jaccard相似度
        relevance = len(overlap) / len(query_words | doc_words) if (query_words | doc_words) else 0.0

        return relevance

    def _evaluate_retrieval_quality(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> Tuple[RetrievalQuality, float]:
        """
        评估检索结果质量

        Args:
            results: 检索结果列表
            query: 用户查询
        Returns:
            (质量等级, 平均相关性分数) 元组
        """
        if not results:
            return RetrievalQuality.LOW, 0.0

        # 计算每条结果的相关性分数
        relevance_scores = [
            self._calculate_relevance_score(query, r.doc)
            for r in results
        ]

        avg_score = sum(relevance_scores) / len(relevance_scores)
        max_score = max(relevance_scores) if relevance_scores else 0.0

        # 根据分数判断质量等级
        if avg_score >= 0.6 and max_score >= 0.7:
            return RetrievalQuality.HIGH, avg_score
        elif avg_score >= 0.3:
            return RetrievalQuality.MEDIUM, avg_score
        else:
            return RetrievalQuality.LOW, avg_score

    def _knowledge_refinement(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """
        知识去污：当检索结果质量中等时，对结果进行过滤和提纯

        策略：
        1. 计算每个文档与查询的相关性
        2. 过滤掉低于阈值的片段
        3. 保留核心相关内容

        Args:
            results: 原始检索结果
            query: 用户查询
        Returns:
            提纯后的检索结果
        """
        if not results:
            return []

        # 重新评估每个结果
        scored_results = [
            (r, self._calculate_relevance_score(query, r.doc))
            for r in results
        ]

        # 过滤低相关性结果
        threshold = 0.3
        refined = [
            r for r, score in scored_results
            if score >= threshold
        ]

        # 如果过滤后为空，保留最好的几个
        if not refined:
            results_sorted = sorted(results, key=lambda x: x.score, reverse=True)
            refined = results_sorted[:1]

        return refined

    def generate_with_crag(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        使用CRAG机制进行问答

        完整流程：
        1. 执行基础检索
        2. 评估检索质量
        3. 根据质量等级选择策略
        4. 生成答案

        Args:
            query: 用户查询
            top_k: 检索数量

        Returns:
            包含决策过程和结果的字典
        """
        result = {
            "query": query,
            "retrieval_quality": None,
            "quality_score": 0.0,
            "strategy": None,
            "retrieved_content": [],
            "final_answer": None,
            "used_web_search": False
        }

        # 步骤1：执行检索
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        # 步骤2：评估检索质量
        quality, score = self._evaluate_retrieval_quality(retrieved, query)
        result["retrieval_quality"] = quality.value
        result["quality_score"] = round(score, 3)

        # 步骤3：根据质量决定策略
        if quality == RetrievalQuality.HIGH:
            # 高质量：直接使用检索结果
            result["strategy"] = "direct_use"
            result["retrieved_content"] = [r.doc.content for r in retrieved]
            context = "\n".join(result["retrieved_content"])
            result["final_answer"] = f"基于高质量检索结果回答：{context[:200]}..."

        elif quality == RetrievalQuality.MEDIUM:
            # 中等质量：进行知识去污
            result["strategy"] = "knowledge_refinement"
            refined = self._knowledge_refinement(retrieved, query)
            result["retrieved_content"] = [r.doc.content for r in refined]
            context = "\n".join(result["retrieved_content"])
            result["final_answer"] = f"基于提纯后的检索结果回答：{context[:200]}..."

        else:
            # 低质量：考虑使用网络搜索或依赖自身知识
            result["strategy"] = "fallback"

            if self.web_search_func:
                # 有网络搜索功能，使用网络搜索
                web_results = self.web_search_func(query)
                result["final_answer"] = f"基于网络搜索结果回答：{web_results[:200]}..."
                result["used_web_search"] = True
            else:
                # 没有网络搜索，依赖自身知识
                result["final_answer"] = "检索结果质量过低，基于模型自身知识回答。"

        return result


# ============================================================================
# 第六部分：综合复杂查询处理系统
# ============================================================================

class ComplexQueryHandler:
    """
    复杂查询处理器 - 整合多种技术处理复杂查询

    完整流程：
    1. Query Understanding - 理解查询类型
    2. Query Decomposition - 分解复杂问题
    3. Multi-hop Retrieval - 多跳迭代检索
    4. CRAG Quality Control - 质量评估与纠正
    5. Answer Synthesis - 综合答案生成
    """

    def __init__(
        self,
        vector_store: MockVectorStore,
        decomposer: Optional[QueryDecomposer] = None
    ):
        """
        初始化复杂查询处理器

        Args:
            vector_store: 向量数据库
            decomposer: 查询分解器
        """
        self.vector_store = vector_store
        self.decomposer = decomposer or QueryDecomposer()
        self.multi_hop_retriever = MultiHopRetriever(vector_store)
        self.crag = CorrectiveRAG(
            QueryDecompositionRetriever(vector_store, self.decomposer)
        )

    def process_query(
        self,
        query: str,
        use_decomposition: bool = True,
        use_multi_hop: bool = True,
        use_crag: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        处理复杂查询的完整流程

        Args:
            query: 用户查询
            use_decomposition: 是否使用查询分解
            use_multi_hop: 是否使用多跳检索
            use_crag: 是否使用CRAG质量控制
            top_k: 返回结果数量

        Returns:
            包含处理过程和结果的字典
        """
        result = {
            "original_query": query,
            "query_type": None,
            "sub_queries": [],
            "retrieval_results": [],
            "final_answer": None,
            "processing_steps": []
        }

        # 步骤1：判断查询类型
        query_type = self.decomposer._identify_query_type(query)
        result["query_type"] = query_type.value
        result["processing_steps"].append(f"查询类型识别：{query_type.value}")

        # 步骤2：查询分解（如果需要）
        if use_decomposition:
            sub_queries = self.decomposer.decompose(query)
            result["sub_queries"] = [
                {"id": sq.query_id, "text": sq.query_text, "depends_on": sq.depends_on}
                for sq in sub_queries
            ]
            result["processing_steps"].append(f"查询分解：分解为{len(sub_queries)}个子问题")

        # 步骤3：多跳检索（如果需要）
        if use_multi_hop and query_type != QueryType.SIMPLE:
            multi_hop_result = self.multi_hop_retriever.retrieve_with_hops(query, top_k=top_k)
            result["retrieval_results"] = [
                {
                    "content": r.doc.content,
                    "score": r.score,
                    "hop": hop_info["hop_number"]
                }
                for hop_info in multi_hop_result["hops"]
                for r in hop_info["results"]
            ]
            result["processing_steps"].extend(multi_hop_result["reasoning_path"])

        # 步骤4：CRAG质量控制
        if use_crag:
            crag_result = self.crag.generate_with_crag(query, top_k=top_k)
            result["quality_assessment"] = {
                "quality": crag_result["retrieval_quality"],
                "score": crag_result["quality_score"],
                "strategy": crag_result["strategy"]
            }
            result["final_answer"] = crag_result["final_answer"]
            result["processing_steps"].append(
                f"质量评估：{crag_result['retrieval_quality']}，策略：{crag_result['strategy']}"
            )
        else:
            # 简单检索
            simple_results = self.vector_store.similarity_search(query, top_k=top_k)
            result["retrieval_results"] = [
                {"content": r.doc.content, "score": r.score}
                for r in simple_results
            ]
            context = "\n".join([r.doc.content for r in simple_results])
            result["final_answer"] = f"基于检索结果回答：{context[:200]}..."
            result["processing_steps"].append("执行简单检索")

        return result


# ============================================================================
# 第七部分：演示代码
# ============================================================================

def run_demo():
    """
    演示函数：展示复杂查询处理技术的效果
    """
    print("=" * 70)
    print("复杂查询处理技术演示")
    print("=" * 70)

    # 创建模拟向量数据库
    vector_store = MockVectorStore()

    # 添加测试文档
    test_documents = [
        Document(
            content="克里斯托弗·诺兰是一位著名导演，代表作包括《盗梦空间》、《星际穿越》、《蝙蝠侠：黑暗骑士》等。",
            doc_id="doc_001",
            metadata={"source": "电影资料库", "type": "人物"}
        ),
        Document(
            content="《盗梦空间》是由克里斯托弗·诺兰执导的科幻电影，于2010年上映。",
            doc_id="doc_002",
            metadata={"source": "电影资料库", "type": "作品"}
        ),
        Document(
            content="《星际穿越》是一部关于太空探索的科幻电影，由克里斯托弗·诺兰执导。",
            doc_id="doc_003",
            metadata={"source": "电影资料库", "type": "作品"}
        ),
        Document(
            content="Python是一种高级编程语言，广泛用于Web开发、数据科学和AI领域。",
            doc_id="doc_004",
            metadata={"source": "编程教程", "type": "技术"}
        ),
        Document(
            content="Java是一种面向对象的编程语言，主要用于企业级应用和Android开发。",
            doc_id="doc_005",
            metadata={"source": "编程教程", "type": "技术"}
        ),
        Document(
            content="深度学习是机器学习的一个分支，使用神经网络进行特征学习。",
            doc_id="doc_006",
            metadata={"source": "AI教程", "type": "技术"}
        ),
    ]
    vector_store.add_documents(test_documents)

    # -------- 演示1：查询类型识别 --------
    print("\n【演示1】查询类型识别")
    print("-" * 50)

    decomposer = QueryDecomposer()
    test_queries = [
        "什么是Python？",
        "Python和Java有什么区别？",
        "谁写的《盗梦空间》？他还导演了什么电影？",
        "介绍机器学习，包括定义、历史和应用"
    ]

    for q in test_queries:
        q_type = decomposer._identify_query_type(q)
        print(f"查询：{q}")
        print(f"类型：{q_type.value}")
        print()

    # -------- 演示2：Query Decomposition --------
    print("\n【演示2】Query Decomposition 查询分解")
    print("-" * 50)

    complex_query = "谁写的《盗梦空间》？他还导演了什么电影？"
    sub_queries = decomposer.decompose(complex_query)

    print(f"原始查询：{complex_query}")
    print(f"分解结果：")
    for sq in sub_queries:
        deps = f"（依赖：{', '.join(sq.depends_on)})" if sq.depends_on else ""
        print(f"  - [{sq.query_id}] {sq.query_text} {deps}")

    # -------- 演示3：多跳检索 --------
    print("\n【演示3】Multi-Hop 多跳检索")
    print("-" * 50)

    multi_hop = MultiHopRetriever(vector_store, max_hops=2)
    multi_hop_result = multi_hop.retrieve_with_hops(
        "诺兰还导演了什么电影？",
        top_k=2
    )

    print(f"原始查询：诺兰还导演了什么电影？")
    print(f"跳数：{len(multi_hop_result['hops'])}")

    for hop_info in multi_hop_result["hops"]:
        print(f"\n第{hop_info['hop_number']}跳：")
        print(f"  查询：{hop_info['query']}")
        print(f"  结果数：{hop_info['result_count']}")

    print("\n推理路径：")
    for reason in multi_hop_result["reasoning_path"]:
        print(f"  - {reason}")

    # -------- 演示4：CRAG质量评估 --------
    print("\n【演示4】CRAG 可纠正检索增强")
    print("-" * 50)

    base_retriever = QueryDecompositionRetriever(vector_store, decomposer)
    crag = CorrectiveRAG(base_retriever)

    # 测试高质量检索
    crag_result = crag.generate_with_crag("盗梦空间是谁导演的？", top_k=3)
    print(f"查询：盗梦空间是谁导演的？")
    print(f"质量评估：{crag_result['retrieval_quality']} (分数：{crag_result['quality_score']})")
    print(f"策略：{crag_result['strategy']}")
    print(f"答案：{crag_result['final_answer']}")

    # 测试低质量检索
    print()
    crag_result_low = crag.generate_with_crag("量子计算的最新进展是什么？", top_k=3)
    print(f"查询：量子计算的最新进展是什么？")
    print(f"质量评估：{crag_result_low['retrieval_quality']} (分数：{crag_result_low['quality_score']})")
    print(f"策略：{crag_result_low['strategy']}")
    print(f"答案：{crag_result_low['final_answer']}")

    # -------- 演示5：完整复杂查询处理流程 --------
    print("\n【演示5】ComplexQueryHandler 完整流程")
    print("-" * 50)

    handler = ComplexQueryHandler(vector_store, decomposer)

    test_complex = "比较Python和Java的区别和应用场景"
    full_result = handler.process_query(
        test_complex,
        use_decomposition=True,
        use_multi_hop=True,
        use_crag=True
    )

    print(f"原始查询：{test_complex}")
    print(f"查询类型：{full_result['query_type']}")
    print(f"\n处理步骤：")
    for i, step in enumerate(full_result['processing_steps'], 1):
        print(f"  {i}. {step}")

    if full_result.get('sub_queries'):
        print(f"\n子问题分解：")
        for sq in full_result['sub_queries']:
            print(f"  - {sq['text']}")

    if full_result.get('quality_assessment'):
        print(f"\n质量评估：")
        print(f"  质量等级：{full_result['quality_assessment']['quality']}")
        print(f"  质量分数：{full_result['quality_assessment']['score']}")
        print(f"  采用策略：{full_result['quality_assessment']['strategy']}")

    print(f"\n最终答案：")
    print(f"  {full_result['final_answer']}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


def main():
    """主函数"""
    print("开始运行复杂查询处理演示...")
    print()
    run_demo()


if __name__ == "__main__":
    main()
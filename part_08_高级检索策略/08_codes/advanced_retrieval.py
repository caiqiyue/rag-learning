"""
第10节 高级检索策略 - 代码示例

本文件演示了多种高级检索策略的实现，包括：
1. Sentence Window Retrieval - 句子窗口检索
2. Parent Document Retrieval - 父子文档检索
3. HyDE - 假设文档嵌入检索
4. Contextual Compression - 上下文压缩
5. Self-RAG - 自适应检索（概念演示）
6. GraphRAG - 知识图谱检索（基础实现）
7. 融合示例 - 组合多种策略

作者：RAG学习课程
日期：2026/05/30
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

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
class检索结果:
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
class QueryContext:
    """
    查询上下文，用于在检索过程中传递上下文信息

    Attributes:
        query: 用户输入的查询问题
        transformed_query: 变换后的查询（用于HyDE等策略）
        retrieval_needed: 是否需要检索（用于Self-RAG）
       反思_reasoning: 反思推理过程
    """
    query: str
    transformed_query: Optional[str] = None
    retrieval_needed: Optional[bool] = None
    反思_reasoning: Optional[str] = None


# ============================================================================
# 第二部分：模拟向量数据库和检索器
# ============================================================================

class MockVectorStore:
    """
    模拟向量数据库，用于演示检索过程

    在实际应用中，这会被替换为真实的向量数据库（如FAISS、ChromaDB等）
    此类使用简单的余弦相似度计算来演示原理
    """

    def __init__(self):
        """初始化向量数据库，存储文档及其向量表示"""
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
            # 模拟：为每个文档生成一个基于内容的简单向量表示
            # 实际应用中会使用真实的Embedding模型
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
        # 这是一个极其简化的向量化方法，仅用于演示
        # 实际应用中请使用专业的Embedding API
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
        # 简化的余弦相似度计算
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
    ) -> List[检索结果]:
        """
        基于向量相似度的检索方法

        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量
            score_threshold: 相似度阈值，低于此值的文档将被过滤
        Returns:
            按相关性排序的检索结果列表
        """
        query_embedding = self._simple_embed(query)

        # 计算所有文档与查询的相似度
        similarities = []
        for doc_id, embedding in self.embeddings.items():
            score = self.similarity(query_embedding, embedding)
            if score >= score_threshold:
                similarities.append((doc_id, score))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k个结果
        results = []
        for doc_id, score in similarities[:top_k]:
            results.append(检索结果(
                doc=self.documents[doc_id],
                score=score,
                node_id=doc_id
            ))

        return results


class BaseRetriever(ABC):
    """
    检索器的抽象基类，定义检索器的基本接口

    所有具体的检索策略都应继承此类并实现retrieve方法
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[检索结果]:
        """
        执行检索的抽象方法

        Args:
            query: 用户查询
            top_k: 返回结果数量
            **kwargs: 其他参数
        Returns:
            检索结果列表
        """
        pass


# ============================================================================
# 第三部分：Sentence Window Retrieval 实现
# ============================================================================

class SentenceWindowRetriever(BaseRetriever):
    """
    句子窗口检索器（Sentence Window Retrieval）

    核心思想：
    - 将文档分割为单个或几个句子的窗口
    - 检索时不仅返回匹配的句子，还返回其周围的上下文

    优点：
    - 精确句子级别匹配保证相关性
    - 保留周围句子维持语义完整
    - 窗口大小可调，灵活性高

    适用场景：
    - 需要精确信息查找
    - 希望保持上下文完整性
    """

    def __init__(
        self,
        vector_store: MockVectorStore,
        window_size: int = 2
    ):
        """
        初始化句子窗口检索器

        Args:
            vector_store: 向量数据库实例
            window_size: 上下文窗口大小（前后各扩展的句子数）
        """
        self.vector_store = vector_store
        self.window_size = window_size  # 窗口大小：前后各扩展的句子数
        # 模拟：存储句子及其所属文档的映射
        self.sentences: List[Tuple[str, str, int]] = []  # (句子内容, 文档ID, 句子序号)

    def _split_into_sentences(self, documents: List[Document]) -> None:
        """
        将文档分割为句子

        实际应用中可使用专业分句库（如nltk、spacy等）

        Args:
            documents: 要分割的文档列表
        """
        self.sentences = []
        for doc in documents:
            # 简单的分句方法：按句号、问号、感叹号分割
            import re
            sentence_list = re.split(r'[。！？.?!]', doc.content)
            for idx, sent in enumerate(sentence_list):
                sent = sent.strip()
                if sent:
                    self.sentences.append((sent, doc.doc_id, idx))

        # 为每个句子创建文档对象并添加到向量库
        for sent, doc_id, idx in self.sentences:
            doc = Document(
                content=sent,
                doc_id=f"{doc_id}_sent_{idx}",
                metadata={"is_sentence": True, "parent_doc_id": doc_id}
            )
            self.vector_store.add_documents([doc])

    def _expand_window(
        self,
        doc_id: str,
        sentence_idx: int
    ) -> List[str]:
        """
        扩展上下文窗口，获取周围句子

        Args:
            doc_id: 文档ID
            sentence_idx: 句子序号
        Returns:
            扩展后的上下文句子列表
        """
        # 找到同一文档的所有句子
        related_sentences = [
            (idx, sent) for sent, d_id, idx in self.sentences
            if d_id == doc_id
        ]

        # 获取窗口范围内的句子
        start_idx = max(0, sentence_idx - self.window_size)
        end_idx = min(len(related_sentences), sentence_idx + self.window_size + 1)

        expanded = []
        for idx in range(start_idx, end_idx):
            for sent in related_sentences:
                if sent[0] == idx:
                    expanded.append(sent[1])
                    break

        return expanded

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[检索结果]:
        """
        执行句子窗口检索

        Args:
            query: 用户查询
            top_k: 返回结果数量
            **kwargs: 其他参数（支持window_size覆盖）

        Returns:
            检索结果列表，每条结果包含扩展后的上下文
        """
        # 允许临时覆盖窗口大小
        window_size = kwargs.get('window_size', self.window_size)

        # 步骤1：检索最相关的句子
        initial_results = self.vector_store.similarity_search(
            query=query,
            top_k=top_k * 2,  # 检索更多结果以便扩展
            score_threshold=0.0
        )

        # 步骤2：对每个结果扩展上下文窗口
        expanded_results = []
        for result in initial_results:
            # 从node_id提取句子序号
            node_id = result.node_id
            if node_id and '_sent_' in node_id:
                parent_doc_id = result.doc.metadata.get('parent_doc_id', '')
                sentence_idx = int(node_id.split('_sent_')[-1])

                # 扩展窗口获取上下文
                expanded_context = self._expand_window(parent_doc_id, sentence_idx)

                # 创建新的文档对象，包含扩展后的上下文
                expanded_doc = Document(
                    content=' '.join(expanded_context),
                    doc_id=result.doc.doc_id,
                    metadata={
                        **result.doc.metadata,
                        "expanded": True,
                        "window_size": window_size
                    }
                )

                expanded_results.append(检索结果(
                    doc=expanded_doc,
                    score=result.score,
                    node_id=node_id
                ))
            else:
                expanded_results.append(result)

        # 返回top_k个扩展后的结果
        return expanded_results[:top_k]


# ============================================================================
# 第四部分：Parent Document Retrieval 实现
# ============================================================================

class ParentDocumentRetriever(BaseRetriever):
    """
    父子文档检索器（Parent Document Retrieval）

    核心思想：
    - 维护父子文档层级关系
    - 先检索小文档块（子文档）
    - 如果相关，再获取其父文档进行增强

    优点：
    - 精确匹配子文档
    - 保留父文档的完整上下文
    - 适合需要完整背景信息的场景

    适用场景：
    - 需要完整文档背景
    - 答案需要跨越多个段落
    """

    def __init__(
        self,
        vector_store: MockVectorStore,
        parent_chunk_size: int = 500,
        child_chunk_size: int = 100
    ):
        """
        初始化父子文档检索器

        Args:
            vector_store: 向量数据库实例
            parent_chunk_size: 父文档的分块大小（字符数）
            child_chunk_size: 子文档的分块大小（字符数）
        """
        self.vector_store = vector_store
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

        # 存储父子关系映射
        self.parent_map: Dict[str, str] = {}  # child_id -> parent_id
        self.chunks: Dict[str, Document] = {}  # 存储所有分块

    def _split_into_chunks(self, document: Document) -> List[Document]:
        """
        将文档分割为父子块

        策略：先按父块分割，每个父块再分割为多个子块

        Args:
            document: 要分割的文档
        Returns:
            分割后的子块列表
        """
        content = document.content
        parent_chunks = []
        child_chunks = []

        # 简单的分块逻辑：按字符数分割
        for i in range(0, len(content), self.parent_chunk_size):
            parent_content = content[i:i + self.parent_chunk_size]
            parent_id = f"{document.doc_id}_parent_{i // self.parent_chunk_size}"
            parent_doc = Document(
                content=parent_content,
                doc_id=parent_id,
                metadata={**document.metadata, "is_parent": True}
            )
            parent_chunks.append(parent_doc)

            # 再将父块分割为子块
            for j in range(0, len(parent_content), self.child_chunk_size):
                child_content = parent_content[j:j + self.child_chunk_size]
                child_id = f"{parent_id}_child_{j // self.child_chunk_size}"
                child_doc = Document(
                    content=child_content,
                    doc_id=child_id,
                    metadata={**document.metadata, "is_child": True}
                )
                child_chunks.append(child_doc)

                # 建立父子映射关系
                self.parent_map[child_id] = parent_id

        return child_chunks

    def index_documents(self, documents: List[Document]) -> None:
        """
        为文档建立索引，包括父子分块

        Args:
            documents: 原始文档列表
        """
        for doc in documents:
            # 分割为子块
            child_chunks = self._split_into_chunks(doc)

            # 添加所有子块到向量库
            self.vector_store.add_documents(child_chunks)

            # 存储分块引用
            for chunk in child_chunks:
                self.chunks[chunk.doc_id] = chunk

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[检索结果]:
        """
        执行父子文档检索

        Args:
            query: 用户查询
            top_k: 返回结果数量

        Returns:
            检索结果列表，返回父文档以提供完整上下文
        """
        # 步骤1：使用子块进行检索
        child_results = self.vector_store.similarity_search(
            query=query,
            top_k=top_k * 2,
            score_threshold=0.0
        )

        # 步骤2：获取相关子块对应的父文档
        parent_results = []
        seen_parents = set()

        for result in child_results:
            child_id = result.node_id

            # 查找父文档ID
            if child_id in self.parent_map:
                parent_id = self.parent_map[child_id]

                # 避免重复添加同一个父文档
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)

                    # 找到父文档内容
                    parent_doc = self.chunks.get(parent_id)
                    if parent_doc:
                        parent_results.append(检索结果(
                            doc=parent_doc,
                            score=result.score,
                            node_id=parent_id
                        ))

        # 返回top_k个父文档结果
        return parent_results[:top_k]


# ============================================================================
# 第五部分：HyDE 假设文档嵌入检索
# ============================================================================

class HypotheticalDocumentEmbedder(BaseRetriever):
    """
    HyDE - 假设文档嵌入检索（Hypothetical Document Embeddings）

    核心思想：
    - 先让LLM根据用户问题生成一个"假设性答案文档"
    - 用这个假设文档去检索，而不是直接用用户问题
    - 因为假设答案可能与真实文档的表述更接近

    为什么HyDE有效？
    - 假设答案可能包含与真实文档相似的表述
    - 生成的答案包含更多可能相关的概念
    - 检索空间从"问题"扩展到"答案"

    注意：本实现是演示版本，实际使用需要接入真实的LLM API
    """

    def __init__(
        self,
        vector_store: MockVectorStore,
        llm_simulator: Any = None  # 实际应用中传入真实的LLM实例
    ):
        """
        初始化HyDE检索器

        Args:
            vector_store: 向量数据库实例
            llm_simulator: LLM模拟器（用于演示，实际应用中传入真实LLM）
        """
        self.vector_store = vector_store
        self.llm_simulator = llm_simulator

    def _generate_hypothetical_document(
        self,
        query: str
    ) -> str:
        """
        生成假设文档

        在实际应用中，这里会调用LLM API生成答案
        演示版本使用模板生成模拟的假设答案

        Args:
            query: 用户查询
        Returns:
            假设生成的答案文档
        """
        if self.llm_simulator:
            # 实际应用：调用真实LLM
            prompt = f"""根据以下问题，生成一个假设性的答案文档。
            这个答案文档应该详细回答问题，包含可能的概念和术语。

            问题：{query}

            假设答案文档："""
            return self.llm_simulator.generate(prompt)
        else:
            # 演示版本：使用模拟的假设答案
            # 实际应用中请接入真实的LLM
            return f"""假设答案：关于{query}的详细解答。
            本文档提供了对{query}的全面分析，包括：
            1. 相关的核心概念和定义
            2. 主要原理和机制
            3. 实际应用场景和案例
            4. 常见问题和解决方案
            这是HyDE策略生成的假设文档，用于改善检索效果。"""

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[检索结果]:
        """
        执行HyDE检索

        Args:
            query: 用户查询
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        # 步骤1：生成假设文档
        hypothetical_doc = self._generate_hypothetical_document(query)

        # 步骤2：用假设文档去检索（而不是直接用query）
        results = self.vector_store.similarity_search(
            query=hypothetical_doc,
            top_k=top_k,
            **kwargs
        )

        return results

    def retrieve_with_trace(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[List[检索结果], str, QueryContext]:
        """
        执行带追踪的HyDE检索，返回中间过程信息

        Args:
            query: 用户查询
            top_k: 返回结果数量

        Returns:
            (检索结果, 假设文档内容, 查询上下文) 元组
        """
        # 生成假设文档
        hypothetical_doc = self._generate_hypothetical_document(query)

        # 执行检索
        results = self.vector_store.similarity_search(
            query=hypothetical_doc,
            top_k=top_k
        )

        # 构建上下文
        context = QueryContext(
            query=query,
            transformed_query=hypothetical_doc,
            retrieval_needed=True,
            反思_reasoning="HyDE通过生成假设答案来改善检索匹配度"
        )

        return results, hypothetical_doc, context


# ============================================================================
# 第六部分：Contextual Compression 上下文压缩
# ============================================================================

class ContextualCompressor:
    """
    上下文压缩器（Contextual Compression）

    核心思想：
    - 对检索到的文档进行处理，提取最相关的部分
    - 同时去除冗余信息，减少无关内容干扰

    压缩策略：
    - LLMCompactor：使用LLM判断并提取关键句
    - RecursiveCompression：递归分割并选择最相关部分
    - SimpleCompressor：基于规则的简单压缩（演示用）

    注意：本实现是演示版本，实际使用需要接入真实的LLM API
    """

    def __init__(self, llm_simulator: Any = None):
        """
        初始化上下文压缩器

        Args:
            llm_simulator: LLM模拟器（用于演示）
        """
        self.llm_simulator = llm_simulator

    def _extract_key_sentences(
        self,
        document: Document,
        query: str,
        max_sentences: int = 3
    ) -> str:
        """
        从文档中提取与查询最相关的句子

        这是一个简化版本，实际应用中可使用：
        - LLM判断哪些句子最相关
        - 更复杂的文本相似度计算
        - 基于关键词的匹配

        Args:
            document: 要处理的文档
            query: 用户查询
            max_sentences: 最多保留的句子数
        Returns:
            压缩后的文本
        """
        import re

        # 简单分句
        sentences = re.split(r'[。！？.?!]', document.content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 模拟相关性评分（实际应用中应使用更好的算法）
        # 这里简单地计算查询词在句子中出现的次数
        query_words = set(query.lower().split())

        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            # 计算查询词在句子中的匹配数
            matches = sum(1 for word in query_words if word in sent_lower)
            scored_sentences.append((sent, matches))

        # 按匹配数降序排序
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # 取前max_sentences个句子
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]

        return '。'.join(top_sentences) + '。' if top_sentences else document.content

    def compress(
        self,
        results: List[检索结果],
        query: str,
        compression_type: str = "simple"
    ) -> List[检索结果]:
        """
        对检索结果进行压缩

        Args:
            results: 原始检索结果
            query: 用户查询
            compression_type: 压缩类型（simple/llm）

        Returns:
            压缩后的检索结果
        """
        compressed_results = []

        for result in results:
            original_content = result.doc.content

            if compression_type == "simple":
                # 使用简单压缩：提取关键句子
                compressed_content = self._extract_key_sentences(
                    result.doc, query, max_sentences=3
                )
            elif compression_type == "llm" and self.llm_simulator:
                # 使用LLM压缩（实际应用中）
                prompt = f"""根据以下查询，从文档中提取最相关的部分。

                查询：{query}

                文档：{original_content}

                要求：
                1. 保留与查询最相关的内容
                2. 去除冗余信息
                3. 保持内容的连贯性
                """
                compressed_content = self.llm_simulator.generate(prompt)
            else:
                # 不压缩
                compressed_content = original_content

            # 创建压缩后的文档
            compressed_doc = Document(
                content=compressed_content,
                doc_id=result.doc.doc_id,
                metadata={
                    **result.doc.metadata,
                    "compressed": True,
                    "original_length": len(original_content),
                    "compressed_length": len(compressed_content)
                }
            )

            compressed_results.append(检索结果(
                doc=compressed_doc,
                score=result.score,
                node_id=result.node_id
            ))

        return compressed_results


# ============================================================================
# 第七部分：Self-RAG 自适应检索（概念演示）
# ============================================================================

class SelfRAGSimulator:
    """
    Self-RAG 自适应检索模拟器

    Self-RAG是斯坦福大学提出的框架，核心思想是让LLM自己判断"需不需要检索"

    关键机制：使用特殊的反思token让模型在生成过程中判断：
    - [Retrieval]：需要检索
    - [No Retrieval]：不需要检索
    - [Relevant]：检索结果相关
    - [Irrelevant]：检索结果不相关

    注意：本实现是概念演示，实际Self-RAG需要微调模型来支持反思token
    """

    def __init__(self, retriever: BaseRetriever, llm_simulator: Any = None):
        """
        初始化Self-RAG模拟器

        Args:
            retriever: 基础检索器
            llm_simulator: LLM模拟器
        """
        self.retriever = retriever
        self.llm_simulator = llm_simulator

    def _should_retrieve(self, query: str) -> Tuple[bool, str]:
        """
        判断是否需要检索

        实际应用中，这里会使用LLM来判断
        演示版本使用启发式规则

        Args:
            query: 用户查询
        Returns:
            (是否需要检索, 判断理由) 元组
        """
        if self.llm_simulator:
            # 实际应用：使用LLM判断
            prompt = f"""判断以下查询是否需要从外部知识库检索信息来回答。
            如果问题涉及具体事实、数据、专业知识等需要外部信息的内容，回答"是"。
            如果问题是一般性对话或可以依靠模型自身知识回答，回答"否"。

            查询：{query}

            判断（是/否）："""
            decision = self.llm_simulator.generate(prompt).strip()
            return "是" in decision or "需要" in decision, f"LLM判断结果：{decision}"
        else:
            # 演示版本：启发式规则
            # 如果问题以"什么"、"谁"、"如何"、"为什么"等开头，认为需要检索
            question_starts = ["什么", "谁", "如何", "为什么", "请问", "能否"]
            needs_retrieval = any(query.startswith(start) for start in question_starts)

            reasoning = "问题似乎是寻求具体信息，执行检索" if needs_retrieval else "问题似乎是闲聊，不需要检索"
            return needs_retrieval, reasoning

    def _evaluate_relevance(
        self,
        query: str,
        retrieved_doc: Document
    ) -> Tuple[bool, str]:
        """
        评估检索结果与查询的相关性

        Args:
            query: 用户查询
            retrieved_doc: 检索到的文档
        Returns:
            (是否相关, 评估理由) 元组
        """
        if self.llm_simulator:
            # 实际应用：使用LLM判断相关性
            prompt = f"""判断以下检索到的文档是否与查询相关。

            查询：{query}

            文档：{retrieved_doc.content}

            判断（相关/不相关）："""
            decision = self.llm_simulator.generate(prompt).strip()
            return "相关" in decision, f"LLM评估：{decision}"
        else:
            # 演示版本：简单关键词匹配
            query_words = set(query.lower().split())
            doc_words = set(retrieved_doc.content.lower().split())
            overlap = len(query_words & doc_words)
            is_relevant = overlap >= 1

            reasoning = f"文档与查询有{overlap}个共同词汇，相关" if is_relevant else "文档与查询几乎没有重叠，不相关"
            return is_relevant, reasoning

    def generate_with_self_rag(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        使用Self-RAG机制进行问答

        这个方法模拟了Self-RAG的自适应检索和自我验证过程

        Args:
            query: 用户查询
            top_k: 检索结果数量

        Returns:
            包含完整推理过程的字典，包括：
            - decision: 是否执行了检索
            - retrieved_results: 检索结果
            - used_knowledge: 是否使用了检索知识
            - generation: 生成的答案
            - reasoning: 推理过程
        """
        result = {
            "query": query,
            "decision": None,
            "retrieved_results": [],
            "used_knowledge": False,
            "generation": None,
            "reasoning": []
        }

        # 步骤1：判断是否需要检索
        needs_retrieval, reason = self._should_retrieve(query)
        result["decision"] = needs_retrieval
        result["reasoning"].append(f"检索决策：{reason}")

        if needs_retrieval:
            # 步骤2：执行检索
            retrieved = self.retriever.retrieve(query, top_k=top_k)
            result["retrieved_results"] = retrieved
            result["reasoning"].append(f"执行检索，找到{len(retrieved)}个结果")

            # 步骤3：评估每个检索结果的相关性
            relevant_docs = []
            for i, res in enumerate(retrieved):
                is_relevant, eval_reason = self._evaluate_relevance(query, res.doc)
                result["reasoning"].append(f"结果{i+1}相关性评估：{eval_reason}")

                if is_relevant:
                    relevant_docs.append(res)

            # 步骤4：根据相关性决定是否使用检索结果
            if relevant_docs:
                result["used_knowledge"] = True
                result["reasoning"].append(f"使用{len(relevant_docs)}个相关文档进行生成")
                # 合并相关文档内容作为上下文
                context = "\n".join([doc.content for doc in relevant_docs])
            else:
                result["reasoning"].append("所有检索结果都不相关，依赖模型自身知识")
                context = None
        else:
            result["reasoning"].append("决定不执行检索，直接使用模型知识回答")
            context = None

        # 步骤5：生成答案
        if self.llm_simulator:
            if context:
                prompt = f"基于以下信息回答问题：\n{context}\n\n问题：{query}"
            else:
                prompt = query
            result["generation"] = self.llm_simulator.generate(prompt)
        else:
            # 演示版本：简单生成
            if context:
                result["generation"] = f"根据检索到的信息回答：{context[:100]}..."
            else:
                result["generation"] = "这是一个基于模型自身知识的回答。"

        return result


# ============================================================================
# 第八部分：GraphRAG 知识图谱检索（基础实现）
# ============================================================================

@dataclass
class GraphNode:
    """
    知识图谱节点

    Attributes:
        node_id: 节点唯一标识
        entity_type: 实体类型（如人物、地点、概念）
        entity_name: 实体名称
        description: 实体描述
        properties: 实体属性
    """
    node_id: str
    entity_type: str
    entity_name: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    """
    知识图谱关系

    Attributes:
        source_id: 源节点ID
        target_id: 目标节点ID
        relation_type: 关系类型（如"工作于"、"位于"等）
        properties: 关系属性
    """
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


class SimpleKnowledgeGraph:
    """
    简化的知识图谱实现

    用于演示GraphRAG的基本原理
    实际应用中需要使用专业的图数据库（如Neo4j、NebulaGraph等）
    """

    def __init__(self):
        """初始化空的知识图谱"""
        self.nodes: Dict[str, GraphNode] = {}
        self.relations: List[GraphRelation] = []
        self.adjacency: Dict[str, List[str]] = {}  # 邻接表：节点ID -> 关联节点ID列表

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        entity_name: str,
        description: str,
        properties: Dict[str, Any] = None
    ) -> None:
        """
        添加实体到知识图谱

        Args:
            entity_id: 实体ID
            entity_type: 实体类型
            entity_name: 实体名称
            description: 实体描述
            properties: 其他属性
        """
        node = GraphNode(
            node_id=entity_id,
            entity_type=entity_type,
            entity_name=entity_name,
            description=description,
            properties=properties or {}
        )
        self.nodes[entity_id] = node
        self.adjacency[entity_id] = []

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Dict[str, Any] = None
    ) -> None:
        """
        添加关系到知识图谱

        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            relation_type: 关系类型
            properties: 关系属性
        """
        if source_id not in self.adjacency:
            self.adjacency[source_id] = []
        if target_id not in self.adjacency:
            self.adjacency[target_id] = []

        relation = GraphRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {}
        )
        self.relations.append(relation)
        self.adjacency[source_id].append(target_id)

    def find_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> List[Tuple[GraphNode, str, int]]:
        """
        查找与给定实体相关联的其他实体

        使用广度优先搜索（BFS）查找多跳关联

        Args:
            entity_id: 起始实体ID
            max_depth: 最大搜索深度

        Returns:
            (关联实体, 关系类型, 距离) 列表
        """
        results = []
        visited = {entity_id}
        queue = [(entity_id, "", 0)]  # (当前节点, 关系类型, 深度)

        while queue:
            current_id, relation_type, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            for neighbor_id in self.adjacency.get(current_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)

                    # 找到关系类型
                    edge_relation = ""
                    for rel in self.relations:
                        if rel.source_id == current_id and rel.target_id == neighbor_id:
                            edge_relation = rel.relation_type
                            break

                    if neighbor_id in self.nodes:
                        results.append((self.nodes[neighbor_id], edge_relation, depth + 1))

                    queue.append((neighbor_id, edge_relation, depth + 1))

        return results

    def community_search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        社区搜索：查找与查询相关的实体社区

        这模拟了GraphRAG的社区级别搜索能力

        Args:
            query: 用户查询
            max_results: 最大返回结果数

        Returns:
            包含实体及其社区上下文的搜索结果
        """
        query_words = set(query.lower().split())
        results = []

        for node_id, node in self.nodes.items():
            # 计算简单的相关性分数
            text = f"{node.entity_name} {node.description}".lower()
            score = sum(1 for word in query_words if word in text)

            if score > 0:
                # 查找关联实体
                related = self.find_related_entities(node_id, max_depth=1)

                results.append({
                    "primary_entity": node,
                    "relevance_score": score,
                    "community_context": {
                        "related_entities": [
                            {
                                "entity": n.entity_name,
                                "type": n.entity_type,
                                "relation": rel
                            }
                            for n, rel, _ in related
                        ]
                    }
                })

        # 按相关性排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]


class GraphRAGRetriever(BaseRetriever):
    """
    GraphRAG 知识图谱检索器

    核心思想：
    - 利用知识图谱的结构化关系增强检索
    - 能回答需要关联多个实体的问题
    - 社区级别的摘要提供文档概览

    工作流程：
    1. 用户查询 → 定位相关实体
    2. 扩展到关联实体（多跳）
    3. 构建社区级别上下文
    4. 用于答案生成
    """

    def __init__(self, knowledge_graph: SimpleKnowledgeGraph):
        """
        初始化GraphRAG检索器

        Args:
            knowledge_graph: 知识图谱实例
        """
        self.knowledge_graph = knowledge_graph

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        max_hops: int = 2
    ) -> List[检索结果]:
        """
        执行基于知识图谱的检索

        Args:
            query: 用户查询
            top_k: 返回结果数量
            max_hops: 最大跳数（关联扩展范围）

        Returns:
            检索结果列表
        """
        # 社区搜索
        community_results = self.knowledge_graph.community_search(
            query=query,
            max_results=top_k
        )

        results = []
        for item in community_results:
            entity = item["primary_entity"]

            # 构建包含社区上下文的文档内容
            context_parts = [entity.description]

            # 添加关联实体信息
            for related in item["community_context"]["related_entities"]:
                context_parts.append(
                    f"与{entity.entity_name}相关的{related['type']}：{related['entity']}（关系：{related['relation']}）"
                )

            combined_content = "\n".join(context_parts)

            doc = Document(
                content=combined_content,
                doc_id=entity.node_id,
                metadata={
                    "entity_type": entity.entity_type,
                    "entity_name": entity.entity_name,
                    "is_graph_rag": True,
                    "max_hops": max_hops
                }
            )

            results.append(检索结果(
                doc=doc,
                score=float(item["relevance_score"]),
                node_id=entity.node_id
            ))

        return results


# ============================================================================
# 第九部分：CRAG 纠正式检索增强（概念演示）
# ============================================================================

class CorrectiveRAG:
    """
    CRAG - 纠正式检索增强（Corrective Retrieval Augmented Generation）

    核心思想：
    - 主动纠正低质量检索结果
    - 根据检索质量动态调整策略

    决策机制：
    1. 检索结果质量高 → 直接用于生成
    2. 检索结果质量中 → 知识去污（Knowledge Refinement）
    3. 检索结果质量低 → 网络搜索替代
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
            web_search_func: 网络搜索函数（可选）
        """
        self.retriever = retriever
        self.web_search_func = web_search_func

    def _evaluate_retrieval_quality(
        self,
        results: List[检索结果],
        query: str
    ) -> Tuple[str, float]:
        """
        评估检索结果质量

        Args:
            results: 检索结果列表
            query: 用户查询
        Returns:
            (质量等级, 质量分数) 元组
            质量等级："high", "medium", "low"
        """
        if not results:
            return "low", 0.0

        # 计算平均分数
        avg_score = sum(r.score for r in results) / len(results)
        max_score = max(r.score for r in results)

        # 简单评估逻辑
        if avg_score >= 0.7 and max_score >= 0.8:
            return "high", avg_score
        elif avg_score >= 0.4:
            return "medium", avg_score
        else:
            return "low", avg_score

    def _knowledge_refinement(
        self,
        results: List[检索结果],
        query: str
    ) -> List[检索结果]:
        """
        知识去污：当检索结果质量中等时，对结果进行过滤和提纯

        Args:
            results: 原始检索结果
            query: 用户查询
        Returns:
            提纯后的检索结果
        """
        # 简单策略：过滤掉低于平均分的低质量结果
        if not results:
            return []

        avg_score = sum(r.score for r in results) / len(results)
        refined = [r for r in results if r.score >= avg_score * 0.8]

        return refined if refined else results[:1]  # 至少保留一个

    def generate_with_crag(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        使用CRAG机制进行问答

        Args:
            query: 用户查询
            top_k: 检索数量

        Returns:
            包含决策过程和结果的字典
        """
        result = {
            "query": query,
            "retrieval_quality": None,
            "strategy": None,
            "retrieved_content": [],
            "final_answer": None
        }

        # 步骤1：执行检索
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        # 步骤2：评估检索质量
        quality, score = self._evaluate_retrieval_quality(retrieved, query)
        result["retrieval_quality"] = quality
        result["retrieved_content"] = [r.doc.content for r in retrieved]

        # 步骤3：根据质量决定策略
        if quality == "high":
            result["strategy"] = "direct_use"
            context = "\n".join([r.doc.content for r in retrieved])
            result["final_answer"] = f"基于高质量检索结果回答：{context[:200]}..."

        elif quality == "medium":
            result["strategy"] = "knowledge_refinement"
            refined = self._knowledge_refinement(retrieved, query)
            context = "\n".join([r.doc.content for r in refined])
            result["final_answer"] = f"基于提纯后的检索结果回答：{context[:200]}..."

        else:
            result["strategy"] = "web_search"
            if self.web_search_func:
                web_results = self.web_search_func(query)
                result["final_answer"] = f"基于网络搜索结果回答：{web_results[:200]}..."
            else:
                result["final_answer"] = "检索结果质量过低，但未配置网络搜索功能。"

        return result


# ============================================================================
# 第十部分：融合示例 - 组合多种策略
# ============================================================================

class AdvancedFusionRetriever:
    """
    高级融合检索器 - 组合多种检索策略

    完整的RAG流程可能包含：
    1. Query Rewriting - 改写问题
    2. Hybrid Retrieval - 混合检索（向量+关键词）
    3. Sentence Window - 扩展上下文
    4. Rerank - 精排
    5. Contextual Compression - 压缩

    这个融合检索器演示了如何组合使用这些策略
    """

    def __init__(
        self,
        vector_store: MockVectorStore,
        retrievers: Dict[str, BaseRetriever],
        compressor: Optional[ContextualCompressor] = None
    ):
        """
        初始化融合检索器

        Args:
            vector_store: 向量数据库
            retrievers: 各种检索器字典
            compressor: 上下文压缩器
        """
        self.vector_store = vector_store
        self.retrievers = retrievers
        self.compressor = compressor

    def _rewrite_query(self, query: str) -> str:
        """
        查询重写 - 将用户问题转换为更适合检索的形式

        Args:
            query: 原始查询
        Returns:
            重写后的查询
        """
        # 简单示例：去除口语化表达
        # 实际应用中应使用LLM进行更智能的重写

        # 去除常见口语化前缀
        prefixes_to_remove = ["请问", "我想问一下", "麻烦问一下", "能不能告诉我"]
        rewritten = query
        for prefix in prefixes_to_remove:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        return rewritten

    def _fuse_results(
        self,
        result_sets: List[List[检索结果]],
        weights: List[float] = None
    ) -> List[检索结果]:
        """
        融合多个检索器的结果

        使用 Reciprocal Rank Fusion 方法：
        RRF_score = sum(1 / (k + rank)) 其中k通常为60

        Args:
            result_sets: 多个检索器的结果列表
            weights: 各检索器的权重（可选）
        Returns:
            融合后的排序结果
        """
        if not result_sets:
            return []

        if weights is None:
            weights = [1.0] * len(result_sets)

        k = 60  # RRF算法中的常数

        # 收集所有文档及其排名
        doc_scores: Dict[str, float] = {}
        doc_first_result: Dict[str, 检索结果] = {}

        for result_set, weight in zip(result_sets, weights):
            for rank, result in enumerate(result_set, 1):
                doc_id = result.doc.doc_id

                # 计算RRF分数
                rrf_score = weight * (1 / (k + rank))

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_first_result[doc_id] = result

                doc_scores[doc_id] += rrf_score

        # 按融合分数排序
        sorted_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = []
        for doc_id, score in sorted_doc_ids:
            fused_results.append(检索结果(
                doc=doc_first_result[doc_id].doc,
                score=score,
                node_id=doc_first_result[doc_id].node_id
            ))

        return fused_results

    def retrieve(
        self,
        query: str,
        use_rewrite: bool = True,
        use_compression: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        执行融合检索

        完整流程：
        1. 查询重写
        2. 多检索器并行检索
        3. 结果融合
        4. 上下文压缩
        5. 返回结果

        Args:
            query: 用户查询
            use_rewrite: 是否使用查询重写
            use_compression: 是否使用上下文压缩
            top_k: 返回结果数量

        Returns:
            包含检索过程和结果的字典
        """
        result = {
            "original_query": query,
            "rewritten_query": None,
            "retrieval_strategies_used": [],
            "fused_results": [],
            "final_results": []
        }

        # 步骤1：查询重写
        if use_rewrite:
            result["rewritten_query"] = self._rewrite_query(query)
            search_query = result["rewritten_query"]
            result["retrieval_strategies_used"].append("Query Rewriting")
        else:
            search_query = query

        # 步骤2：多检索器并行检索
        result_sets = []
        strategy_names = []

        if "sentence_window" in self.retrievers:
            sw_results = self.retrievers["sentence_window"].retrieve(search_query, top_k=top_k)
            result_sets.append(sw_results)
            strategy_names.append("Sentence Window")

        if "parent_doc" in self.retrievers:
            pd_results = self.retrievers["parent_doc"].retrieve(search_query, top_k=top_k)
            result_sets.append(pd_results)
            strategy_names.append("Parent Document")

        if "hyde" in self.retrievers:
            hyde_results = self.retrievers["hyde"].retrieve(search_query, top_k=top_k)
            result_sets.append(hyde_results)
            strategy_names.append("HyDE")

        result["retrieval_strategies_used"].extend(strategy_names)

        # 步骤3：结果融合
        if result_sets:
            fused = self._fuse_results(result_sets)
            result["fused_results"] = fused
            result["retrieval_strategies_used"].append("Result Fusion (RRF)")
        else:
            fused = []

        # 步骤4：上下文压缩
        if use_compression and fused and self.compressor:
            compressed = self.compressor.compress(fused, search_query, compression_type="simple")
            result["final_results"] = compressed
            result["retrieval_strategies_used"].append("Contextual Compression")
        else:
            result["final_results"] = fused

        # 限制返回数量
        result["final_results"] = result["final_results"][:top_k]

        return result


# ============================================================================
# 第十一部分：演示代码
# ============================================================================

def run_demo():
    """
    演示函数：展示各种高级检索策略的效果
    """
    print("=" * 60)
    print("高级检索策略演示")
    print("=" * 60)

    # 创建模拟向量数据库
    vector_store = MockVectorStore()

    # 添加测试文档
    test_documents = [
        Document(
            content="机器学习是人工智能的子领域，它使用数据来训练模型。深度学习是机器学习的分支，使用神经网络。",
            doc_id="doc_001",
            metadata={"source": "AI基础教程"}
        ),
        Document(
            content="自然语言处理是人工智能的重要应用领域，主要处理文本和语音数据。Transformer架构是NLP领域的重大突破。",
            doc_id="doc_002",
            metadata={"source": "NLP教程"}
        ),
        Document(
            content="Python是一种广泛使用的高级编程语言，在数据科学和机器学习领域特别流行。",
            doc_id="doc_003",
            metadata={"source": "编程入门"}
        ),
    ]
    vector_store.add_documents(test_documents)

    # -------- 演示1：基础向量检索 --------
    print("\n【演示1】基础向量检索")
    print("-" * 40)
    basic_results = vector_store.similarity_search("机器学习", top_k=2)
    for i, res in enumerate(basic_results, 1):
        print(f"结果{i}: {res.doc.content[:50]}... (得分: {res.score:.3f})")

    # -------- 演示2：Sentence Window Retrieval --------
    print("\n【演示2】Sentence Window Retrieval")
    print("-" * 40)
    sw_retriever = SentenceWindowRetriever(vector_store, window_size=1)
    sw_retriever._split_into_sentences(test_documents)
    sw_results = sw_retriever.retrieve("机器学习是什么？", top_k=2)
    for i, res in enumerate(sw_results, 1):
        print(f"结果{i}: {res.doc.content[:60]}...")

    # -------- 演示3：Parent Document Retrieval --------
    print("\n【演示3】Parent Document Retrieval")
    print("-" * 40)
    pd_retriever = ParentDocumentRetriever(vector_store, parent_chunk_size=100, child_chunk_size=30)
    pd_retriever.index_documents(test_documents)
    pd_results = pd_retriever.retrieve("深度学习", top_k=2)
    for i, res in enumerate(pd_results, 1):
        print(f"结果{i}: {res.doc.content[:60]}... (ID: {res.node_id})")

    # -------- 演示4：HyDE --------
    print("\n【演示4】HyDE 假设文档检索")
    print("-" * 40)
    hyde = HypotheticalDocumentEmbedder(vector_store)
    hyde_results, hypothetical_doc, context = hyde.retrieve_with_trace("什么是Transformer？", top_k=2)
    print(f"假设生成的答案:\n{hypothetical_doc[:80]}...")
    print(f"\n实际检索到{len(hyde_results)}个结果")

    # -------- 演示5：Contextual Compression --------
    print("\n【演示5】Contextual Compression 上下文压缩")
    print("-" * 40)
    compressor = ContextualCompressor()
    # 先检索
    initial = vector_store.similarity_search("Python编程", top_k=3)
    print(f"压缩前结果数: {len(initial)}")
    for res in initial:
        print(f"  原始长度: {len(res.doc.content)}字符")

    compressed = compressor.compress(initial, "Python编程", compression_type="simple")
    print(f"\n压缩后结果数: {len(compressed)}")
    for res in compressed:
        print(f"  压缩后长度: {len(res.doc.content)}字符")
        print(f"  内容: {res.doc.content[:50]}...")

    # -------- 演示6：GraphRAG --------
    print("\n【演示6】GraphRAG 知识图谱检索")
    print("-" * 40)
    # 构建简单的知识图谱
    kg = SimpleKnowledgeGraph()
    kg.add_entity("e1", "技术", "机器学习", "一种使用数据训练模型的人工智能方法")
    kg.add_entity("e2", "子领域", "深度学习", "使用神经网络的机器学习分支")
    kg.add_entity("e3", "架构", "Transformer", "一种革命性的神经网络架构")
    kg.add_entity("e4", "应用", "自然语言处理", "处理人类语言的技术")
    kg.add_relation("e1", "e2", "包含")
    kg.add_relation("e2", "e3", "使用")
    kg.add_relation("e4", "e3", "使用")

    graph_retriever = GraphRAGRetriever(kg)
    graph_results = graph_retriever.retrieve("深度学习使用什么架构？", top_k=2)
    for i, res in enumerate(graph_results, 1):
        print(f"结果{i}: {res.doc.content[:60]}...")

    # -------- 演示7：Self-RAG --------
    print("\n【演示7】Self-RAG 自适应检索")
    print("-" * 40)
    # 使用已有的检索器作为基础
    self_rag = SelfRAGSimulator(vector_store)
    self_rag_result = self_rag.generate_with_self_rag("什么是机器学习？", top_k=2)
    print(f"是否执行检索: {self_rag_result['decision']}")
    print(f"推理过程:")
    for reasoning in self_rag_result['reasoning']:
        print(f"  - {reasoning}")
    print(f"生成答案: {self_rag_result['generation']}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


def main():
    """
    主函数：运行所有高级检索策略的演示
    """
    print("开始运行高级检索策略演示...")
    print()
    run_demo()


if __name__ == "__main__":
    main()
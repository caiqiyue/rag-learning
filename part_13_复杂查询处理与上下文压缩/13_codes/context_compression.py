"""
第15节 上下文压缩 - 代码示例

本文件演示了上下文压缩技术的多种实现：
1. ContextualCompressor - 基于相关性的上下文压缩
2. LLMCompactor - 基于LLM的智能压缩
3. RecursiveCompression - 递归压缩算法
4. 压缩效果评估与RAG集成

作者：RAG学习课程
日期：2026/05/30
"""

import os
import re
import math
from typing import List, Dict, Any, Optional, Tuple, Callable
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
        metadata: 文档的元数据信息
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
        score: 相关性得分（0-1之间）
        node_id: 节点标识符
    """
    doc: Document
    score: float
    node_id: Optional[str] = None


@dataclass
class CompressionResult:
    """
    压缩结果对象

    Attributes:
        original_content: 原始内容
        compressed_content: 压缩后内容
        compression_ratio: 压缩比
        extracted_key_points: 提取的关键点
        preserved_metadata: 保留的元数据
    """
    original_content: str
    compressed_content: str
    compression_ratio: float
    extracted_key_points: List[str] = field(default_factory=list)
    preserved_metadata: Dict[str, Any] = field(default_factory=dict)


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
# 第二部分：模拟向量数据库
# ============================================================================

class MockVectorStore:
    """模拟向量数据库，用于演示"""

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量数据库"""
        for doc in documents:
            self.documents[doc.doc_id] = doc
            self.embeddings[doc.doc_id] = self._simple_embed(doc.content)

    def _simple_embed(self, text: str) -> List[float]:
        """简单的文本向量化方法（演示用）"""
        words = text.lower().split()
        vector = [0.0] * 10
        for i, word in enumerate(words[:10]):
            vector[i] = len(word) / 10.0
        return vector

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
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
        """基于向量相似度的检索"""
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


class SimpleRetriever(BaseRetriever):
    """简单检索器实现"""

    def __init__(self, vector_store: MockVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[RetrievalResult]:
        return self.vector_store.similarity_search(query, top_k=top_k)


# ============================================================================
# 第三部分：ContextualCompressor 上下文压缩器
# ============================================================================

class ContextualCompressor:
    """
    上下文压缩器（Contextual Compression）

    核心思想：
    - 对检索到的文档进行处理，提取最相关的部分
    - 同时去除冗余信息，减少无关内容干扰

    工作流程：
    原始文档 → 相关性评估 → 关键信息提取 → 压缩内容

    优势：
    - 减少无关信息干扰
    - 提高生成质量
    - 降低Token消耗
    - 维持语义完整性

    适用场景：
    - 检索结果包含大量冗余信息
    - 上下文窗口有限但文档很长
    - 需要精确回答的场景
    """

    def __init__(
        self,
        max_sentences: int = 3,
        min_relevance_score: float = 0.2
    ):
        """
        初始化上下文压缩器

        Args:
            max_sentences: 压缩后保留的最大句子数
            min_relevance_score: 最小相关性分数阈值
        """
        self.max_sentences = max_sentences
        self.min_relevance_score = min_relevance_score

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子

        Args:
            text: 输入文本
        Returns:
            句子列表
        """
        # 使用简单的分句策略
        sentences = re.split(r'[。！？.?!]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _calculate_relevance_score(
        self,
        sentence: str,
        query: str
    ) -> float:
        """
        计算句子与查询的相关性分数

        使用词汇重叠和长度归一化

        Args:
            sentence: 句子文本
            query: 查询文本
        Returns:
            相关性分数（0-1之间）
        """
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())

        if not query_words or not sentence_words:
            return 0.0

        # 计算词汇重叠度（Jaccard相似度）
        overlap = query_words & sentence_words
        jaccard = len(overlap) / len(query_words | sentence_words)

        # 计算查询词在句子中的覆盖率
        coverage = len(overlap) / len(query_words)

        # 综合得分
        score = 0.7 * coverage + 0.3 * jaccard

        return score

    def _extract_key_sentences(
        self,
        document: Document,
        query: str
    ) -> Tuple[List[str], List[float]]:
        """
        从文档中提取与查询最相关的句子

        Args:
            document: 文档对象
            query: 查询文本
        Returns:
            (相关句子列表, 对应分数列表) 元组
        """
        sentences = self._split_into_sentences(document.content)

        # 计算每个句子的相关性分数
        scored_sentences = []
        for sent in sentences:
            score = self._calculate_relevance_score(sent, query)
            scored_sentences.append((sent, score))

        # 按分数降序排序
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # 选择高分句子
        selected = [
            (sent, score)
            for sent, score in scored_sentences
            if score >= self.min_relevance_score
        ][:self.max_sentences]

        # 如果没有达到阈值，选择分数最高的句子
        if not selected and scored_sentences:
            selected = [scored_sentences[0]]

        return [s for s, _ in selected], [score for _, score in selected]

    def compress(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """
        对检索结果进行压缩

        Args:
            results: 原始检索结果列表
            query: 用户查询
        Returns:
            压缩后的检索结果列表
        """
        compressed_results = []

        for result in results:
            original_content = result.doc.content
            original_length = len(original_content)

            # 提取关键句子
            key_sentences, scores = self._extract_key_sentences(result.doc, query)

            if key_sentences:
                compressed_content = '。'.join(key_sentences)
                if not compressed_content.endswith('。'):
                    compressed_content += '。'
            else:
                compressed_content = original_content

            compressed_length = len(compressed_content)
            compression_ratio = compressed_length / original_length if original_length > 0 else 1.0

            # 创建压缩后的文档
            compressed_doc = Document(
                content=compressed_content,
                doc_id=result.doc.doc_id,
                metadata={
                    **result.doc.metadata,
                    "compressed": True,
                    "original_length": original_length,
                    "compressed_length": compressed_length,
                    "compression_ratio": round(compression_ratio, 2),
                    "extracted_sentences": len(key_sentences)
                }
            )

            compressed_results.append(RetrievalResult(
                doc=compressed_doc,
                score=result.score,
                node_id=result.node_id
            ))

        return compressed_results

    def compress_to_text(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> str:
        """
        将压缩结果合并为单一文本

        Args:
            results: 压缩后的检索结果
            query: 查询文本（用于分隔）
        Returns:
            合并后的文本
        """
        compressed_texts = []
        for r in results:
            compressed_texts.append(r.doc.content)

        return "\n\n".join(compressed_texts)


# ============================================================================
# 第四部分：LLMCompactor 基于LLM的智能压缩
# ============================================================================

class LLMCompactor:
    """
    基于LLM的压缩器（LLM Compactor）

    核心思想：使用大语言模型来判断和提取文档中最相关的部分

    工作原理：
    1. 将文档和查询提交给LLM
    2. LLM分析文档内容，识别与查询最相关的部分
    3. LLM生成精炼的压缩内容

    优势：
    - 更高的语义理解能力
    - 能捕捉隐式的相关性问题
    - 压缩内容更流畅连贯

    局限：
    - 需要额外的LLM API调用
    - 增加延迟和成本

    注意：本实现使用模拟的LLM，实际应用中需要接入真实API
    """

    def __init__(self, llm_simulator: Any = None):
        """
        初始化LLM压缩器

        Args:
            llm_simulator: LLM模拟器（用于演示，实际应用中传入真实LLM）
        """
        self.llm_simulator = llm_simulator

    def _create_compression_prompt(
        self,
        document: Document,
        query: str
    ) -> str:
        """
        创建压缩提示词

        Args:
            document: 待压缩的文档
            query: 查询文本
        Returns:
            压缩提示词
        """
        return f"""根据以下查询，从文档中提取最相关的信息。

查询：{query}

文档：{document.content}

要求：
1. 只保留与查询直接相关的内容
2. 去除冗余的背景描述和无关信息
3. 保持内容的准确性和完整性
4. 如果文档与查询无关，返回"不相关"

压缩后的内容："""

    def _llm_compress(
        self,
        document: Document,
        query: str
    ) -> str:
        """
        使用LLM进行压缩

        Args:
            document: 待压缩的文档
            query: 查询文本
        Returns:
            压缩后的内容
        """
        if self.llm_simulator:
            # 实际应用：调用真实LLM
            prompt = self._create_compression_prompt(document, query)
            compressed = self.llm_simulator.generate(prompt)

            if "不相关" in compressed:
                return ""  # 文档与查询无关

            return compressed.strip()
        else:
            # 演示版本：使用模拟压缩
            # 简单地取文档的前半部分作为压缩结果
            sentences = re.split(r'[。！？.?!]', document.content)
            sentences = [s.strip() for s in sentences if s.strip()]

            # 保留前1-2个句子
            kept = sentences[:2]
            return '。'.join(kept) + '。' if kept else document.content[:200]

    def compress(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[CompressionResult]:
        """
        对检索结果进行LLM压缩

        Args:
            results: 原始检索结果
            query: 查询文本
        Returns:
            压缩结果列表
        """
        compression_results = []

        for result in results:
            original_content = result.doc.content
            original_length = len(original_content)

            # 使用LLM压缩
            compressed_content = self._llm_compress(result.doc, query)

            # 如果LLM认为不相关，标记为不可用
            if not compressed_content:
                compressed_content = "[文档与查询不相关]"
                compression_ratio = 0.0
            else:
                compressed_length = len(compressed_content)
                compression_ratio = compressed_length / original_length if original_length > 0 else 0.0

            compression_results.append(CompressionResult(
                original_content=original_content,
                compressed_content=compressed_content,
                compression_ratio=compression_ratio,
                extracted_key_points=[],  # LLM压缩不显式提取关键点
                preserved_metadata={
                    "doc_id": result.doc.doc_id,
                    "original_score": result.score,
                    "compression_method": "llm"
                }
            ))

        return compression_results


# ============================================================================
# 第五部分：RecursiveCompression 递归压缩
# ============================================================================

class RecursiveCompression:
    """
    递归压缩器（Recursive Compression）

    核心思想：通过递归分割文档，逐步选择最相关的部分

    工作原理：
    1. 将文档递归分割为较小的块
    2. 评估每个块与查询的相关性
    3. 选择高分块，丢弃低分块
    4. 对保留的块继续分割，直到满足大小要求

    适用场景：
    - 长文档处理
    - 需要多层次压缩
    - 精确信息提取
    """

    def __init__(
        self,
        initial_chunk_size: int = 500,
        min_chunk_size: int = 100,
        maxChunks_per_level: int = 5
    ):
        """
        初始化递归压缩器

        Args:
            initial_chunk_size: 初始块大小（字符数）
            min_chunk_size: 最小块大小（字符数）
            maxChunks_per_level: 每层保留的最大块数
        """
        self.initial_chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunks_per_level = maxChunks_per_level

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int
    ) -> List[str]:
        """
        将文本分割为块

        Args:
            text: 输入文本
            chunk_size: 块大小
        Returns:
            块列表
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def _calculate_chunk_score(
        self,
        chunk: str,
        query: str
    ) -> float:
        """
        计算块与查询的相关性分数

        Args:
            chunk: 文本块
            query: 查询文本
        Returns:
            相关性分数
        """
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())

        if not query_words:
            return 0.0

        # 计算查询词在块中的出现次数（归一化）
        matches = sum(1 for word in query_words if word in chunk_words)
        score = matches / len(query_words)

        return score

    def _recursive_compress(
        self,
        chunks: List[str],
        query: str,
        current_size: int
    ) -> List[str]:
        """
        递归压缩的内部方法

        Args:
            chunks: 当前层的块列表
            query: 查询文本
            current_size: 当前块大小
        Returns:
            压缩后的块列表
        """
        if not chunks or current_size <= self.min_chunk_size:
            return chunks

        # 评估每个块的相关性
        scored_chunks = [
            (chunk, self._calculate_chunk_score(chunk, query))
            for chunk in chunks
        ]

        # 按分数排序，保留高分块
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        selected = [chunk for chunk, score in scored_chunks[:self.max_chunks_per_level]]

        # 如果选择的块仍然太大，递归压缩
        if any(len(c) > current_size for c in selected):
            new_size = max(self.min_chunk_size, current_size // 2)
            new_chunks = []
            for chunk in selected:
                new_chunks.extend(self._split_into_chunks(chunk, new_size))
            return self._recursive_compress(new_chunks, query, new_size)

        return selected

    def compress(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """
        对检索结果进行递归压缩

        Args:
            results: 原始检索结果
            query: 查询文本
        Returns:
            压缩后的检索结果
        """
        compressed_results = []

        for result in results:
            original_content = result.doc.content
            original_length = len(original_content)

            # 初始分割
            chunks = self._split_into_chunks(
                original_content,
                self.initial_chunk_size
            )

            # 递归压缩
            compressed_chunks = self._recursive_compress(
                chunks,
                query,
                self.initial_chunk_size
            )

            compressed_content = "\n".join(compressed_chunks)
            compressed_length = len(compressed_content)
            compression_ratio = compressed_length / original_length if original_length > 0 else 1.0

            # 创建压缩后的文档
            compressed_doc = Document(
                content=compressed_content,
                doc_id=result.doc.doc_id,
                metadata={
                    **result.doc.metadata,
                    "compressed": True,
                    "compression_method": "recursive",
                    "original_length": original_length,
                    "compressed_length": compressed_length,
                    "compression_ratio": round(compression_ratio, 2),
                    "chunks_before": len(chunks),
                    "chunks_after": len(compressed_chunks)
                }
            )

            compressed_results.append(RetrievalResult(
                doc=compressed_doc,
                score=result.score,
                node_id=result.node_id
            ))

        return compressed_results


# ============================================================================
# 第六部分：压缩效果评估
# ============================================================================

class CompressionEvaluator:
    """
    压缩效果评估器

    用于评估不同压缩策略的效果，包括：
    - 压缩比（Compression Ratio）
    - 信息保留度（Information Retention）
    - 相关性保持度（Relevance Preservation）
    """

    def __init__(self):
        """初始化评估器"""
        pass

    def calculate_compression_ratio(
        self,
        original_length: int,
        compressed_length: int
    ) -> float:
        """
        计算压缩比

        Args:
            original_length: 原始长度
            compressed_length: 压缩后长度
        Returns:
            压缩比（压缩后/原始）
        """
        if original_length == 0:
            return 1.0
        return compressed_length / original_length

    def calculate_token_savings(
        self,
        original_length: int,
        compressed_length: int
    ) -> Tuple[int, float]:
        """
        计算Token节省量

        假设：每1.5个字符约等于1个Token

        Args:
            original_length: 原始长度
            compressed_length: 压缩后长度
        Returns:
            (节省的Token数, 节省百分比)
        """
        chars_per_token = 1.5

        original_tokens = original_length / chars_per_token
        compressed_tokens = compressed_length / chars_per_token

        saved_tokens = original_tokens - compressed_tokens
        savings_percent = (saved_tokens / original_tokens * 100) if original_tokens > 0 else 0

        return int(saved_tokens), round(savings_percent, 2)

    def evaluate_compression_quality(
        self,
        original_results: List[RetrievalResult],
        compressed_results: List[RetrievalResult],
        query: str
    ) -> Dict[str, Any]:
        """
        评估压缩质量

        Args:
            original_results: 原始检索结果
            compressed_results: 压缩后的检索结果
            query: 查询文本
        Returns:
            评估报告字典
        """
        report = {
            "total_original_length": 0,
            "total_compressed_length": 0,
            "overall_compression_ratio": 0.0,
            "total_token_savings": 0,
            "token_savings_percent": 0.0,
            "per_document_results": []
        }

        total_original = 0
        total_compressed = 0

        for orig, comp in zip(original_results, compressed_results):
            orig_len = len(orig.doc.content)
            comp_len = len(comp.doc.content)

            total_original += orig_len
            total_compressed += comp_len

            ratio = self.calculate_compression_ratio(orig_len, comp_len)
            saved, percent = self.calculate_token_savings(orig_len, comp_len)

            per_doc = {
                "doc_id": orig.doc.doc_id,
                "original_length": orig_len,
                "compressed_length": comp_len,
                "compression_ratio": round(ratio, 2),
                "token_savings": saved,
                "savings_percent": percent
            }
            report["per_document_results"].append(per_doc)

        report["total_original_length"] = total_original
        report["total_compressed_length"] = total_compressed
        report["overall_compression_ratio"] = round(
            total_compressed / total_original if total_original > 0 else 1.0,
            2
        )

        total_saved, total_percent = self.calculate_token_savings(
            total_original,
            total_compressed
        )
        report["total_token_savings"] = total_saved
        report["token_savings_percent"] = total_percent

        return report


# ============================================================================
# 第七部分：完整RAG流程集成演示
# ============================================================================

class CompressionAwareRAG:
    """
    支持压缩的RAG系统

    完整的RAG + 压缩流程：
    1. 检索（Retrieval）- 从向量数据库获取相关文档
    2. 压缩（Compression）- 对检索结果进行压缩
    3. 生成（Generation）- 使用压缩后的上下文生成答案

    支持多种压缩策略：
    - contextual: 基于相关性的压缩
    - llm: 基于LLM的智能压缩
    - recursive: 递归压缩
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        compressor: Optional[ContextualCompressor] = None,
        llm_simulator: Any = None
    ):
        """
        初始化支持压缩的RAG系统

        Args:
            retriever: 基础检索器
            compressor: 上下文压缩器
            llm_simulator: LLM模拟器
        """
        self.retriever = retriever
        self.compressor = compressor or ContextualCompressor()
        self.llm_simulator = llm_simulator

        # 初始化其他压缩器
        self.llm_compactor = LLMCompactor(llm_simulator)
        self.recursive_compressor = RecursiveCompression()

        # 评估器
        self.evaluator = CompressionEvaluator()

    def _generate_answer(
        self,
        context: str,
        query: str
    ) -> str:
        """
        使用LLM生成答案（模拟版本）

        Args:
            context: 上下文文本
            query: 用户查询
        Returns:
            生成的答案
        """
        if self.llm_simulator:
            prompt = f"基于以下信息回答问题：\n\n{context}\n\n问题：{query}"
            return self.llm_simulator.generate(prompt)
        else:
            # 演示版本
            return f"根据检索到的信息（{len(context)}字符），回答：{query[:20]}..."

    def query(
        self,
        query: str,
        top_k: int = 5,
        compression_method: str = "contextual",
        return_compression_stats: bool = False
    ) -> Dict[str, Any]:
        """
        执行带压缩的查询

        Args:
            query: 用户查询
            top_k: 检索数量
            compression_method: 压缩方法（contextual/llm/recursive）
            return_compression_stats: 是否返回压缩统计信息

        Returns:
            包含答案和处理信息的字典
        """
        result = {
            "query": query,
            "method": f"CompressionRAG ({compression_method})",
            "retrieval_count": 0,
            "compression_ratio": 0.0,
            "token_savings": 0,
            "answer": None
        }

        # 步骤1：检索
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        result["retrieval_count"] = len(retrieved)

        if not retrieved:
            result["answer"] = "未找到相关信息"
            return result

        # 步骤2：压缩
        if compression_method == "contextual":
            compressed = self.compressor.compress(retrieved, query)
        elif compression_method == "llm":
            compression_results = self.llm_compactor.compress(retrieved, query)
            compressed = [
                RetrievalResult(
                    doc=Document(
                        content=r.compressed_content,
                        doc_id=orig.doc.doc_id,
                        metadata=orig.doc.metadata
                    ),
                    score=orig.score,
                    node_id=orig.node_id
                )
                for r, orig in zip(compression_results, retrieved)
            ]
        elif compression_method == "recursive":
            compressed = self.recursive_compressor.compress(retrieved, query)
        else:
            compressed = retrieved

        # 步骤3：评估压缩效果
        eval_report = self.evaluator.evaluate_compression_quality(
            retrieved, compressed, query
        )
        result["compression_ratio"] = eval_report["overall_compression_ratio"]
        result["token_savings"] = eval_report["total_token_savings"]

        # 步骤4：合并压缩结果
        context = self.compressor.compress_to_text(compressed, query)

        # 步骤5：生成答案
        result["answer"] = self._generate_answer(context, query)

        if return_compression_stats:
            result["compression_stats"] = eval_report

        return result

    def compare_compression_methods(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        对比不同压缩方法的效果

        Args:
            query: 测试查询
            top_k: 检索数量
        Returns:
            各方法的评估结果字典
        """
        methods = ["contextual", "llm", "recursive", "none"]
        results = {}

        for method in methods:
            result = self.query(
                query,
                top_k=top_k,
                compression_method=method,
                return_compression_stats=True
            )
            results[method] = {
                "compression_ratio": result["compression_ratio"],
                "token_savings": result["token_savings"],
                "answer_length": len(result["answer"]) if result["answer"] else 0
            }

        return results


# ============================================================================
# 第八部分：演示代码
# ============================================================================

def run_demo():
    """
    演示函数：展示上下文压缩技术的效果
    """
    print("=" * 70)
    print("上下文压缩技术演示")
    print("=" * 70)

    # 创建模拟向量数据库
    vector_store = MockVectorStore()

    # 添加测试文档（使用较长的文档来展示压缩效果）
    test_documents = [
        Document(
            content="""Python是一种广泛使用的高级编程语言，在数据科学、机器学习、Web开发和自动化领域都有重要应用。
            Python的设计哲学强调代码的可读性和简洁的语法，相比其他语言，Python让开发者可以用更少的代码完成同样的任务。
            Python拥有丰富的标准库和第三方库支持，包括NumPy、Pandas、TensorFlow、PyTorch等著名库。
            Python的易学性使其成为编程入门者的首选语言，同时也受到专业开发者的青睐。
            Python支持多种编程范式，包括面向对象、过程式和函数式编程。
            Python的主要特点包括：简洁易读的语法、强大的标准库、跨平台支持、动态类型系统、自动内存管理。
            Python的应用场景非常广泛，包括Web开发（Django、Flask）、数据科学（Jupyter、 Pandas）、机器学习（TensorFlow、PyTorch）、
            自动化脚本、网络爬虫、游戏开发等。Python的社区非常活跃，拥有大量的学习资源和开源项目。""",
            doc_id="doc_001",
            metadata={"source": "Python教程", "category": "编程语言"}
        ),
        Document(
            content="""Java是一种面向对象的编程语言，由Sun Microsystems于1995年发布。
            Java的设计目标是"一次编写，到处运行"，通过Java虚拟机（JVM）实现跨平台能力。
            Java语言结构清晰、类型安全，适合开发企业级应用和Android应用。
            Java拥有强大的企业级框架支持，包括Spring、Hibernate、Struts等。
            Java的主要特点包括：面向对象、平台无关性、自动内存管理、丰富的API、强大的安全性。
            Java的类型系统分为基本类型和引用类型，其异常处理机制也非常完善。
            Java的应用场景主要包括：企业级Web应用、Android应用开发、大数据处理（Hadoop基于Java）、
            金融系统、桌面应用等。Java拥有世界上最大的开发者社区之一。""",
            doc_id="doc_002",
            metadata={"source": "Java教程", "category": "编程语言"}
        ),
        Document(
            content="""机器学习是人工智能的一个分支，专门研究如何让计算机系统从数据中学习和改进。
            机器学习算法可以自动发现数据中的模式和规律，而无需明确编程指令。
            机器学习主要分为三类：监督学习、无监督学习和强化学习。
            监督学习使用标注数据进行训练，常见算法包括线性回归、决策树、支持向量机、神经网络等。
            无监督学习处理未标注数据，常见算法包括聚类、降维、关联规则等。
            强化学习通过与环境交互学习最优策略，常见应用包括游戏AI和机器人控制。
            深度学习是机器学习的一个子领域，使用多层神经网络进行特征学习。
            机器学习在图像识别、自然语言处理、推荐系统、金融预测等领域有广泛应用。""",
            doc_id="doc_003",
            metadata={"source": "ML教程", "category": "人工智能"}
        ),
    ]
    vector_store.add_documents(test_documents)

    # 创建检索器
    retriever = SimpleRetriever(vector_store)

    # -------- 演示1：基础压缩效果 --------
    print("\n【演示1】ContextualCompressor 基础压缩效果")
    print("-" * 60)

    compressor = ContextualCompressor(max_sentences=2)
    query = "Python的特点和应用场景"

    # 执行检索
    retrieved = retriever.retrieve(query, top_k=2)
    print(f"查询：{query}")
    print(f"检索到 {len(retrieved)} 个文档\n")

    # 压缩
    compressed = compressor.compress(retrieved, query)

    print("压缩效果对比：")
    print("-" * 60)
    for orig, comp in zip(retrieved, compressed):
        orig_len = len(orig.doc.content)
        comp_len = len(comp.doc.content)
        ratio = comp_len / orig_len if orig_len > 0 else 1.0

        print(f"\n文档ID：{orig.doc.doc_id}")
        print(f"  原始长度：{orig_len} 字符")
        print(f"  压缩后：{comp_len} 字符")
        print(f"  压缩比：{ratio:.2%}")
        print(f"  压缩后内容：{comp.doc.content[:100]}...")

    # -------- 演示2：压缩效果评估 --------
    print("\n\n【演示2】压缩效果评估")
    print("-" * 60)

    evaluator = CompressionEvaluator()
    eval_report = evaluator.evaluate_compression_quality(retrieved, compressed, query)

    print(f"总体压缩统计：")
    print(f"  原始总长度：{eval_report['total_original_length']} 字符")
    print(f"  压缩后总长度：{eval_report['total_compressed_length']} 字符")
    print(f"  总体压缩比：{eval_report['overall_compression_ratio']:.2%}")
    print(f"  Token节省：约 {eval_report['total_token_savings']} tokens")
    print(f"  节省比例：{eval_report['token_savings_percent']:.1f}%")

    # -------- 演示3：递归压缩 --------
    print("\n【演示3】RecursiveCompression 递归压缩")
    print("-" * 60)

    recursive_compressor = RecursiveCompression(
        initial_chunk_size=200,
        min_chunk_size=50,
        maxChunks_per_level=3
    )

    recursive_compressed = recursive_compressor.compress(retrieved, query)

    print(f"查询：{query}")
    print(f"递归压缩效果：")
    for comp in recursive_compressed:
        meta = comp.doc.metadata
        print(f"\n文档 {comp.doc.doc_id}：")
        print(f"  原始块数：{meta.get('chunks_before', 'N/A')}")
        print(f"  压缩后块数：{meta.get('chunks_after', 'N/A')}")
        print(f"  压缩比：{meta.get('compression_ratio', 'N/A')}")
        print(f"  内容：{comp.doc.content[:80]}...")

    # -------- 演示4：集成RAG系统 --------
    print("\n\n【演示4】CompressionAwareRAG 完整流程")
    print("-" * 60)

    # 创建支持压缩的RAG系统
    rag = CompressionAwareRAG(retriever, compressor)

    test_query = "Python和Java有什么区别？"
    result = rag.query(
        test_query,
        top_k=2,
        compression_method="contextual",
        return_compression_stats=True
    )

    print(f"查询：{test_query}")
    print(f"使用方法：{result['method']}")
    print(f"检索文档数：{result['retrieval_count']}")
    print(f"压缩比：{result['compression_ratio']:.2%}")
    print(f"Token节省：约 {result['token_savings']} tokens")
    print(f"\n生成答案：{result['answer']}")

    # -------- 演示5：压缩方法对比 --------
    print("\n\n【演示5】不同压缩方法对比")
    print("-" * 60)

    comparison = rag.compare_compression_methods("Python的特点", top_k=2)

    print(f"{'方法':<15} {'压缩比':<12} {'Token节省':<12} {'答案长度':<10}")
    print("-" * 50)
    for method, stats in comparison.items():
        method_name = {
            "contextual": "上下文压缩",
            "llm": "LLM压缩",
            "recursive": "递归压缩",
            "none": "不压缩"
        }.get(method, method)

        print(f"{method_name:<15} {stats['compression_ratio']:<12.2f} "
              f"{stats['token_savings']:<12} {stats['answer_length']:<10}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


def main():
    """主函数"""
    print("开始运行上下文压缩演示...")
    print()
    run_demo()


if __name__ == "__main__":
    main()
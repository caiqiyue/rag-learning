"""
RAG 质量评估体系 - 企业级 RAG 评估完整实现

本模块实现了 RAG 系统评估的核心功能，包括：
1. 检索质量评估：召回率、精确率、MRR、NDCG
2. 生成质量评估：Faithfulness、Answer Relevance、Context Precision/Recall
3. RAGAS 评估框架集成
4. Bad Case 分析与优化建议

依赖安装：
    pip install ragas datasets numpy

关联知识框架：
    - kp_016: 检索质量评估指标 - 召回率、精确率、命中率、MRR、NDCG
    - kp_017: 生成质量评估指标 - Faithfulness、Answer Relevancy、Contextual Precision/Recall
    - kp_018: 评估框架与参数调优 - DeepEval和Trulens评估框架
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# ============================================================================
# 第一部分：数据结构定义
# ============================================================================

class ErrorType(Enum):
    """RAG 系统错误类型枚举"""
    # 检索相关错误
    RETRIEVAL_EMPTY = "retrieval_empty"           # 召回为空
    RETRIEVAL_IRRELEVANT = "retrieval_irrelevant" # 召回不相关
    RETRIEVAL_WRONG = "retrieval_wrong"           # 召回了错误内容
    RETRIEVAL_INCOMPLETE = "retrieval_incomplete" # 召回不完整

    # 生成相关错误
    HALLUCINATION = "hallucination"               # 幻觉（编造信息）
    INCOMPLETE = "incomplete"                     # 回答不完整
    CONTRADICT = "contradict"                     # 回答矛盾
    FORMAT_ERROR = "format_error"                 # 格式错误
    OFF_TOPIC = "off_topic"                       # 答非所问

    # 整体问题
    TIMEOUT = "timeout"                          # 响应超时
    CONTEXT_OVERFLOW = "context_overflow"         # 上下文溢出

    # 无错误
    NONE = "none"


class Severity(Enum):
    """错误严重程度枚举"""
    CRITICAL = "critical"  # 严重错误，影响用户使用
    HIGH = "high"          # 高优先级问题
    MEDIUM = "medium"      # 中等优先级
    LOW = "low"            # 低优先级，可后续优化


@dataclass
class Document:
    """
    文档数据结构

    Attributes:
        content: 文档内容
        doc_id: 文档唯一标识符
        metadata: 文档元数据（如来源、类型等）
        relevance_score: 相关性分数（可选）
    """
    content: str
    doc_id: str
    metadata: Dict = field(default_factory=dict)
    relevance_score: Optional[float] = None


@dataclass
class TestCase:
    """
    测试用例数据结构

    一个完整的测试用例包含：用户问题、检索结果、生成回答、标准答案

    Attributes:
        query: 用户查询问题
        retrieved_docs: 检索返回的文档列表
        answer: RAG 系统生成的回答
        ground_truth: 标准/期望回答
        relevant_doc_ids: 真正相关的文档ID列表（用于评估检索质量）
        context: 供给 LLM 的上下文内容
    """
    query: str
    retrieved_docs: List[Document]
    answer: str
    ground_truth: str
    relevant_doc_ids: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """
    检索质量指标数据结构

    包含评估检索系统效果的核心指标：
    - 精确率（Precision）：检索结果中相关文档的比例
    - 召回率（Recall）：相关文档被检索到的比例
    - F1 分数：精确率和召回率的调和平均
    - MRR（Mean Reciprocal Rank）：第一个相关结果的倒数排名均值
    - NDCG（Normalized Discounted Cumulative Gain）：归一化折损累计增益

    Attributes:
        precision: 精确率，范围 [0, 1]
        recall: 召回率，范围 [0, 1]
        f1: F1 分数，范围 [0, 1]
        mrr: 平均倒数排名，范围 [0, 1]
        ndcg: 归一化折损累计增益，范围 [0, 1]
    """
    precision: float
    recall: float
    f1: float
    mrr: float
    ndcg: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mrr": self.mrr,
            "ndcg": self.ndcg
        }


@dataclass
class GenerationMetrics:
    """
    生成质量指标数据结构

    基于 RAGAS 框架的核心指标：
    - Faithfulness（忠实度）：回答是否忠实于检索到的内容，没有编造
    - Answer Relevance（回答相关性）：回答与问题的相关程度
    - Context Precision（上下文精确度）：检索上下文中有多少相关文档
    - Context Recall（上下文召回率）：检索上下文包含了多少标准答案中的信息

    Attributes:
        faithfulness: 忠实度得分，范围 [0, 1]
        answer_relevance: 回答相关性得分，范围 [0, 1]
        context_precision: 上下文精确度得分，范围 [0, 1]
        context_recall: 上下文召回率得分，范围 [0, 1]
    """
    faithfulness: float
    answer_relevance: float
    context_precision: float = 0.0
    context_recall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall
        }


@dataclass
class EvaluationResult:
    """
    完整评估结果数据结构

    包含检索质量指标、生成质量指标以及 Bad Case 分析结果

    Attributes:
        retrieval_metrics: 检索质量指标
        generation_metrics: 生成质量指标
        overall_score: 整体评分（综合检索和生成）
        bad_cases: Bad Case 列表
        total_cases: 测试用例总数
    """
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    overall_score: float = 0.0
    bad_cases: List[Dict] = field(default_factory=list)
    total_cases: int = 0


# ============================================================================
# 第二部分：检索质量评估器
# ============================================================================

class RetrievalEvaluator:
    """
    检索质量评估器

    负责评估 RAG 系统中检索阶段的质量，计算以下指标：
    - 精确率（Precision）：检索结果中相关文档的比例
    - 召回率（Recall）：相关文档被检索到的比例
    - F1 分数：精确率和召回率的调和平均
    - MRR（Mean Reciprocal Rank）：第一个相关结果的位置
    - NDCG（Normalized Discounted Cumulative Gain）：考虑排序位置的评估指标

    使用方法：
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate(
            query="RAG 是什么",
            retrieved_docs=[doc1, doc2, doc3],
            relevant_doc_ids=["doc_0", "doc_1"]
        )
    """

    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str],
        relevance_scores: Optional[List[float]] = None
    ) -> RetrievalMetrics:
        """
        评估检索质量

        Args:
            query: 用户查询问题
            retrieved_docs: 检索返回的文档列表
            relevant_doc_ids: 真正相关的文档 ID 列表（ground truth）
            relevance_scores: 可选，相关性分数列表（用于 NDCG 计算）

        Returns:
            RetrievalMetrics: 包含各项检索指标的对象

        计算公式：
            Precision = TP / (TP + FP) = 相关文档数 / 检索总数
            Recall = TP / (TP + FN) = 相关文档数 / 相关文档总数
            F1 = 2 * Precision * Recall / (Precision + Recall)
            MRR = mean(1 / rank_i) 其中 rank_i 是第 i 个相关文档的位置
        """
        # 提取检索到的文档 ID
        retrieved_ids = [doc.doc_id for doc in retrieved_docs]

        # ========== 精确率计算 ==========
        # 精确率 = 检索结果中相关文档的数量 / 检索总数
        # 例如：检索了 10 个文档，其中 7 个是相关的，精确率 = 0.7
        true_positives = len(set(retrieved_ids) & set(relevant_doc_ids))
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0

        # ========== 召回率计算 ==========
        # 召回率 = 检索到的相关文档数 / 实际相关文档总数
        # 例如：共有 10 个相关文档，只检索到了 7 个，召回率 = 0.7
        recall = true_positives / len(relevant_doc_ids) if relevant_doc_ids else 0.0

        # ========== F1 分数计算 ==========
        # F1 是精确率和召回率的调和平均，避免极端值
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # ========== MRR（平均倒数排名）计算 ==========
        # MRR 衡量第一个相关结果的位置好坏
        # 如果第一个结果就是相关的，MRR = 1；第二个相关，MRR = 0.5；以此类推
        mrr = self._calculate_mrr(retrieved_ids, relevant_doc_ids)

        # ========== NDCG 计算 ==========
        # NDCG 考虑排序位置，越靠前的相关结果权重越高
        ndcg = self._calculate_ndcg(retrieved_ids, relevant_doc_ids, relevance_scores)

        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            mrr=mrr,
            ndcg=ndcg
        )

    def _calculate_mrr(
        self,
        retrieved_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        计算 MRR（Mean Reciprocal Rank）

        MRR = (1 / rank_1 + 1 / rank_2 + ... + 1 / rank_k) / k
        其中 rank_i 是第 i 个相关文档在检索结果中的位置（从 1 开始）

        Args:
            retrieved_ids: 检索到的文档 ID 列表
            relevant_doc_ids: 真正相关的文档 ID 列表

        Returns:
            float: MRR 分数，范围 [0, 1]
        """
        for i, doc_id in enumerate(retrieved_ids):
            # 找到第一个相关文档，返回其倒数排名
            # 位置从 1 开始计数，所以 i+1
            if doc_id in relevant_doc_ids:
                return 1.0 / (i + 1)

        # 没有找到相关文档，返回 0
        return 0.0

    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_doc_ids: List[str],
        relevance_scores: Optional[List[float]] = None
    ) -> float:
        """
        计算 NDCG（Normalized Discounted Cumulative Gain）

        DCG = sum(rel_i / log2(i + 1)) for i in range(len(retrieved_docs))
        NDCG = DCG / IDCG（理想情况下的 DCG）

        Args:
            retrieved_ids: 检索到的文档 ID 列表
            relevant_doc_ids: 真正相关的文档 ID 列表
            relevance_scores: 可选的相关性分数列表

        Returns:
            float: NDCG 分数，范围 [0, 1]
        """
        if not retrieved_ids or not relevant_doc_ids:
            return 0.0

        # 如果没有提供相关性分数，使用二元分数（相关=1，不相关=0）
        if relevance_scores is None:
            relevance_scores = [
                1.0 if doc_id in relevant_doc_ids else 0.0
                for doc_id in retrieved_ids
            ]

        # ========== 计算 DCG ==========
        # DCG = sum(rel_i / log2(i + 2))
        # 从位置 1 开始，使用 log2(i+2) 作为折扣因子
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / np.log2(i + 2)

        # ========== 计算 IDCG ==========
        # IDCG 是理想排序下的 DCG，即把所有相关文档排在前面
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += rel / np.log2(i + 2)

        # ========== 计算 NDCG ==========
        if idcg > 0:
            return dcg / idcg
        return 0.0


# ============================================================================
# 第三部分：生成质量评估器
# ============================================================================

class GenerationEvaluator:
    """
    生成质量评估器

    负责评估 RAG 系统中生成阶段的质量，基于 RAGAS 框架计算以下指标：
    - Faithfulness（忠实度）：回答是否忠实于检索到的上下文
    - Answer Relevance（回答相关性）：回答与问题的相关程度
    - Context Precision（上下文精确度）：检索上下文的精确程度
    - Context Recall（上下文召回率）：检索上下文包含标准答案信息的程度

    注意：在实际项目中，这些指标可以通过调用 LLM 来计算，
         这里提供的实现适用于没有 LLM API 的场景。

    使用方法：
        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate(
            query="RAG 是什么",
            answer="RAG 是检索增强生成技术...",
            context=["RAG 是...", "它由 Facebook 提出..."],
            ground_truth="RAG 是检索增强生成..."
        )
    """

    def __init__(self, use_llm: bool = False, llm_model=None):
        """
        初始化生成质量评估器

        Args:
            use_llm: 是否使用 LLM 进行评估（需要配置 LLM API）
            llm_model: LLM 模型实例（当 use_llm=True 时需要提供）
        """
        self.use_llm = use_llm
        self.llm_model = llm_model

    def evaluate(
        self,
        query: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> GenerationMetrics:
        """
        评估生成质量

        Args:
            query: 用户查询问题
            answer: RAG 系统生成的回答
            context: 检索到的上下文列表
            ground_truth: 标准/期望回答（用于计算 Context Recall）

        Returns:
            GenerationMetrics: 包含各项生成质量指标的对象

        计算说明：
            - Faithfulness：检查回答中的事实是否都能在上下文中找到依据
            - Answer Relevance：计算回答与问题之间的语义相似度
            - Context Precision：评估检索到的上下文中有多少是真正相关的
            - Context Recall：评估检索上下文是否覆盖了标准答案的信息
        """
        # 合并上下文文本
        context_text = " ".join(context)

        # ========== 计算 Faithfulness（忠实度）==========
        # 忠实度衡量回答是否基于检索到的内容，没有编造信息
        # 简化计算：检查回答中的关键实体/数字是否能在上下文中找到
        faithfulness = self._calculate_faithfulness(answer, context_text)

        # ========== 计算 Answer Relevance（回答相关性）==========
        # 回答相关性衡量回答是否针对问题，有用程度如何
        answer_relevance = self._calculate_answer_relevance(query, answer, context_text)

        # ========== 计算 Context Precision（上下文精确度）==========
        # 上下文精确度衡量检索到的上下文中相关内容的比例
        context_precision = self._calculate_context_precision(context, context_text)

        # ========== 计算 Context Recall（上下文召回率）==========
        # 上下文召回率衡量检索上下文是否覆盖了标准答案的信息
        # 只有提供 ground_truth 时才能计算
        context_recall = 0.0
        if ground_truth:
            context_recall = self._calculate_context_recall(context, ground_truth)

        return GenerationMetrics(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall
        )

    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        计算忠实度（Faithfulness）

        忠实度评估回答中的信息是否都能在上下文中找到支持。
        简化实现：计算回答中在上下文中出现的词占总词数的比例。

        Args:
            answer: 生成的回答
            context: 检索到的上下文

        Returns:
            float: 忠实度分数，范围 [0, 1]
        """
        if not answer or not context:
            return 0.0

        # 提取回答中的关键信息（这里简化为词集合）
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        # 计算重叠度（过滤掉常见停用词）
        stop_words = {"的", "是", "在", "了", "和", "与", "或", "以及", "这", "那", "有"}
        answer_keywords = answer_words - stop_words
        context_keywords = context_words - stop_words

        if not answer_keywords:
            return 1.0  # 回答为空时，假定是忠实的

        # 计算有多少关键信息能在上下文中找到
        overlap = len(answer_keywords & context_keywords)
        faithfulness = overlap / len(answer_keywords)

        return faithfulness

    def _calculate_answer_relevance(
        self,
        query: str,
        answer: str,
        context: str
    ) -> float:
        """
        计算回答相关性（Answer Relevance）

        回答相关性评估回答是否真正针对问题，是否完整解决了用户疑问。

        简化实现：分析回答长度和内容覆盖率

        Args:
            query: 用户问题
            answer: 生成的回答
            context: 检索上下文

        Returns:
            float: 回答相关性分数，范围 [0, 1]
        """
        if not answer:
            return 0.0

        # 长度得分：回答长度适中（不太短也不太长）得分更高
        # 假设理想回答长度在 100-500 字之间
        length = len(answer)
        if length < 50:
            length_score = length / 50 * 0.5  # 太短，得分上限 0.5
        elif length > 1000:
            length_score = max(0.5, 1.0 - (length - 1000) / 1000 * 0.3)
        else:
            length_score = 1.0

        # 重复度检查：回答重复内容多会降低得分
        words = answer.split()
        unique_ratio = len(set(words)) / len(words) if words else 0

        # 上下文利用率：回答应该包含上下文中的关键信息
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        context_utilization = len(answer_words & context_words) / len(context_words) if context_words else 0

        # 综合得分
        relevance = (
            length_score * 0.4 +
            unique_ratio * 0.3 +
            context_utilization * 0.3
        )

        return min(1.0, max(0.0, relevance))

    def _calculate_context_precision(
        self,
        context: List[str],
        context_text: str
    ) -> float:
        """
        计算上下文精确度（Context Precision）

        上下文精确度评估检索到的上下文中，真正相关内容的比例。

        Args:
            context: 检索到的上下文列表
            context_text: 合并后的上下文文本

        Returns:
            float: 上下文精确度分数，范围 [0, 1]
        """
        if not context:
            return 0.0

        # 简化实现：检查上下文是否包含实质性的信息
        # 如果上下文重复度高或太短，精确度会降低

        total_length = sum(len(doc) for doc in context)
        avg_length = total_length / len(context)

        # 平均长度过短可能表示内容质量低
        if avg_length < 50:
            return 0.3

        # 检查上下文之间的重复度
        if len(context) > 1:
            first_doc_words = set(context[0].lower().split())
            repeat_count = 0
            for doc in context[1:]:
                doc_words = set(doc.lower().split())
                overlap = len(first_doc_words & doc_words)
                if overlap / len(first_doc_words) > 0.7:
                    repeat_count += 1

            # 重复度高会降低精确度
            repeat_ratio = repeat_count / (len(context) - 1) if len(context) > 1 else 0
            precision = 1.0 - repeat_ratio * 0.5
        else:
            precision = 1.0

        return precision

    def _calculate_context_recall(
        self,
        context: List[str],
        ground_truth: str
    ) -> float:
        """
        计算上下文召回率（Context Recall）

        上下文召回率评估检索到的上下文是否覆盖了标准答案的信息。

        Args:
            context: 检索到的上下文列表
            ground_truth: 标准/期望回答

        Returns:
            float: 上下文召回率分数，范围 [0, 1]
        """
        if not ground_truth or not context:
            return 0.0

        # 提取标准答案中的关键信息
        gt_words = set(ground_truth.lower().split())
        stop_words = {"的", "是", "在", "了", "和", "与", "或", "以及", "这", "那"}
        gt_keywords = gt_words - stop_words

        if not gt_keywords:
            return 1.0

        # 检查检索上下文是否覆盖了这些关键信息
        context_text = " ".join(context).lower()
        context_words = set(context_text.split())

        covered = sum(1 for kw in gt_keywords if kw in context_words)
        recall = covered / len(gt_keywords)

        return recall


# ============================================================================
# 第四部分：RAGAS 评估框架
# ============================================================================

class RAGASEvaluator:
    """
    RAGAS 评估框架集成器

    RAGAS（Retrieval-Augmented Generation Assessment）是一个专门用于评估
    RAG 系统的开源框架，提供标准化的评估流程和指标计算。

    主要功能：
    - 集成 RetrievalEvaluator 和 GenerationEvaluator
    - 支持批量评估多个测试用例
    - 生成详细的评估报告

    使用方法：
        evaluator = RAGASEvaluator()
        results = evaluator.evaluate_batch(test_cases)
        evaluator.print_report(results)
    """

    def __init__(self, use_llm: bool = False, llm_model=None):
        """
        初始化 RAGAS 评估器

        Args:
            use_llm: 是否使用 LLM 进行评估
            llm_model: LLM 模型实例
        """
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator(use_llm, llm_model)

    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """
        评估单个测试用例

        Args:
            test_case: 测试用例对象

        Returns:
            EvaluationResult: 包含检索和生成质量指标的评估结果
        """
        # 提取 relevant_doc_ids
        relevant_ids = test_case.relevant_doc_ids or []

        # ========== 检索质量评估 ==========
        retrieval_metrics = self.retrieval_evaluator.evaluate(
            query=test_case.query,
            retrieved_docs=test_case.retrieved_docs,
            relevant_doc_ids=relevant_ids
        )

        # ========== 生成质量评估 ==========
        context_list = [doc.content for doc in test_case.retrieved_docs]
        generation_metrics = self.generation_evaluator.evaluate(
            query=test_case.query,
            answer=test_case.answer,
            context=context_list,
            ground_truth=test_case.ground_truth
        )

        # ========== 计算整体评分 ==========
        # 整体评分 = 0.4 * 检索评分 + 0.6 * 生成评分
        # 生成质量通常更重要，因此权重更高
        retrieval_score = (retrieval_metrics.f1 + retrieval_metrics.mrr) / 2
        generation_score = (
            generation_metrics.faithfulness +
            generation_metrics.answer_relevance
        ) / 2
        overall_score = 0.4 * retrieval_score + 0.6 * generation_score

        return EvaluationResult(
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            overall_score=overall_score,
            total_cases=1
        )

    def evaluate_batch(self, test_cases: List[TestCase]) -> Dict:
        """
        批量评估多个测试用例

        Args:
            test_cases: 测试用例列表

        Returns:
            Dict: 包含平均指标和详细结果的字典
        """
        if not test_cases:
            return {
                "retrieval_metrics": RetrievalMetrics(0, 0, 0, 0, 0),
                "generation_metrics": GenerationMetrics(0, 0, 0, 0),
                "overall_score": 0.0,
                "total_cases": 0
            }

        # 存储所有评估结果
        retrieval_results = {
            "precision": [], "recall": [], "f1": [], "mrr": [], "ndcg": []
        }
        generation_results = {
            "faithfulness": [],
            "answer_relevance": [],
            "context_precision": [],
            "context_recall": []
        }
        overall_scores = []

        # 逐个评估
        for case in test_cases:
            result = self.evaluate(case)

            # 收集检索指标
            for key in retrieval_results:
                retrieval_results[key].append(
                    result.retrieval_metrics.to_dict()[key]
                )

            # 收集生成指标
            for key in generation_results:
                generation_results[key].append(
                    result.generation_metrics.to_dict()[key]
                )

            overall_scores.append(result.overall_score)

        # 计算平均值
        avg_retrieval = {
            key: np.mean(values) if values else 0.0
            for key, values in retrieval_results.items()
        }
        avg_generation = {
            key: np.mean(values) if values else 0.0
            for key, values in generation_results.items()
        }

        return {
            "retrieval_metrics": RetrievalMetrics(
                precision=avg_retrieval["precision"],
                recall=avg_retrieval["recall"],
                f1=avg_retrieval["f1"],
                mrr=avg_retrieval["mrr"],
                ndcg=avg_retrieval["ndcg"]
            ),
            "generation_metrics": GenerationMetrics(
                faithfulness=avg_generation["faithfulness"],
                answer_relevance=avg_generation["answer_relevance"],
                context_precision=avg_generation["context_precision"],
                context_recall=avg_generation["context_recall"]
            ),
            "overall_score": np.mean(overall_scores) if overall_scores else 0.0,
            "total_cases": len(test_cases)
        }

    def print_report(self, results: Dict) -> None:
        """
        打印评估报告

        Args:
            results: evaluate_batch 返回的结果字典
        """
        print("=" * 60)
        print("RAG 质量评估报告")
        print("=" * 60)
        print(f"\n总共评估 {results['total_cases']} 个测试用例")
        print(f"整体评分: {results['overall_score']:.4f}")

        print("\n--- 检索质量指标 ---")
        rm = results["retrieval_metrics"]
        print(f"  精确率 (Precision):    {rm.precision:.4f}")
        print(f"  召回率 (Recall):       {rm.recall:.4f}")
        print(f"  F1 分数 (F1):          {rm.f1:.4f}")
        print(f"  平均倒数排名 (MRR):    {rm.mrr:.4f}")
        print(f"  NDCG:                  {rm.ndcg:.4f}")

        print("\n--- 生成质量指标 ---")
        gm = results["generation_metrics"]
        print(f"  忠实度 (Faithfulness):      {gm.faithfulness:.4f}")
        print(f"  回答相关性 (Answer Relevancy): {gm.answer_relevance:.4f}")
        print(f"  上下文精确度 (Context Precision): {gm.context_precision:.4f}")
        print(f"  上下文召回率 (Context Recall): {gm.context_recall:.4f}")

        print("\n" + "=" * 60)


# ============================================================================
# 第五部分：Bad Case 分析器
# ============================================================================

class BadCaseAnalyzer:
    """
    Bad Case 分析器

    Bad Case（坏案例）是指 RAG 系统回答错误或不理想的案例。
    通过分析 Bad Case，可以定位系统问题，指导优化方向。

    主要功能：
    - 自动分类错误类型（检索问题、生成问题、整体问题）
    - 统计各类问题占比
    - 生成优化建议

    使用方法：
        analyzer = BadCaseAnalyzer()
        analysis = analyzer.analyze(test_results)
        analyzer.print_analysis(analysis)
    """

    def __init__(self):
        """初始化 Bad Case 分析器"""
        # 定义错误分类体系
        self.categories = {
            "retrieval": {
                "empty": "召回为空（没有检索到任何内容）",
                "irrelevant": "召回不相关（检索到的内容与问题无关）",
                "wrong": "召回了错误内容（检索到了误导性信息）",
                "incomplete": "召回不完整（遗漏了重要的相关信息）"
            },
            "generation": {
                "hallucination": "幻觉（回答中包含上下文中没有的信息）",
                "incomplete": "回答不完整（没有完整回答问题）",
                "contradict": "回答矛盾（回答内容与上下文矛盾）",
                "format_error": "格式错误（输出格式不符合要求）",
                "off_topic": "答非所问（没有针对问题回答）"
            },
            "other": {
                "timeout": "响应超时",
                "context_overflow": "上下文溢出"
            }
        }

    def analyze(self, test_results: List[Dict]) -> Dict:
        """
        分析 Bad Cases

        Args:
            test_results: 测试结果列表，每个元素包含：
                - query: 用户问题
                - answer: 生成的回答
                - expected: 期望的回答
                - error_type: 错误类型
                - severity: 严重程度（critical/high/medium/low）
                - details: 详细错误描述

        Returns:
            Dict: 包含分类统计和详细案例的分析结果
        """
        if not test_results:
            return {
                "total_cases": 0,
                "bad_case_count": 0,
                "categories": {},
                "detailed_cases": {},
                "percentages": {}
            }

        # 初始化统计结构
        stats = {
            "retrieval": {k: 0 for k in self.categories["retrieval"].keys()},
            "generation": {k: 0 for k in self.categories["generation"].keys()},
            "other": {k: 0 for k in self.categories["other"].keys()}
        }

        # 存储详细案例
        detailed_cases = {
            "retrieval": {k: [] for k in self.categories["retrieval"].keys()},
            "generation": {k: [] for k in self.categories["generation"].keys()},
            "other": {k: [] for k in self.categories["other"].keys()}
        }

        # 统计
        total = len(test_results)
        for result in test_results:
            error_type = result.get("error_type", "none")
            severity = result.get("severity", "medium")

            # 分类并记录
            if error_type in stats["retrieval"]:
                stats["retrieval"][error_type] += 1
                detailed_cases["retrieval"][error_type].append(result)
            elif error_type in stats["generation"]:
                stats["generation"][error_type] += 1
                detailed_cases["generation"][error_type].append(result)
            elif error_type in stats["other"]:
                stats["other"][error_type] += 1
                detailed_cases["other"][error_type].append(result)

        # 计算占比
        percentages = {
            category: {
                sub_category: (count / total * 100) if total > 0 else 0
                for sub_category, count in sub_categories.items()
            }
            for category, sub_categories in stats.items()
        }

        # 坏案例总数
        bad_case_count = sum(
            sum(sub.values()) for sub in stats.values()
        )

        return {
            "total_cases": total,
            "bad_case_count": bad_case_count,
            "bad_case_rate": bad_case_count / total if total > 0 else 0,
            "categories": stats,
            "percentages": percentages,
            "detailed_cases": detailed_cases
        }

    def generate_optimization_suggestions(self, analysis: Dict) -> Dict[str, List[str]]:
        """
        根据 Bad Case 分析结果生成优化建议

        Args:
            analysis: analyze() 方法返回的分析结果

        Returns:
            Dict[str, List[str]]: 各类问题的优化建议
        """
        suggestions = {
            "retrieval": [],
            "generation": [],
            "overall": []
        }

        categories = analysis.get("categories", {})
        percentages = analysis.get("percentages", {})

        # ========== 检索问题优化建议 ==========
        retrieval_stats = categories.get("retrieval", {})
        retrieval_pct = percentages.get("retrieval", {})

        if retrieval_stats.get("empty", 0) > 0:
            pct = retrieval_pct.get("empty", 0)
            suggestions["retrieval"].append(
                f"【召回为空】占比 {pct:.1f}%，建议："
                "1) 检查 Embedding 模型是否适合当前领域 "
                "2) 优化文档分割策略，避免关键信息被切断 "
                "3) 考虑使用 HyDE 等高级检索技术"
            )

        if retrieval_stats.get("irrelevant", 0) > 0:
            pct = retrieval_pct.get("irrelevant", 0)
            suggestions["retrieval"].append(
                f"【召回不相关】占比 {pct:.1f}%，建议："
                "1) 调整 chunk_size，减少单块文档长度 "
                "2) 使用更精确的检索器（如 BM25） "
                "3) 添加查询改写（Query Rewriting）"
            )

        if retrieval_stats.get("wrong", 0) > 0:
            pct = retrieval_pct.get("wrong", 0)
            suggestions["retrieval"].append(
                f"【召回错误】占比 {pct:.1f}%，建议："
                "1) 审查知识库内容质量，清理错误信息 "
                "2) 增加元数据过滤，缩小检索范围 "
                "3) 使用 Cross-Encoder 进行重排序"
            )

        # ========== 生成问题优化建议 ==========
        gen_stats = categories.get("generation", {})
        gen_pct = percentages.get("generation", {})

        if gen_stats.get("hallucination", 0) > 0:
            pct = gen_pct.get("hallucination", 0)
            suggestions["generation"].append(
                f"【幻觉】占比 {pct:.1f}%，建议："
                "1) 在提示中强调'仅根据提供的上下文回答' "
                "2) 增加 Faithfulness 评估反馈 "
                "3) 使用结构化输出减少自由发挥空间"
            )

        if gen_stats.get("incomplete", 0) > 0:
            pct = gen_pct.get("incomplete", 0)
            suggestions["generation"].append(
                f"【回答不完整】占比 {pct:.1f}%，建议："
                "1) 增加检索的 top_k 数量 "
                "2) 调整提示模板，要求'完整回答' "
                "3) 使用 Chain-of-Thought 提示策略"
            )

        if gen_stats.get("contradict", 0) > 0:
            pct = gen_pct.get("contradict", 0)
            suggestions["generation"].append(
                f"【回答矛盾】占比 {pct:.1f}%，建议："
                "1) 检索后增加冲突检测逻辑 "
                "2) 使用更可靠的 LLM 版本 "
                "3) 在提示中加入'不要与上下文矛盾'的约束"
            )

        # ========== 整体优化建议 ==========
        if analysis.get("bad_case_rate", 0) > 0.3:
            suggestions["overall"].append(
                "坏案例率超过 30%，建议进行系统性审查："
                "1) 重新评估知识库质量和覆盖范围 "
                "2) 检查端到端流程中的信息损失点 "
                "3) 考虑引入人工审核环节"
            )

        return suggestions

    def print_analysis(self, analysis: Dict) -> None:
        """
        打印 Bad Case 分析报告

        Args:
            analysis: analyze() 方法返回的分析结果
        """
        print("=" * 70)
        print("RAG Bad Case 分析报告")
        print("=" * 70)

        print(f"\n总共测试用例: {analysis['total_cases']}")
        print(f"Bad Case 数量: {analysis['bad_case_count']}")
        print(f"坏案例率: {analysis['bad_case_rate']*100:.2f}%")

        print("\n" + "-" * 70)
        print("一、检索问题统计")
        print("-" * 70)
        retrieval_stats = analysis.get("categories", {}).get("retrieval", {})
        retrieval_pct = analysis.get("percentages", {}).get("retrieval", {})
        for error_type, count in retrieval_stats.items():
            pct = retrieval_pct.get(error_type, 0)
            desc = self.categories["retrieval"].get(error_type, "")
            print(f"  {error_type:15s}: {count:4d} 例 ({pct:5.2f}%) - {desc}")

        print("\n" + "-" * 70)
        print("二、生成问题统计")
        print("-" * 70)
        gen_stats = analysis.get("categories", {}).get("generation", {})
        gen_pct = analysis.get("percentages", {}).get("generation", {})
        for error_type, count in gen_stats.items():
            pct = gen_pct.get(error_type, 0)
            desc = self.categories["generation"].get(error_type, "")
            print(f"  {error_type:15s}: {count:4d} 例 ({pct:5.2f}%) - {desc}")

        print("\n" + "=" * 70)


# ============================================================================
# 第六部分：企业级评估流程
# ============================================================================

class RAGEvaluationPipeline:
    """
    企业级 RAG 评估流程

    提供完整的评估流程，包括：
    - 测试数据准备
    - 批量评估
    - Bad Case 分析
    - 优化建议生成
    - 评估报告输出

    使用方法：
        pipeline = RAGEvaluationPipeline()
        report = pipeline.run_full_evaluation(test_cases)
    """

    def __init__(self, use_llm: bool = False, llm_model=None):
        """
        初始化评估流程

        Args:
            use_llm: 是否使用 LLM 进行评估
            llm_model: LLM 模型实例
        """
        self.ragas_evaluator = RAGASEvaluator(use_llm, llm_model)
        self.bad_case_analyzer = BadCaseAnalyzer()

    def run_full_evaluation(
        self,
        test_cases: List[TestCase],
        include_bad_case_analysis: bool = True
    ) -> Dict:
        """
        运行完整评估流程

        Args:
            test_cases: 测试用例列表
            include_bad_case_analysis: 是否包含 Bad Case 分析

        Returns:
            Dict: 包含评估结果和分析报告的完整字典
        """
        print("开始 RAG 质量评估...")
        print(f"测试用例数量: {len(test_cases)}")

        # ========== 第一步：批量评估 ==========
        print("\n[1/3] 执行批量评估...")
        evaluation_results = self.ragas_evaluator.evaluate_batch(test_cases)

        # ========== 第二步：准备 Bad Case 分析数据 ==========
        bad_case_data = []
        for case in test_cases:
            # 检测是否有问题
            # 简化逻辑：答案过短或与上下文重叠度低视为有问题
            has_issue = False
            error_type = ErrorType.NONE.value
            severity = Severity.LOW.value

            # 检查回答长度
            if len(case.answer) < 20:
                has_issue = True
                error_type = ErrorType.INCOMPLETE.value
                severity = Severity.MEDIUM.value

            # 检查是否答非所问（简化检查）
            if case.query and case.answer:
                query_keywords = set(case.query.lower().split())
                answer_keywords = set(case.answer.lower().split())
                overlap = len(query_keywords & answer_keywords)
                if overlap < 2 and len(query_keywords) > 3:
                    has_issue = True
                    error_type = ErrorType.OFF_TOPIC.value
                    severity = Severity.HIGH.value

            if has_issue:
                bad_case_data.append({
                    "query": case.query,
                    "answer": case.answer,
                    "expected": case.ground_truth,
                    "error_type": error_type,
                    "severity": severity
                })

        # ========== 第三步：Bad Case 分析 ==========
        if include_bad_case_analysis and bad_case_data:
            print("\n[2/3] 分析 Bad Cases...")
            bad_case_analysis = self.bad_case_analyzer.analyze(bad_case_data)
            optimization_suggestions = (
                self.bad_case_analyzer.generate_optimization_suggestions(
                    bad_case_analysis
                )
            )
        else:
            print("\n[2/3] 跳过 Bad Case 分析（无坏案例）")
            bad_case_analysis = None
            optimization_suggestions = None

        # ========== 第四步：生成完整报告 ==========
        print("\n[3/3] 生成评估报告...")
        report = {
            "evaluation_summary": {
                "total_cases": evaluation_results["total_cases"],
                "overall_score": evaluation_results["overall_score"],
                "retrieval_metrics": evaluation_results["retrieval_metrics"].to_dict(),
                "generation_metrics": evaluation_results["generation_metrics"].to_dict()
            },
            "bad_case_analysis": bad_case_analysis,
            "optimization_suggestions": optimization_suggestions
        }

        return report

    def print_final_report(self, report: Dict) -> None:
        """
        打印最终评估报告

        Args:
            report: run_full_evaluation 返回的报告字典
        """
        print("\n")
        print("=" * 70)
        print("RAG 质量评估最终报告")
        print("=" * 70)

        # 评估摘要
        summary = report["evaluation_summary"]
        print(f"\n总共评估用例数: {summary['total_cases']}")
        print(f"整体评分: {summary['overall_score']:.4f}")

        print("\n--- 检索质量 ---")
        rm = summary["retrieval_metrics"]
        print(f"  精确率: {rm['precision']:.4f}")
        print(f"  召回率: {rm['recall']:.4f}")
        print(f"  F1: {rm['f1']:.4f}")
        print(f"  MRR: {rm['mrr']:.4f}")
        print(f"  NDCG: {rm['ndcg']:.4f}")

        print("\n--- 生成质量 ---")
        gm = summary["generation_metrics"]
        print(f"  忠实度: {gm['faithfulness']:.4f}")
        print(f"  回答相关性: {gm['answer_relevance']:.4f}")
        print(f"  上下文精确度: {gm['context_precision']:.4f}")
        print(f"  上下文召回率: {gm['context_recall']:.4f}")

        # Bad Case 分析
        if report.get("bad_case_analysis"):
            print("\n--- Bad Case 分析 ---")
            bca = report["bad_case_analysis"]
            print(f"  坏案例数: {bca['bad_case_count']}/{bca['total_cases']}")
            print(f"  坏案例率: {bca['bad_case_rate']*100:.2f}%")

        # 优化建议
        if report.get("optimization_suggestions"):
            print("\n--- 优化建议 ---")
            suggestions = report["optimization_suggestions"]
            if suggestions.get("retrieval"):
                print("\n  [检索优化]")
                for s in suggestions["retrieval"]:
                    print(f"    - {s}")
            if suggestions.get("generation"):
                print("\n  [生成优化]")
                for s in suggestions["generation"]:
                    print(f"    - {s}")
            if suggestions.get("overall"):
                print("\n  [整体优化]")
                for s in suggestions["overall"]:
                    print(f"    - {s}")

        print("\n" + "=" * 70)


# ============================================================================
# 第七部分：使用示例与测试
# ============================================================================

def create_sample_test_cases() -> List[TestCase]:
    """
    创建示例测试用例，用于演示评估功能

    Returns:
        List[TestCase]: 示例测试用例列表
    """
    test_cases = [
        # 用例 1：正常案例
        TestCase(
            query="RAG 是什么意思？",
            retrieved_docs=[
                Document(
                    content="RAG 是 Retrieval-Augmented Generation（检索增强生成）的缩写。"
                            "它是一种结合检索系统和生成模型的技术，可以提高生成内容的准确性。",
                    doc_id="doc_0"
                ),
                Document(
                    content="RAG 由 Facebook AI Research 在 2020 年提出。",
                    doc_id="doc_1"
                )
            ],
            answer="RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写，"
                   "它是一种结合检索系统和生成模型的技术。",
            ground_truth="RAG 是检索增强生成技术，是一种结合检索和生成的方法。",
            relevant_doc_ids=["doc_0", "doc_1"],
            context=["RAG 是检索增强生成...", "RAG 由 Facebook 提出..."]
        ),

        # 用例 2：回答不完整
        TestCase(
            query="请详细介绍一下 RAG 的工作原理？",
            retrieved_docs=[
                Document(
                    content="RAG 工作流程包括两个阶段：离线索引阶段和在线检索阶段。",
                    doc_id="doc_0"
                )
            ],
            answer="RAG 是一个检索系统。",
            ground_truth="RAG 工作流程包括两个阶段：离线索引阶段（构建向量数据库）和在线检索阶段（检索相关文档并送给生成器）。",
            relevant_doc_ids=["doc_0"],
            context=["RAG 工作流程包括两个阶段..."]
        ),

        # 用例 3：存在幻觉
        TestCase(
            query="RAG 是谁发明的？",
            retrieved_docs=[
                Document(
                    content="RAG 由 Facebook AI Research 在 2020 年提出。",
                    doc_id="doc_0"
                )
            ],
            answer="RAG 是由 Google 在 2019 年发明的。",
            ground_truth="RAG 由 Facebook AI Research 在 2020 年提出。",
            relevant_doc_ids=["doc_0"],
            context=["RAG 由 Facebook AI Research 在 2020 年提出..."]
        ),

        # 用例 4：检索不相关
        TestCase(
            query="Python 怎么定义函数？",
            retrieved_docs=[
                Document(
                    content="Java 是一种面向对象的编程语言。",
                    doc_id="doc_0"
                ),
                Document(
                    content="JavaScript 是网页开发的主流语言。",
                    doc_id="doc_1"
                )
            ],
            answer="Python 是一种编程语言。",
            ground_truth="Python 定义函数使用 def 关键字，如：def my_func(): pass",
            relevant_doc_ids=[],
            context=["Java 是一种编程语言...", "JavaScript 是网页开发语言..."]
        ),

        # 用例 5：完全正常
        TestCase(
            query="什么是向量数据库？",
            retrieved_docs=[
                Document(
                    content="向量数据库是一种专门用于存储和检索高维向量的数据库系统，"
                            "常用于相似性搜索和机器学习应用。",
                    doc_id="doc_0"
                )
            ],
            answer="向量数据库是一种专门用于存储和检索高维向量的数据库系统，"
                   "常用于相似性搜索和机器学习应用。",
            ground_truth="向量数据库是存储高维向量并进行相似性搜索的数据库。",
            relevant_doc_ids=["doc_0"],
            context=["向量数据库是一种专门用于存储和检索高维向量的数据库系统..."]
        )
    ]

    return test_cases


if __name__ == "__main__":
    """
    RAG 质量评估演示程序

    本程序演示如何使用 RAG 评估体系对 RAG 系统进行全面评估
    """
    print("=" * 70)
    print("RAG 质量评估体系演示")
    print("=" * 70)

    # ========== 1. 创建测试用例 ==========
    print("\n【步骤 1】准备测试用例...")
    test_cases = create_sample_test_cases()
    print(f"已创建 {len(test_cases)} 个测试用例")

    # ========== 2. 创建评估器 ==========
    print("\n【步骤 2】初始化评估器...")
    evaluator = RAGASEvaluator()
    bad_case_analyzer = BadCaseAnalyzer()
    print("评估器初始化完成")

    # ========== 3. 单用例评估演示 ==========
    print("\n【步骤 3】单用例评估演示...")
    sample_case = test_cases[0]
    result = evaluator.evaluate(sample_case)
    print(f"查询: {sample_case.query}")
    print(f"检索 F1: {result.retrieval_metrics.f1:.4f}")
    print(f"生成 Faithfulness: {result.generation_metrics.faithfulness:.4f}")
    print(f"整体评分: {result.overall_score:.4f}")

    # ========== 4. 批量评估 ==========
    print("\n【步骤 4】批量评估...")
    batch_results = evaluator.evaluate_batch(test_cases)
    evaluator.print_report(batch_results)

    # ========== 5. Bad Case 分析 ==========
    print("\n【步骤 5】Bad Case 分析...")
    # 准备 Bad Case 数据
    bad_cases = []
    for case in test_cases:
        # 检测问题
        has_issue = False
        error_type = None
        severity = "low"

        if len(case.answer) < 30:
            has_issue = True
            error_type = "incomplete"
            severity = "medium"

        # 幻觉检测（简化版：回答与上下文有明显冲突）
        if "Google" in case.answer and "Facebook" not in case.answer:
            if "Facebook" in " ".join([doc.content for doc in case.retrieved_docs]):
                has_issue = True
                error_type = "hallucination"
                severity = "high"

        if has_issue:
            bad_cases.append({
                "query": case.query,
                "answer": case.answer,
                "expected": case.ground_truth,
                "error_type": error_type,
                "severity": severity
            })

    if bad_cases:
        analysis = bad_case_analyzer.analyze(bad_cases)
        bad_case_analyzer.print_analysis(analysis)

        # 生成优化建议
        suggestions = bad_case_analyzer.generate_optimization_suggestions(analysis)
        print("\n优化建议:")
        for category, suggestion_list in suggestions.items():
            if suggestion_list:
                print(f"\n  [{category}]")
                for s in suggestion_list:
                    print(f"    {s}")

    # ========== 6. 完整流程演示 ==========
    print("\n【步骤 6】运行完整评估流程...")
    pipeline = RAGEvaluationPipeline()
    full_report = pipeline.run_full_evaluation(test_cases)
    pipeline.print_final_report(full_report)

    print("\n评估完成！")
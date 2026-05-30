# 第八课：RAG 评估体系

## 本节概述

本节课将学习如何系统性评估 RAG 系统的效果，识别问题并持续优化。你将掌握：

- RAG 评估的三大维度
- 主流评估工具（RAGAS、Trulens）
- 评估指标详解
- Bad Case 分析方法
- 系统性优化流程

**学习时长**：约 2-3 小时

---

## 8.1 为什么需要评估 RAG？

### 8.1.1 没有评估的代价

```
没有评估的 RAG 系统 = 盲人骑瞎马

常见问题：
├── 检索质量差 → 回答不相关
├── 生成幻觉 → 回答与文档矛盾
├── 上下文丢失 → 回答不完整
└── 性能问题 → 响应慢、超时
```

### 8.1.2 评估的三个维度

```
RAG 评估三角

           检索质量
            /    \
           /      \
          /        \
         /    生成   \
        /    质量    \
       ───────────────
         整体效果
```

| 维度 | 关注点 | 评估指标 |
|------|--------|----------|
| **检索质量** | 检索到的文档是否相关 | 召回率、准确率、MRR |
| **生成质量** | 回答是否准确、完整 | 相关性、正确性、幻觉率 |
| **整体效果** | 系统是否满足用户需求 | RAGAS、用户体验 |

---

## 8.2 检索质量评估

### 8.2.1 基础指标

```python
def evaluate_retrieval(
    query: str,
    retrieved_docs: List[Document],
    relevant_doc_ids: List[str]
) -> Dict:
    """
    评估检索质量
    
    Args:
        query: 查询
        retrieved_docs: 检索返回的文档
        relevant_doc_ids: 真正相关的文档 ID（ground truth）
    """
    retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]
    
    # 计算指标
    true_positives = len(set(retrieved_ids) & set(relevant_doc_ids))
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
    recall = true_positives / len(relevant_doc_ids) if relevant_doc_ids else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / (i + 1)
            break
    
    # NDCG (Normalized Discounted Cumulative Gain)
    # ... (实现较复杂，需要相关性分数)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "retrieved_count": len(retrieved_ids),
        "relevant_count": len(relevant_doc_ids)
    }
```

### 8.2.2 评估指标解读

| 指标 | 含义 | 理想值 | 说明 |
|------|------|--------|------|
| **Precision** | 准确率 | 高 | 检索结果中相关的比例 |
| **Recall** | 召回率 | 高 | 相关文档被检索到的比例 |
| **F1** | 精确率和召回率的调和平均 | 高 | 综合评估 |
| **MRR** | 平均倒数排名 | 高 | 第一个相关结果的位置 |
| **NDCG** | 归一化折损累计增益 | 高 | 排序质量 |

---

## 8.3 生成质量评估

### 8.3.1 使用 LLM 评估

```python
def evaluate_generation(
    query: str,
    retrieved_docs: List[Document],
    generated_answer: str,
    llm
) -> Dict:
    """
    使用 LLM 评估生成质量
    """
    # 构造评估 Prompt
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    evaluation_prompt = f"""请评估以下问答系统的回答质量。

问题：{query}

参考文档：
{context}

回答：{generated_answer}

请从以下维度打分（1-5分，5分最高）：

1. 相关性（回答是否针对问题）：
2. 正确性（回答是否与文档一致）：
3. 完整性（回答是否完整）：
4. 幻觉程度（是否编造信息）：
5. 整体质量：

同时请指出：
- 回答中的优点
- 回答中的问题（如有）
- 改进建议
"""
    
    evaluation = llm.predict(evaluation_prompt)
    
    return {
        "evaluation": evaluation,
        "评估人": "LLM"
    }
```

### 8.3.2 幻觉检测

```python
def detect_hallucination(
    answer: str,
    retrieved_docs: List[Document],
    llm
) -> Dict:
    """
    检测回答中的幻觉
    
    幻觉类型：
    1. 实体错误：提到的实体（人名、地名等）与文档不符
    2. 事实错误：陈述的事实与文档矛盾
    3. 过度推断：基于文档推断，但推断过度
    """
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    hallucination_prompt = f"""请检测以下回答是否包含幻觉。

参考文档：
{context}

回答：
{answer}

请分析：
1. 回答中是否有任何与文档不符的信息？
2. 是否有编造的事实或数字？
3. 是否有过度推断的内容？

如果发现幻觉，请具体指出。
"""
    
    result = llm.predict(hallucination_prompt)
    
    has_hallucination = "没有" not in result and "无" not in result
    
    return {
        "has_hallucination": has_hallucination,
        "analysis": result
    }
```

---

## 8.4 RAGAS 评估框架

### 8.4.1 RAGAS 简介

RAGAS（Retrieval-Augmented Generation Assessment）是一个开源的 RAG 评估框架。

```python
# 安装
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

def evaluate_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str]
) -> Dict:
    """
    使用 RAGAS 评估 RAG 系统
    """
    # 构造数据集
    from datasets import Dataset
    
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    # 执行评估
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,        # 忠实度：回答是否忠实于检索内容
            answer_relevancy,    # 回答相关性：回答是否针对问题
            context_precision,   # 上下文精确度
            context_recall       # 上下文召回率
        ]
    )
    
    return result
```

### 8.4.2 RAGAS 指标解读

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **Faithfulness** | 回答对检索内容的忠实程度 | 1.0 |
| **Answer Relevancy** | 回答与问题的相关程度 | 1.0 |
| **Context Precision** | 检索上下文的精确程度 | 1.0 |
| **Context Recall** | 检索上下文包含正确答案的程度 | 1.0 |

---

## 8.5 Trulens 评估

### 8.5.1 Trulens 简介

Trulens 是另一个流行的 RAG 评估和可观测性框架。

```python
# 安装
# pip install trulens trulens-apps-langchain

from trulens import Feedback
from trulens.apps.langchain import LangChain
from trulens.providers import OpenAI

def evaluate_with_trulens():
    """
    使用 Trulens 评估
    """
    # 初始化
    provider = OpenAI()
    
    # 定义反馈函数
    f_answer_correctness = Feedback(
        provider.mean_literalness,
        on=LangChain.agent.composite_postamble
    ).on_input_output()
    
    f_answer_relevance = Feedback(
        provider.relevance,
        on=LangChain.agent.composite_postamble
    ).on_input_output()
    
    # 创建带评估的链
    tru = Tru()
    
    # 运行评估
    # ...
```

---

## 8.6 Bad Case 分析

### 8.6.1 什么是 Bad Case？

```
Bad Case = RAG 系统回答错误的案例

分类：
├── 检索问题
│   ├── 召回为空（搜不到）
│   ├── 召回不相关
│   └── 召回了错误内容
│
├── 生成问题
│   ├── 幻觉（编造信息）
│   ├── 回答不完整
│   ├── 回答矛盾
│   └── 格式错误
│
└── 整体问题
    ├── 响应超时
    ├── 上下文溢出
    └── 重复回答
```

### 8.6.2 Bad Case 分析流程

```python
def analyze_bad_cases(
    test_results: List[Dict]
) -> Dict:
    """
    Bad Case 分析
    
    test_results 格式：
    [{
        "query": str,
        "retrieved_docs": List[Document],
        "answer": str,
        "expected": str,
        "error_type": "retrieval" | "generation" | "other"
    }]
    """
    categories = {
        "retrieval": {
            "empty": [],      # 召回为空
            "irrelevant": [],  # 召回不相关
            "wrong": []        # 召回错误
        },
        "generation": {
            "hallucination": [],
            "incomplete": [],
            "contradict": [],
            "format_error": []
        },
        "other": []
    }
    
    for result in test_results:
        error_type = result.get("error_type")
        
        if error_type == "retrieval_empty":
            categories["retrieval"]["empty"].append(result)
        elif error_type == "retrieval_irrelevant":
            categories["retrieval"]["irrelevant"].append(result)
        elif error_type == "hallucination":
            categories["generation"]["hallucination"].append(result)
        # ... 更多分类
    
    # 统计各类问题占比
    total = len(test_results)
    summary = {
        category: {
            sub_category: {
                "count": len(cases),
                "percentage": len(cases) / total * 100 if total > 0 else 0
            }
            for sub_category, cases in sub_categories.items()
        }
        for category, sub_categories in categories.items()
    }
    
    return {
        "summary": summary,
        "total_cases": total,
        "bad_cases_by_category": categories
    }
```

---

## 8.7 优化迭代流程

### 8.7.1 PDCA 循环

```
        ┌─────────────────┐
        │     Plan        │
        │   制定评估计划   │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │      Do         │
        │   执行测试       │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │    Check        │
        │   分析结果       │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │      Act        │
        │   优化改进       │
        └────────┬────────┘
                 ↓
                 │
        ← ← ← ← ← ← ←
        
        (持续迭代)
```

### 8.7.2 优化策略对照表

| 问题 | 可能原因 | 优化策略 |
|------|----------|----------|
| 召回为空 | Embedding 质量差 | 换用更好的 Embedding |
| 召回不相关 | 切分策略不当 | 调整 chunk_size |
| 幻觉严重 | 检索内容不准确 | 优化检索 + 增加引用 |
| 回答不完整 | 上下文不足 | 调整召回数量 |
| 响应慢 | 向量库查询慢 | 添加索引优化 |

---

## 本节总结

### 核心要点

1. **评估三维度**：检索质量、生成质量、整体效果
2. **RAGAS**：Faithfulness、Answer Relevancy、Context Precision/Recall
3. **Bad Case 分析**：分类统计、定位问题
4. **PDCA 循环**：持续评估和改进

### 代码文件

- `code/evaluator.py`: RAG 评估工具

### 思考题

1. 为什么 RAGAS 的 Faithfulness 指标很重要？
2. Bad Case 分析的目的是什么？如何指导优化方向？
3. 评估数据集如何构建？有什么最佳实践？

### 下节预告

下一节课我们将学习企业级 RAG 实战，掌握从原型到生产的架构设计、工程化最佳实践。

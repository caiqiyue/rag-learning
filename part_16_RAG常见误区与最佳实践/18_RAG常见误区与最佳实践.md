# 第18节 RAG常见误区与最佳实践

## 学习目标

1. **识别RAG开发中的常见误区**
2. **掌握企业级RAG最佳实践**
3. **能够避免RAG系统设计中的常见陷阱**

---

## 关联知识框架

| 知识点 | 说明 |
|--------|------|
| kp_001 | RAG核心概念与基本原理 |
| kp_004 | 文档文本分割策略 |
| kp_007 | Top-K相似度检索与后处理 |
| kp_012 | 检索器Retriever类型与选择 |
| kp_014 | 混合检索策略 |

---

## 常见误区

### 1. 分割策略误区

| 误区 | 正确做法 |
|------|----------|
| chunk_size越大越好 | 根据业务场景选择合适大小 |
| 不考虑重叠率 | 适度重叠保持上下文连贯 |
| 所有文档用同一策略 | 根据文档类型选择分割方法 |

### 2. 检索策略误区

| 误区 | 正确做法 |
|------|----------|
| 只用向量检索 | 混合检索效果更好 |
| 忽略Top-K参数 | 根据召回率调整 |
| 不做结果过滤 | 设置相似度阈值 |

### 3. 提示词工程误区

| 误区 | 正确做法 |
|------|----------|
| 提示词过于简单 | 详细说明上下文格式 |
| 不指定输出格式 | 明确要求引用来源 |
| 忽略模型能力差异 | 根据模型调整提示词 |

### 4. 元数据过滤误区

| 误区 | 正确做法 |
|------|----------|
| 不使用元数据 | 充分利用元数据过滤 |
| 元数据设计混乱 | 提前规划元数据字段 |
| 忽略元数据质量 | 确保元数据准确性 |

### 5. 评估迭代误区

| 误区 | 正确做法 |
|------|----------|
| 不做评估 | 建立完整评估流程 |
| 只看生成质量 | 同时评估检索质量 |
| 不做Bad Case分析 | 深入分析失败案例 |

---

## 企业级最佳实践

### 分割策略

```python
# 推荐：根据文档结构选择分割策略
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

# 通用文本：SentenceSplitter
通用文档分割器 = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# 专业文档：SemanticSplitterNodeParser
专业文档分割器 = SemanticSplitterNodeParser(buffer_size=2, breakpoint_percentile_threshold=80)
```

### 混合检索

```python
# 向量检索 + BM25 = 混合检索
from llama_index.core.retrievers import QueryFusionRetriever

混合检索器 = QueryFusionRetriever(
    retrievers=[向量检索器, BM25检索器],
    mode="reciprocal_rerank",  # RRF融合
    similarity_top_k=5
)
```

### 提示词模板

```python
提示词模板 = """
你是一个专业的问答助手。请基于以下上下文回答用户问题。

【上下文】
{context}

【问题】
{question}

【要求】
1. 只基于提供的上下文回答
2. 如果上下文中没有相关信息，回复"信息不足，无法回答"
3. 在回答中标明信息来源
"""
```

---

## 代码案例

### 文件路径

```
E:/ai-learning/09-agent-engineering/rag-learning/part_16_RAG常见误区与最佳实践/16_codes/rag_best_practices.py
```

### 运行说明

```bash
pip install langchain langchain-community openai faiss-cpu

python E:/ai-learning/09-agent-engineering/rag-learning/part_16_RAG常见误区与最佳实践/16_codes/rag_best_practices.py
```

---

## 课后思考

1. 在实际项目中，如何判断当前RAG系统的主要瓶颈是检索还是生成？
2. 为什么混合检索通常比单一向量检索效果更好？
3. 企业级RAG系统中，监控指标应该如何选择和设置？
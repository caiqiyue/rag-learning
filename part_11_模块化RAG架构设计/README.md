# 第13节 模块化RAG架构设计

## 学习目标

1. **理解模块化RAG的核心思想**
   - Dataflow：文档从原始输入到可检索状态的处理流程
   - Router：根据查询特征选择最优检索路径
   - Query Pipeline：组件串联与流程编排

2. **掌握LlamaIndex Pipeline组件的使用方法**
   - QueryPipeline的声明式构建
   - RetrieverComponent、FnComponent等组件的使用
   - 组件间的连接与数据流动

3. **能够设计可扩展的企业级RAG架构**
   - 模块独立优化与灵活替换
   - 多检索器组合策略
   - 端到端流程的可观测性

---

## 关联知识框架

| 知识点 | 说明 |
|--------|------|
| kp_027 | Modular RAG模块化架构 - 将RAG系统拆分为独立模块，支持灵活组合与独立优化 |
| kp_026 | Agentic RAG智能体RAG - 结合Agent架构，让RAG系统具备自主决策能力 |

---

## 核心概念

### 1. 什么是模块化RAG？

模块化RAG（Modular RAG）是RAG架构的最新演进阶段，它将传统的端到端RAG系统拆分为多个独立模块，每个模块负责特定功能，通过灵活组合形成完整的检索增强生成系统。

**核心模块：**

| 模块 | 功能 | 关键组件 |
|------|------|----------|
| Dataflow | 文档处理流程 | Document Loader、Parser、Node Splitter、Embedder |
| Router | 查询路由选择 | 规则路由、模型路由、向量路由 |
| Retriever | 信息检索 | BM25Retriever、VectorRetriever、QueryFusionRetriever |
| Reranker | 结果精排 | ColbertRerank、SentenceTransformerRerank |
| Generator | 答案生成 | LLM、ResponseSynthesizer |

### 2. 为什么需要模块化架构？

传统的Naive RAG存在以下局限：

- **紧耦合**：检索和生成紧密绑定，难以单独优化
- **策略单一**：所有查询使用相同的检索策略
- **难以扩展**：新增组件需要修改核心逻辑

模块化RAG的优势：

- **独立优化**：每个模块可单独调优，不影响其他模块
- **灵活组合**：根据场景选择不同模块组合
- **便于调试**：问题可定位到具体模块
- **支持A/B测试**：不同模块组合可对比效果

### 3. Router的工作原理

Router是模块化RAG的"大脑"，负责判断当前查询应该使用哪种检索策略：

```
用户查询 -> 特征分析 -> 路由决策 -> 选定检索器 -> 执行检索
```

**路由策略类型：**

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| 基于规则 | 关键词匹配、查询长度 | 规则明确的查询 |
| 基于模型 | 零样本分类模型 | 需要语义理解 |
| 基于向量 | 查询与路径描述的向量相似度 | 多路径选择 |

### 4. Query Pipeline的作用

Query Pipeline是模块化RAG的执行引擎，将多个组件串联成完整的处理管道：

- **声明式配置**：通过代码定义流程，易于理解和维护
- **组件重用**：同一组件可在多个Pipeline中复用
- **条件分支**：支持基于查询特征的动态路由
- **可观测性**：每个步骤都有日志，便于调试

---

## 代码案例

### 代码文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| 模块化RAG示例 | `E:/ai-learning/09-agent-engineering/rag-learning/part_11_模块化RAG架构设计/11_codes/modular_rag.py` | 演示Dataflow、Router、QueryPipeline组件 |

### 运行说明

#### 环境准备

```bash
# 安装依赖
pip install llama-index llama-index-retrievers-bm25 llama-index-postprocessor-rerank
```

#### 运行模块化RAG示例

```bash
python E:/ai-learning/09-agent-engineering/rag-learning/part_11_模块化RAG架构设计/11_codes/modular_rag.py
```

该脚本演示：

1. **Dataflow**：文档数据流处理（加载->解析->索引构建）
2. **Router**：查询路由器（根据查询类型选择检索路径）
3. **Query Pipeline**：查询管道构建（组件串联与执行）
4. **ModularRAGSystem**：完整模块化RAG系统集成

---

## 关键代码模式

### Dataflow模式

```python
class DocumentDataflow:
    """文档数据流处理器"""

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    def run(self, docs: List[Document]) -> VectorStoreIndex:
        """执行完整的数据流管道"""
        return (self
            .load_documents(docs)
            .parse_documents()
            .build_index().index)
```

### Router模式

```python
class Router:
    """查询路由器"""

    def route(self, query: str) -> QueryType:
        """根据查询内容判断查询类型"""
        # 基于规则的简单实现
        if has_exact_terms:
            return QueryType.EXACT_MATCH
        return QueryType.GENERAL

    def select_retriever(self, query: str) -> BaseRetriever:
        """根据查询类型选择检索器"""
        query_type = self.route(query)
        # 根据类型返回对应检索器
```

### QueryPipeline模式

```python
from llama_index.core.query_pipeline import QueryPipeline

pipeline = QueryPipeline()
pipeline.add_component("input", InputComponent())
pipeline.add_component("retriever", RetrieverComponent(retriever))
pipeline.add_component("synthesizer", synthesizer)

pipeline.link("input", "retriever")
pipeline.link("retriever", "synthesizer")

response = pipeline.run(input=user_query)
```

---

## 最佳实践

1. **模块边界清晰**：每个模块只负责一个明确的任务，避免职责重叠

2. **接口标准化**：模块间通过标准接口通信，便于替换和测试

3. **配置外部化**：将检索策略、权重等参数通过配置管理，不硬编码

4. **日志完整**：每个模块都应有清晰的日志输出，便于问题定位

5. **渐进式演进**：从简单Pipeline开始，逐步增加复杂度

6. **评估驱动**：使用RAG评估框架验证不同配置的优劣
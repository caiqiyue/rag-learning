# 第11节 混合检索与重排序策略

## 学习目标

1. **理解向量检索与关键词检索的互补原理**
   - 向量检索（密集检索）：基于语义相似度，能理解同义词和深层语义
   - 关键词检索（稀疏检索/BM25）：基于精确匹配，对专业术语效果好
   - 两者优缺互补，结合使用可提升召回质量

2. **掌握BM25、密集检索、混合检索的实现方法**
   - BM25Retriever：基于倒排索引的关键词检索
   - VectorIndexRetriever：基于向量嵌入的语义检索
   - QueryFusionRetriever：使用RRF算法融合多种检索结果

3. **能够使用重排序模型（如BGE-Reranker）优化结果排序**
   - Reranker进行深度的一对一相关性评估
   - Cross-Encoder架构同时输入查询和文档进行注意力交互
   - "召回->精排"管道模式是兼顾召回率与精度的黄金标准

---

## 关联知识框架

| 知识点 | 说明 |
|--------|------|
| kp_013 | 重排序Rerank精排策略 - Cross-Encoder和专用Reranker的精排原理 |
| kp_014 | 混合检索策略 - QueryFusionRetriever的reciprocal_rerank融合模式 |

---

## 核心概念

### 1. 为什么需要混合检索？

单一检索方法存在局限：

| 检索类型 | 优势 | 劣势 |
|----------|------|------|
| 向量检索 | 理解语义相似、同义词 | 对精确关键词匹配效果差 |
| BM25检索 | 精确匹配术语、型号 | 无法理解语义 |

**混合检索**让两种方法同时工作，然后通过结果融合（如RRF算法）综合排名，既捕捉语义关联，又保留精确匹配能力。

### 2. 什么是重排序（Rerank）？

重排序是精排阶段，对召回阶段的结果进行深度评估：

- **召回阶段**：快速从大规模文档中筛选Top-K候选（如K=20）
- **精排阶段**：对候选进行深度评估，输出Top-N结果（如N=5）

### 3. RRF（Reciprocal Rank Fusion）算法

RRF是常用的结果融合算法，核心思想：
- 如果一个文档在多个检索器中都排名靠前，说明它更相关
- 通过公式 `score = sum(1/(rank + k))` 计算融合分数
- k为平滑参数（通常为60）

---

## 代码案例

### 代码文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| 混合检索示例 | `E:/ai-learning/09-agent-engineering/rag-learning/part_09_混合检索与重排序策略/09_codes/hybrid_search.py` | 演示BM25、密集检索、混合检索的实现 |
| 重排序示例 | `E:/ai-learning/09-agent-engineering/rag-learning/part_09_混合检索与重排序策略/09_codes/reranking.py` | 演示BGE-Reranker精排流程 |

### 运行说明

#### 环境准备

```bash
# 安装依赖
pip install llama-index llama-index-retrievers-bm25 rank-bm25

# 下载BGE-Reranker模型（可选，用于重排序示例）
# 方法1：使用ModelScope
modelscope download --model BAAI/bge-reranker-base --local_dir ./bge_reranker

# 方法2：使用HuggingFace
huggingface-cli download BAAI/bge-reranker-base --local-dir ./bge_reranker
```

#### 运行混合检索示例

```bash
python E:/ai-learning/09-agent-engineering/rag-learning/part_09_混合检索与重排序策略/09_codes/hybrid_search.py
```

该脚本演示：
1. BM25检索器的创建和使用
2. 向量检索器的创建和使用
3. QueryFusionRetriever进行混合检索和结果融合
4. 不同融合模式的对比

#### 运行重排序示例

```bash
python E:/ai-learning/09-agent-engineering/rag-learning/part_09_混合检索与重排序策略/09_codes/reranking.py
```

该脚本演示：
1. 基础检索器的创建
2. BGE-Reranker的配置
3. 有Reranker vs 无Reranker的对比
4. 完整的"混合检索 + Reranker精排"流程

---

## 关键代码模式

### 混合检索器创建

```python
from llama_index.core.retrievers import QueryFusionRetriever

# 创建混合检索器
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,          # 最终返回数量
    mode="reciprocal_rerank",     # RRF融合模式
    use_async=True,
    retriever_weights=[0.6, 0.4]  # 可选：检索器权重
)
```

### Reranker创建与使用

```python
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# 创建Reranker
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=3,                      # 精排后返回数量
    score_threshold=0.3           # 分数阈值
)

# 构建QueryEngine
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker]
)
```

---

## 最佳实践

1. **召回数量与精排数量的比例**：推荐10:3或20:5，召回数太少可能漏掉关键信息，太多会增加Reranker负担

2. **模型选择**：
   - Cross-Encoder Reranker（如BGE-Reranker）提供最佳性价比
   - 本地部署适合对延迟要求高的场景
   - 云服务（ Cohere、Jina）适合快速验证

3. **与混合检索结合**：最佳模式是"HybridRetriever + Reranker"，先多路召回再精排

4. **分数阈值调整**：根据实际效果调整，过低引入噪声，过高可能过滤有用信息
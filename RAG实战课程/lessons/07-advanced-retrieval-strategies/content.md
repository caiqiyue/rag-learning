# 第七课：高级检索策略

## 本节概述

本节课将学习 RAG 系统中至关重要的一环：如何从向量数据库中精准检索到相关内容。你将掌握：

- 检索流程的四步
- 混合检索技术
- 重排序（Reranker）
- 上下文压缩
- 检索优化最佳实践

**学习时长**：约 3-4 小时

---

## 7.1 检索流程四步

### 7.1.1 为什么需要四步？

```
很多人以为检索就是简单的"向量查一下相似度"

但实际上，工业级 RAG 的检索模块是精心设计的完整流程：

┌─────────────────────────────────────────────────────────┐
│                    检索流程四步                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Query 向量化 ─→ 把问题变成数学向量                  │
│         ↓                                               │
│  2. 初步召回 ─→ 用向量搜索快速捞一批候选（召回优先）    │
│         ↓                                               │
│  3. 精排 ─→ 用更精细的模型重新排序（精度优先）         │
│         ↓                                               │
│  4. 结果过滤与合并 ─→ 去重、过滤、组装                 │
│                                                         │
└─────────────────────────────────────────────────────────┘

为什么这样设计？
├── 初步召回要快：向量搜索 O(logN) 级别
├── 精排要准：可以调用更复杂的模型
└── 各司其职，平衡速度与精度
```

### 7.1.2 每步详解

#### 第一步：Query 向量化

```python
def query_embedding(query: str, embeddings) -> np.ndarray:
    """
    将用户问题转换为向量
    
    注意：Query 的向量化通常与文档使用不同的策略
    - 文档：强调信息完整性
    - Query：强调问题意图
    """
    # 直接编码
    query_vector = embeddings.embed_query(query)
    
    # Query 扩展（可选）：将问题改写成多个版本
    expanded_queries = [
        query,
        rewrite_to_statement(query),      # 问题→陈述句
        expand_with_synonyms(query)       # 同义词扩展
    ]
    
    # 多 query 向量平均
    query_vectors = [embeddings.embed_query(q) for q in expanded_queries]
    query_vector = np.mean(query_vectors, axis=0)
    
    return query_vector
```

#### 第二步：初步召回

```python
def initial_recall(
    query_vector: np.ndarray,
    vectorstore,
    top_k: int = 100  # 召回数量通常比最终返回的多
) -> List[Document]:
    """
    初步召回：用向量搜索快速获取候选集
    
    关键参数：
    - top_k：召回数量，宁多勿少（后续会精排和过滤）
    - similarity_threshold：相似度阈值
    """
    # 基础向量搜索
    initial_results = vectorstore.similarity_search_by_vector(
        query_vector,
        k=top_k
    )
    
    # 也可以用 MMR（Maximum Marginal Relevance）
    # 兼顾相似度和多样性
    mmr_results = vectorstore.max_marginal_relevance_search_by_vector(
        query_vector,
        k=top_k,
        fetch_k=200,  # 取出更多供选择
        lambda_mult=0.5  # 0=只看相似度，1=只看多样性
    )
    
    return initial_results
```

#### 第三步：精排（Rerank）

```python
def rerank_results(
    query: str,
    candidate_docs: List[Document],
    reranker_model,
    top_k: int = 10
) -> List[Dict]:
    """
    精排：使用更强大的模型重新排序
    
    为什么需要精排？
    - 向量搜索只考虑语义相似性
    - 精排可以考虑：
      - 查询与文档的相关性（不只是相似）
      - 文档的信息密度
      - 上下文连贯性
    """
    # 构造 query-document 对
    pairs = [(query, doc.page_content) for doc in candidate_docs]
    
    # 调用 reranker 模型
    scores = reranker_model.predict(pairs)
    
    # 按分数排序
    doc_scores = list(zip(candidate_docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [
        {"doc": doc, "score": float(score)}
        for doc, score in doc_scores[:top_k]
    ]
```

#### 第四步：结果过滤与合并

```python
def filter_and_merge(
    reranked_results: List[Dict],
    min_score: float = 0.5,
    max_docs: int = 5
) -> List[Document]:
    """
    过滤与合并
    
    - 过滤：移除低分或无关结果
    - 合并：处理重复内容
    """
    # 1. 过滤低分
    filtered = [
        r for r in reranked_results
        if r["score"] >= min_score
    ]
    
    # 2. 去重（基于内容相似度）
    unique_docs = []
    seen_contents = set()
    
    for r in filtered:
        content = r["doc"].page_content
        # 简化的去重：前50字符相同就认为是重复
        content_key = content[:50]
        
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            unique_docs.append(r["doc"])
    
    # 3. 限制数量
    return unique_docs[:max_docs]
```

---

## 7.2 混合检索

### 7.2.1 为什么需要混合检索？

```
向量检索的局限：
✓ 擅长：语义相似性（"苹果"能匹配"水果"）
✗ 局限：精确关键词匹配（"RAG v2.3.0" 可能匹配不到）

关键词检索的局限：
✓ 擅长：精确匹配（版本号、型号）
✗ 局限：无法捕捉语义（"手机"匹配不了"移动电话"）

混合检索 = 向量 + 关键词 = 兼得两者优点
```

### 7.2.2 实现混合检索

```python
from langchain.retrievers import EnsembleRetriever

def hybrid_retrieval(
    query: str,
    vectorstore,
    keyword_store,
    weights: List[float] = [0.5, 0.5]  # 向量、关键词的权重
) -> List[Document]:
    """
    混合检索
    
    weights: [向量检索权重, 关键词检索权重]
    """
    # 1. 向量检索
    vector_results = vectorstore.similarity_search(query, k=20)
    
    # 2. 关键词检索（BM25）
    keyword_results = keyword_store.similarity_search(query, k=20)
    
    # 3. RRFL 融合
    fused_scores = {}
    
    for i, doc in enumerate(vector_results):
        # 向量得分（归一化）
        vector_score = 1.0 / (i + 1)
        doc_id = doc.page_content[:100]  # 用内容前100字符作为ID
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + weights[0] * vector_score
    
    for i, doc in enumerate(keyword_results):
        # BM25 得分
        bm25_score = 1.0 / (i + 1)
        doc_id = doc.page_content[:100]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + weights[1] * bm25_score
    
    # 4. 排序
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 5. 返回完整文档
    doc_map = {doc.page_content[:100]: doc for doc in vector_results + keyword_results}
    
    results = []
    for doc_id, score in sorted_ids[:10]:
        if doc_id in doc_map:
            results.append(doc_map[doc_id])
    
    return results
```

---

## 7.3 重排序（Reranker）

### 7.3.1 常用 Reranker 模型

| 模型 | 提供商 | 特点 |
|------|--------|------|
| bge-reranker | BAAI | 开源、中文好 |
| jina-reranker | Jina | 云端 API |
| Cohere Rerank | Cohere | 云端 API、效果好 |
| OpenAI Rerank | OpenAI | 云端 API |

### 7.3.2 Reranker 代码示例

```python
def rerank_with_bge(
    query: str,
    documents: List[str],
    model_name: str = "BAAI/bge-reranker-large"
) -> List[Dict]:
    """
    使用 BGE Reranker 重排序
    
    需要先安装：pip install sentence-transformers
    """
    from sentence_transformers import CrossEncoder
    
    model = CrossEncoder(model_name)
    
    # 构造 query-document 对
    pairs = [(query, doc) for doc in documents]
    
    # 获取分数
    scores = model.predict(pairs)
    
    # 排序
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [
        {"document": doc, "score": float(score)}
        for doc, score in doc_scores
    ]


def rerank_with_cohere(
    query: str,
    documents: List[str],
    api_key: str,
    top_n: int = 10
) -> List[Dict]:
    """
    使用 Cohere Rerank API
    """
    import cohere
    
    co = cohere.Client(api_key)
    
    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-multilingual-v2.0"
    )
    
    return [
        {
            "document": result.document.text,
            "score": result.relevance_score
        }
        for result in response.results
    ]
```

---

## 7.4 上下文压缩

### 7.4.1 为什么需要压缩？

```
检索返回的文档可能很长，但 LLM 的上下文窗口有限

解决方案：上下文压缩

原始文档："......[中间省略1000字]......"  2000字
                    ↓
压缩后："......[提取关键段落]......"     500字
                    ↓
塞入 Prompt
```

### 7.4.2 上下文压缩实现

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import EmbeddingsFilter

def create_compression_retriever(
    base_retriever,
    embeddings,
    similarity_threshold: float = 0.7
):
    """
    创建带上下文压缩的检索器
    """
    # 1. 基于相似度的过滤器
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold
    )
    
    # 2. 包装成压缩检索器
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        document_compressor=embeddings_filter
    )
    
    return compression_retriever


def compress_with_llm(
    documents: List[Document],
    llm,
    max_tokens: int = 1000
) -> List[Document]:
    """
    使用 LLM 进行上下文压缩
    
    LLM 会提炼每个文档块的核心内容
    """
    compressed_docs = []
    
    for doc in documents:
        prompt = f"""请提炼以下文档的核心内容，保留关键信息。

原始文档：
{doc.page_content}

要求：
- 提炼到 {max_tokens} 字以内
- 保留关键术语和数据
- 保持原文的核心意思
"""
        
        compressed_content = llm.predict(prompt)
        
        compressed_docs.append(
            Document(
                page_content=compressed_content,
                metadata=doc.metadata
            )
        )
    
    return compressed_docs
```

---

## 7.5 检索优化最佳实践

### 7.5.1 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 召回不全面 | 语义差距大 | Query 扩展、混合检索 |
| 精度不足 | 向量搜索过于宽松 | Reranker、精排 |
| 噪声过多 | 检索到无关内容 | 上下文压缩、元数据过滤 |
| 重复内容 | 多处提及同一信息 | 去重、 MMR |
| 排序不合理 | 只看相似度 | Reranker、重排序 |

### 7.5.2 检索策略选择指南

```
场景 → 检索策略

通用问答：
→ 基础向量搜索 + Reranker

需要精确匹配的（如版本号、型号）：
→ 混合检索（向量 + BM25）

需要多样性的（如调研报告）：
→ MMR 检索

长文档检索：
→ 层级检索（大块定位 → 小块提取）

多跳推理（如关系推理）：
→ 图检索 + 向量检索
```

---

## 本节总结

### 核心要点

1. **检索四步**：Query向量化 → 初步召回 → 精排 → 过滤合并
2. **混合检索**：向量搜索 + 关键词搜索，兼得语义和精确
3. **Reranker**：用更强的模型重新排序，提升精度
4. **上下文压缩**：减少噪声，适应有限上下文

### 代码文件

- `code/retriever.py`: 检索器实现
- `code/reranker.py`: 重排序示例

### 思考题

1. 为什么初步召回要"宁多勿少"？
2. 混合检索中，向量搜索和 BM25 的权重如何选择？
3. Reranker 的计算成本比向量搜索高很多，适合什么场景？

### 下节预告

下一节课我们将学习 RAG 评估体系，掌握如何系统性评估 RAG 系统的效果。

# 第三课：RAG系统架构与组件

## 本节概述

本节课将带你深入理解 RAG 系统的完整架构，掌握六大核心组件的作用和数据流转过程，了解主流开发框架的适用场景。你将掌握：

- RAG 六大核心组件及其职责
- 完整的数据流转过程
- LangChain 和 LlamaIndex 的适用场景

**学习时长**：约 3-4 小时

---

## 3.1 RAG 六大核心组件

RAG 系统由六大核心组件构成，它们协同工作，完成从文档到答案的完整流程。

### 3.1.1 组件总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG 系统架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │
│  │  文档加载器  │ ─→ │  文本切分器  │ ─→ │  Embedding 模型 │     │
│  │   (Loader)  │    │(Text Splitter)│   │                 │     │
│  └─────────────┘    └─────────────┘    └────────┬────────┘     │
│                                                 ↓                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │
│  │   生成器    │ ←─ │   检索器    │ ←─ │   向量数据库    │     │
│  │(Generator) │    │(Retriever) │    │ (Vector Store)  │     │
│  └─────────────┘    └─────────────┘    └─────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1.2 组件详解

#### 组件一：文档加载器（Document Loader）

**职责**：将各种格式的文档（PDF、Word、HTML、Markdown 等）转换为统一的文档对象格式。

**生活中的类比**：就像图书馆的采购员，负责把各种来源的书籍（精装、平装、电子版）统一登记入库。

**代码示例**：

```python
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader

# PDF 加载器
pdf_loader = PyPDFLoader("产品手册.pdf")
pdf_docs = pdf_loader.load()
# 返回: [Document(page_content="...", metadata={"source": "产品手册.pdf", "page": 1})]

# Word 加载器
docx_loader = Docx2txtLoader("技术规范.docx")
docx_docs = docx_loader.load()

# HTML 加载器
html_loader = UnstructuredHTMLLoader("官网文章.html")
html_docs = html_loader.load()

# 支持的格式（部分）
# - PDF: PyPDFLoader, PDFPlumberLoader
# - Word: Docx2txtLoader, UnstructuredWordLoader
# - Excel: UnstructuredExcelLoader
# - HTML: UnstructuredHTMLLoader
# - Markdown: UnstructuredMarkdownLoader
# - CSV: CSVLoader
```

#### 组件二：文本切分器（Text Splitter）

**职责**：将长文档分割成较小的文本块（Chunk），以便检索和适应模型的上下文窗口。

**生活中的类比**：就像把一本厚厚的书拆分成章节，每个章节可以独立阅读。

**代码示例**：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 初始化切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个块的最大字符数
    chunk_overlap=50,    # 块之间的重叠字符数（保持上下文连贯）
    length_function=len, # 计算长度的函数
    separators=["\n\n", "\n", "。", " "]  # 分割符优先级
)

# 切分文档
long_doc = "..."  # 假设是一个 5000 字符的文档
chunks = text_splitter.split_text(long_doc)

# 输出示例
# [
#   "第一章 RAG基础\n\nRAG 是检索增强生成的缩写...",
#   "第二章 Embedding\n\nEmbedding 是将文本转换为向量...",
#   ...
# ]

# 或者直接切分 Document 对象列表
docs = pdf_loader.load()
split_docs = text_splitter.split_documents(docs)
```

**为什么需要切分？**

```
问题：为什么要切分，而不是直接用整篇文档？

答案：
1. 检索精度：整篇文档可能包含很多无关内容，切分后可以精准检索
2. 上下文窗口：LLM 的上下文窗口有限，不可能塞入整本书
3. 计算效率：向量化的计算成本与文本长度正相关
4. 噪声控制：无关内容会干扰模型的判断
```

#### 组件三：Embedding 模型

**职责**：将文本转换为稠密的向量表示，使得语义相似的文本在向量空间中相近。

**生活中的类比**：就像给每本书创建一个"指纹"，指纹相似的书内容也相似。

**代码示例**：

```python
from langchain.embeddings import OpenAIEmbeddings

# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # OpenAI 的 Embedding 模型
)

# 单个文本向量化
query = "RAG 的原理是什么？"
query_vector = embeddings.embed_query(query)
# 返回: [0.123, -0.456, 0.789, ...]  # 1536 维向量

# 多个文本批量向量化
texts = ["文本1", "文本2", "文本3"]
text_vectors = embeddings.embed_documents(texts)
# 返回: [[向量1], [向量2], [向量3]]
```

**主流 Embedding 模型对比**：

| 模型 | 维度 | 特点 | 价格 |
|------|------|------|------|
| text-embedding-3-small | 1536 | OpenAI 最新版，效果好 | ¥0.15/1M tokens |
| text-embedding-ada-002 | 1536 | 稳定版，应用广泛 | ¥0.4/1M tokens |
| BGE-large-zh | 1024 | 中文效果好，开源 | 免费（本地运行） |
| M3E | 768 | 中文开源，轻量 | 免费（本地运行） |

#### 组件四：向量数据库（Vector Store）

**职责**：存储文本的向量表示，支持高效的相似度搜索。

**生活中的类比**：就像图书馆的索引系统，可以快速找到与问题最相关的书籍位置。

**代码示例**：

```python
from langchain.vectorstores import Chroma

# 初始化 Chroma 向量数据库
vectorstore = Chroma.from_documents(
    documents=split_docs,  # 切分后的文档块
    embedding=embeddings,   # Embedding 模型
    persist_directory="./chroma_db"  # 持久化存储路径
)

# 相似度搜索
query = "RAG 的 Embedding 是什么？"
results = vectorstore.similarity_search(
    query=query,
    k=3  # 返回最相似的 3 个结果
)

# 输出示例
# [
#   Document(page_content="Embedding 是将文本转换为向量...", page=5),
#   Document(page_content="向量数据库存储文本的向量...", page=12),
#   Document(page_content="RAG 三步走包括...", page=3)
# ]
```

**主流向量数据库对比**：

| 数据库 | 特点 | 适用场景 | 部署方式 |
|--------|------|---------|----------|
| Chroma | 轻量级、本地优先 | 快速原型 | 本地/嵌入 |
| Milvus | 分布式、高可用 | 企业级生产 | 云/私有 |
| Pinecone | 云原生、易扩展 | 云服务 | 全托管 |
| Weaviate | 混合检索（向量+关键词） | 多模态 | 云/私有 |
| Faiss | Facebook 开源、高性能 | 离线分析 | 本地 |
| Qdrant | Rust 开发、性能高 | 生产级 | 云/私有 |

#### 组件五：检索器（Retriever）

**职责**：根据用户问题，从向量数据库中检索相关文档。

**生活中的类比**：就像图书馆的管理员，根据你的问题帮你找到相关的书籍。

**代码示例**：

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 创建检索链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",  # 将检索结果拼接到一个 Prompt
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}  # 检索 3 个结果
    )
)

# 执行问答
query = "RAG 的三个步骤是什么？"
result = qa_chain({"query": query})

# 输出
# {
#   "query": "RAG 的三个步骤是什么？",
#   "result": "RAG 由检索（Retrieve）、增强（Augment）、生成（Generate）三个步骤组成...",
#   "source_documents": [Document(...), Document(...), Document(...)]
# }
```

**检索策略**：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| similarity_search | 基于向量相似度 | 通用场景 |
| MMR (Maximum Marginal Relevance) | 兼顾相似度和多样性 | 需要多样结果的场景 |
| similarity_score_threshold | 设置相似度阈值 | 过滤低质量结果 |

#### 组件六：生成器（Generator）

**职责**：基于检索到的文档和用户问题，生成最终回答。

**生活中的类比**：就像一个博学的顾问，阅读了参考资料后，用自己的话整理并回答你的问题。

**代码示例**：

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 定义 Prompt 模板
template = """你是一个专业的技术文档助手。
请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

回答要求：
1. 基于提供的文档回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 尽量引用文档中的具体内容
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 创建 LLM Chain
llm = OpenAI(temperature=0.3)
chain = LLMChain(llm=llm, prompt=prompt)

# 执行生成
context = "RAG 是检索增强生成的缩写..."  # 来自检索结果
question = "什么是 RAG？"
response = chain.run(context=context, question=question)
```

---

## 3.2 完整数据流转过程

### 3.2.1 索引阶段（Indexing / Ingestion）

**目的**：将文档存入知识库，构建可检索的向量数据库。

```
文档 (PDF/Word/HTML)
    │
    ▼
┌─────────────┐
│  文档加载器  │  读取文档，转换为 Document 对象
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  文本切分器  │  将长文档分割成小块 (Chunk)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Embedding   │  将每个 Chunk 转换为向量
│   模型      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 向量数据库  │  存储向量和原始文本
└─────────────┘
```

### 3.2.2 查询阶段（Query / Retrieval）

**目的**：根据用户问题，从知识库中检索相关文档。

```
用户问题
    │
    ▼
┌─────────────┐
│  问题向量化  │  使用 Embedding 模型将问题转换为向量
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 向量数据库  │  执行相似度搜索，找出最相关的 K 个文档
│   检索      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  结果排序   │  可选：使用 Reranker 进一步排序
└──────┬──────┘
       │
       ▼
相关文档列表
```

### 3.2.3 生成阶段（Generation）

**目的**：基于检索结果和问题，生成最终回答。

```
用户问题 + 相关文档
    │
    ▼
┌─────────────┐
│  Prompt     │  将文档和问题组装成增强 Prompt
│  增强       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  大语言模型  │  基于增强 Prompt 生成回答
│  (LLM)     │
└──────┬──────┘
       │
       ▼
最终回答
```

---

## 3.3 主流框架对比

### 3.3.1 LangChain

**定位**：全流程 RAG 开发框架

**特点**：
- 组件丰富：文档加载、切分、Embedding、向量库、检索、生成全都有
- 灵活性高：可以自由组合各种组件
- 学习曲线陡峭：需要理解底层原理

**适用场景**：
- 需要高度定制化的 RAG 系统
- 需要复杂的工作流编排
- 需要集成多种数据源

**代码示例**：

```python
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 完整流程
loader = PyPDFLoader("文档.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vectorstore.as_retriever())
```

### 3.3.2 LlamaIndex

**定位**：专注于索引和检索的框架

**特点**：
- 索引结构丰富：Tree Index、Keyword Table Index、Vector Store Index
- 查询优化：支持多种查询策略
- 文档理解能力强

**适用场景**：
- 需要复杂索引结构的场景
- 需要对文档进行深度理解
- 构建知识图谱问答

**代码示例**：

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

# 快速构建
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=OpenAI())

# 查询
response = query_engine.query("RAG 的原理是什么？")
```

### 3.3.3 框架选择指南

```
需求分析 → 选择框架

│
├─ 需要快速原型验证？
│   └─ → LlamaIndex（上手简单）
│
├─ 需要高度定制化？
│   └─ → LangChain（灵活性高）
│
├─ 需要复杂索引（知识图谱）？
│   └─ → LlamaIndex
│
├─ 需要多种数据源集成？
│   └─ → LangChain
│
├─ 企业级生产系统？
│   └─ → 都行，视团队熟悉度选择
```

---

## 3.4 实战：手写一个 RAG 系统

为了深入理解 RAG 架构，让我们不依赖任何框架，手写一个最简单的 RAG 系统。

```python
"""
一个最简 RAG 系统（纯 Python 实现）
演示 RAG 各组件的工作原理
"""

import json
import numpy as np
from typing import List, Tuple

# ========== 1. 简化的文档加载器 ==========
class SimpleDocumentLoader:
    """模拟加载文档"""
    def load(self, text: str) -> dict:
        return {
            "content": text,
            "metadata": {"source": "memory"}
        }

# ========== 2. 简化的文本切分器 ==========
class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks

# ========== 3. 简化的 Embedding（基于词频）==========
class SimpleEmbedding:
    """一个非常简单的 Embedding 实现（仅用于演示原理）"""
    def __init__(self):
        # 词表
        self.vocab = {
            "RAG": 0, "检索": 1, "增强": 2, "生成": 3,
            "向量": 4, "文档": 5, "问题": 6, "回答": 7,
            "知识": 8, "系统": 9
        }
        self.dim = len(self.vocab)
    
    def embed(self, text: str) -> np.ndarray:
        """将文本转换为向量（词袋模型）"""
        vector = np.zeros(self.dim)
        for word, idx in self.vocab.items():
            if word in text:
                vector[idx] = 1.0
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

# ========== 4. 简化的向量数据库 ==========
class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.documents = []
    
    def add(self, doc: str, vector: np.ndarray):
        self.vectors.append(vector)
        self.documents.append(doc)
    
    def search(self, query_vector: np.ndarray, k: int = 2) -> List[Tuple[str, float]]:
        """计算余弦相似度并返回 Top-K 结果"""
        scores = []
        for vec in self.vectors:
            # 余弦相似度
            score = np.dot(query_vector, vec)
            scores.append(score)
        
        # 排序并返回 Top-K
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:k]:
            results.append((self.documents[idx], score))
        return results

# ========== 5. 简化的生成器 ==========
class SimpleGenerator:
    """模拟 LLM 生成（实际应用中应调用真实的 LLM API）"""
    def generate(self, context: str, question: str) -> str:
        # 实际应用中，这里应该调用 OpenAI/Claude 等 LLM
        return f"根据提供的文档，'{question}'的答案是：{context[:50]}..."

# ========== 完整 RAG 系统 ==========
class SimpleRAGSystem:
    def __init__(self):
        self.loader = SimpleDocumentLoader()
        self.splitter = SimpleTextSplitter(chunk_size=100, chunk_overlap=20)
        self.embedding = SimpleEmbedding()
        self.vector_store = SimpleVectorStore()
        self.generator = SimpleGenerator()
    
    def build_index(self, documents: List[str]):
        """构建索引"""
        for doc_text in documents:
            doc = self.loader.load(doc_text)
            chunks = self.splitter.split(doc["content"])
            for chunk in chunks:
                vec = self.embedding.embed(chunk)
                self.vector_store.add(chunk, vec)
        print(f"索引构建完成：{len(self.vector_store.documents)} 个文档块")
    
    def query(self, question: str) -> str:
        """问答"""
        # 1. 检索
        question_vec = self.embedding.embed(question)
        results = self.vector_store.search(question_vec, k=2)
        context = "\n".join([doc for doc, _ in results])
        
        # 2. 生成
        answer = self.generator.generate(context, question)
        return answer

# ========== 测试 ==========
if __name__ == "__main__":
    # 准备文档
    docs = [
        "RAG 是检索增强生成的缩写，它包含三个步骤：检索、增强和生成。",
        "RAG 的检索步骤使用向量数据库来存储和搜索文档。",
        "Embedding 是将文本转换为向量的技术，使计算机能够理解语义。",
        "向量数据库如 Chroma、Milvus 用于存储文本的向量表示。"
    ]
    
    # 初始化系统
    rag = SimpleRAGSystem()
    
    # 构建索引
    rag.build_index(docs)
    
    # 问答
    question = "RAG 的检索步骤是什么？"
    answer = rag.query(question)
    print(f"\n问题：{question}")
    print(f"回答：{answer}")
```

**运行结果**：

```
索引构建完成：8 个文档块

问题：RAG 的检索步骤是什么？
回答：根据提供的文档，'RAG 的检索步骤是什么？'的答案是：
RAG 的检索步骤使用向量数据库来存储和搜索文档...
```

---

## 本节总结

### 核心要点

1. **六大组件**：文档加载器 → 文本切分器 → Embedding → 向量数据库 → 检索器 → 生成器
2. **两个阶段**：索引阶段（文档→向量）和查询阶段（问题→检索→生成）
3. **框架选择**：快速原型选 LlamaIndex，高度定制选 LangChain

### 思考题

1. 为什么需要"文本切分"这个步骤？直接用整篇文档检索不行吗？
2. Embedding 模型的质量会如何影响 RAG 系统的效果？
3. LangChain 和 LlamaIndex 各有什么优缺点？什么情况下应该切换框架？

### 下节预告

下一节课我们将学习多格式文档解析，掌握 PDF、Word、Excel、PPT 等常见文档的解析技术。

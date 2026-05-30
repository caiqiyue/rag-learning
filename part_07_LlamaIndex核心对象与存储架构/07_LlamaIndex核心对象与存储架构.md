# 第09节 LlamaIndex核心对象与存储架构

## 学习目标

1. 理解LlamaIndex的核心概念（Document、Node、Index）
2. 掌握StorageContext和Document的存储管理
3. 能够使用LlamaIndex构建完整RAG流程

---

## 知识点讲解

### 1. Document（文档对象）

Document是LlamaIndex中最基础的数据单元，用于表示一个完整的文档。

**核心属性**：
- `text`: 文档的文本内容
- `metadata`: 元数据字典，用于存储文档的来源、日期、作者等信息

**特点**：
- Document是原始数据的容器，不直接参与向量检索
- 会被TextSplitter分割成多个Node
- 支持自定义元数据，便于后续的元数据过滤

### 2. Node（节点对象）

Node是Document分割后的结果，是实际参与索引和检索的单元。

**核心属性**：
- `text`: 节点的文本内容
- `metadata`: 继承自父Document的元数据
- `embedding`: 该节点的向量嵌入（可选）
- `relationships`: 父子节点关系信息

**关键特点**：
- 保留父子关系：知道它是从哪个Document分割而来，以及与其他Node的关联
- 保留索引信息：包含在原始文档中的位置信息
- 是VectorStoreIndex构建的最小单元

### 3. StorageContext（存储架构）

StorageContext是LlamaIndex的存储管理核心，统一管理四种核心存储：

| 存储类型 | 说明 | 典型实现 |
|---------|------|---------|
| **VectorStore** | 向量嵌入存储 | Chroma、Milvus、Faiss、Pinecone |
| **Docstore** | 原始文档/节点内容存储 | SimpleDocumentStore、MongoDocumentStore |
| **IndexStore** | 索引元数据存储 | SimpleIndexStore、MongoIndexStore |
| **GraphStore** | 知识图谱节点关系存储 | SimpleGraphStore |

**创建方式**：
```python
# 默认方式（内存存储）
storage_context = StorageContext.from_defaults()

# 持久化方式
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# 自定义存储后端
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore,
    index_store=index_store,
    graph_store=graph_store
)
```

### 4. VectorStoreIndex（向量索引）

VectorStoreIndex是基于向量相似性搜索的索引实现。

**核心功能**：
- 将Document或Node列表转换为可检索的向量索引
- 支持语义搜索和相似度查询
- 提供QueryEngine和Retriever接口

**创建方式**：
```python
# 从文档创建（自动分割）
index = VectorStoreIndex.from_documents(documents)

# 从节点创建
index = VectorStoreIndex(nodes)

# 指定存储后端
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

### 5. QueryEngine（查询引擎）

QueryEngine是LlamaIndex的查询处理核心，封装了检索和生成流程。

**核心流程**：
1. 接收用户查询
2. 将查询转换为向量
3. 检索相关Node
4. 组装Prompt发送给LLM
5. 返回生成结果

**获取方式**：
```python
query_engine = index.as_query_engine()
response = query_engine.query("用户问题")
```

### 6. Retriever（检索器）

Retriever专门负责从索引中检索相关Node，不涉及LLM生成。

**特点**：
- 轻量级，只返回相关节点
- 支持自定义相似度top_k
- 支持元数据过滤
- 适合评估和调试场景

**获取方式**：
```python
retriever = index.as_retriever(similarity_top_k=5)
results = retriever.retrieve("查询文本")
```

### 7. 核心对象协作关系

```
Document (原始文档)
    ↓ TextSplitter分割
Node (节点)
    ↓ 存储到 StorageContext
VectorStoreIndex (向量索引)
    ↓ 构建
QueryEngine / Retriever (查询接口)
```

**架构层次**：
- StorageContext向下管理具体存储后端
- VectorStoreIndex向上提供查询接口
- QueryEngine处理检索+生成
- Retriever专注于纯检索

---

## 代码案例

### 文件路径

```
E:/ai-learning/09-agent-engineering/rag-learning/part_07_LlamaIndex核心对象与存储架构/07_codes/llamaindex_core.py
```

### 运行说明

1. **安装依赖**：
```bash
pip install llama_index llama-index-embeddings-openai llama-index-vector-stores-chroma chromadb python-dotenv
```

2. **配置环境变量**：在项目根目录创建 `.env` 文件，配置以下变量：
```
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

3. **运行代码**：
```bash
python E:/ai-learning/09-agent-engineering/rag-learning/part_07_LlamaIndex核心对象与存储架构/07_codes/llamaindex_core.py
```

### 代码说明

该脚本演示了：
- Document和Node的创建与使用
- StorageContext的四种存储管理
- VectorStoreIndex构建向量索引
- QueryEngine查询引擎的使用
- Retriever检索器的使用
- 完整RAG流程的实现
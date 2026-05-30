"""
LlamaIndex核心对象与存储架构 - 代码示例

本文件演示LlamaIndex框架的核心对象和存储管理功能，包括：
1. Document和Node的创建与使用
2. StorageContext的四种存储管理
3. VectorStoreIndex构建向量索引
4. QueryEngine和Retriever的使用
5. 完整RAG流程的实现

作者: RAG学习课程
版本: 1.0
"""

# ============================================================
# 第一部分：Document（文档对象）
# ============================================================
# Document是LlamaIndex中最基础的数据单元，用于表示一个完整的文档
# 它包含文本内容和元数据，是构建索引的输入数据

from llama_index.core import Document

# 创建Document对象
# text: 文档的文本内容
# metadata: 元数据字典，用于存储文档的来源、日期等信息
doc = Document(
    text="这是一篇关于人工智能大语言模型的介绍文档。",
    metadata={
        "source": "AI简介.txt",
        "author": "技术团队",
        "date": "2024-01-15"
    }
)

print("=" * 60)
print("第一部分：Document对象创建")
print("=" * 60)
print(f"文档内容: {doc.text}")
print(f"文档元数据: {doc.metadata}")
print(f"文档ID: {doc.doc_id}")


# ============================================================
# 第二部分：Node（节点对象）
# ============================================================
# Node是Document分割后的结果，是实际参与索引和检索的最小单元
# 每个Node包含：文本内容、继承的元数据、可选的向量嵌入、节点关系信息

from llama_index.core import Node

# 创建Node对象
# Node可以独立创建，也可以从Document分割得到
node = Node(
    text="这是节点1的文本内容，包含人工智能的相关知识点。",
    metadata={
        "source": "AI文档",
        "page": 1,
        "section": "第一章"
    },
    # relationships记录该节点与其他节点的关系
    relationships={}  # 可关联父Document、其他Node等
)

print("\n" + "=" * 60)
print("第二部分：Node对象创建")
print("=" * 60)
print(f"节点内容: {node.text}")
print(f"节点元数据: {node.metadata}")
print(f"节点ID: {node.node_id}")


# ============================================================
# 第三部分：StorageContext（存储架构）
# ============================================================
# StorageContext是LlamaIndex的存储管理核心
# 统一管理四种核心存储：VectorStore、Docstore、IndexStore、GraphStore

from llama_index.core import StorageContext
from llama_index.storage.docstore.simple import SimpleDocumentStore
from llama_index.storage.index_store.simple import SimpleIndexStore
from llama_index.storage.graph_store.simple import SimpleGraphStore

print("\n" + "=" * 60)
print("第三部分：StorageContext存储架构")
print("=" * 60)

# 方式1：使用默认配置（内存存储）
# 适用于快速验证和开发测试
storage_context_default = StorageContext.from_defaults()
print("1. 默认存储上下文（内存）创建成功")

# 方式2：持久化存储到磁盘
# 将存储数据保存到指定目录，适合生产环境
storage_context_persist = StorageContext.from_defaults(persist_dir="./storage")
print("2. 持久化存储上下文创建成功")

# 方式3：使用SimpleDocumentStore（文档存储）
# Docstore负责存储原始文档/节点的内容
docstore = SimpleDocumentStore()
print("3. Docstore创建成功")

# 方式4：使用SimpleIndexStore（索引元数据存储）
# IndexStore负责存储索引的元数据信息
index_store = SimpleIndexStore()
print("4. IndexStore创建成功")

# 方式5：使用SimpleGraphStore（图结构存储）
# GraphStore用于知识图谱场景，存储节点关系
graph_store = SimpleGraphStore()
print("5. GraphStore创建成功")

# 方式6：组合多种存储后端
# 可以同时指定VectorStore、Docstore、IndexStore、GraphStore
storage_context_full = StorageContext.from_defaults(
    docstore=docstore,      # 原始文档内容存储
    index_store=index_store,  # 索引元数据存储
    graph_store=graph_store  # 图结构存储
)
print("6. 组合存储上下文创建成功")


# ============================================================
# 第四部分：TextNode（文本节点，Node的子类）
# ============================================================
# TextNode是Node的子类，专门用于文本内容的节点表示
# 这是更常用的节点类型

from llama_index.core.schema import TextNode

# 创建TextNode
text_node = TextNode(
    text="这是一个TextNode节点，包含可嵌入的文本内容。",
    metadata={
        "source": "示例文档",
        "type": "text_node"
    }
)

print("\n" + "=" * 60)
print("第四部分：TextNode文本节点")
print("=" * 60)
print(f"TextNode内容: {text_node.text}")
print(f"TextNode ID: {text_node.node_id}")


# ============================================================
# 第五部分：VectorStoreIndex（向量存储索引）
# ============================================================
# VectorStoreIndex是LlamaIndex最常用的索引类型
# 基于向量相似性搜索，支持语义检索

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

print("\n" + "=" * 60)
print("第五部分：VectorStoreIndex向量索引")
print("=" * 60)

# 创建多个TextNode组成节点列表
nodes = [
    TextNode(
        text="大语言模型是人工智能领域的重要突破。",
        metadata={"source": "AI基础.txt", "topic": "LLM"}
    ),
    TextNode(
        text="Transformer架构是现代LLM的核心技术。",
        metadata={"source": "AI基础.txt", "topic": "Transformer"}
    ),
    TextNode(
        text="RAG技术结合了检索和生成的优势。",
        metadata={"source": "AI基础.txt", "topic": "RAG"}
    ),
    TextNode(
        text="向量数据库用于存储和检索文本嵌入。",
        metadata={"source": "AI基础.txt", "topic": "向量数据库"}
    ),
]

# 方式1：从节点列表创建索引（不使用默认的embed_model）
# 注意：如果没有配置embed_model，会使用默认的设置
index = VectorStoreIndex(nodes)
print("1. 从节点列表创建VectorStoreIndex成功")
print(f"   索引包含 {len(nodes)} 个节点")

# 方式2：从文档列表创建索引（自动分割）
# VectorStoreIndex.from_documents会自动调用TextSplitter分割文档
documents = [
    Document(
        text="这是一个关于机器学习的文档。机器学习是人工智能的子领域。",
        metadata={"source": "机器学习.txt"}
    ),
    Document(
        text="深度学习使用神经网络模型进行特征学习。",
        metadata={"source": "深度学习.txt"}
    )
]
index_from_docs = VectorStoreIndex.from_documents(documents)
print("2. 从文档列表创建VectorStoreIndex成功（自动分割）")

# 方式3：使用自定义storage_context
index_with_storage = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context_default
)
print("3. 使用自定义storage_context创建索引成功")


# ============================================================
# 第六部分：Retriever（检索器）
# ============================================================
# Retriever专门负责从索引中检索相关Node
# 不涉及LLM生成，只返回相关节点列表
# 适合评估、调参和需要获取原始检索结果的场景

print("\n" + "=" * 60)
print("第六部分：Retriever检索器")
print("=" * 60)

# 创建检索器
# similarity_top_k: 返回前k个最相关的节点
retriever = index.as_retriever(similarity_top_k=2)
print("1. Retriever创建成功（top_k=2）")

# 执行检索
query_text = "什么是大语言模型？"
search_results = retriever.retrieve(query_text)
print(f"2. 检索查询: '{query_text}'")
print(f"3. 检索到 {len(search_results)} 个相关节点:")
for i, result in enumerate(search_results):
    print(f"   结果{i+1}:")
    print(f"     相似度分数: {result.score:.4f}")
    print(f"     节点内容: {result.node.get_text()[:50]}...")


# ============================================================
# 第七部分：QueryEngine（查询引擎）
# ============================================================
# QueryEngine封装了检索和生成的全流程
# 接收用户查询，检索相关节点，组装Prompt，调用LLM生成回答

print("\n" + "=" * 60)
print("第七部分：QueryEngine查询引擎")
print("=" * 60)

# 创建查询引擎
# as_query_engine()会使用默认的LLM设置
query_engine = index.as_query_engine()
print("1. QueryEngine创建成功")

# 执行查询（需要配置LLM才能真正运行）
try:
    response = query_engine.query("大语言模型的核心技术是什么？")
    print(f"2. 查询结果: {response}")
except Exception as e:
    print(f"2. 查询执行（需要配置LLM）: {type(e).__name__}")


# ============================================================
# 第八部分：完整RAG流程演示
# ============================================================
# 完整的RAG流程包括：
# 1. 准备文档
# 2. 构建索引
# 3. 创建查询引擎
# 4. 执行问答

print("\n" + "=" * 60)
print("第八部分：完整RAG流程")
print("=" * 60)

# Step 1: 准备文档数据
print("Step 1: 准备文档数据")
rag_documents = [
    Document(
        text="RAG（检索增强生成）是一种结合检索和生成的技术架构。"
              "它首先从知识库中检索相关文档，然后基于检索结果生成回答。"
              "RAG可以解决LLM的幻觉问题和知识时效问题。",
        metadata={"source": "RAG简介", "category": "技术概述"}
    ),
    Document(
        text="LlamaIndex是一个专为LLM应用设计的数据框架。"
              "它提供了文档加载、文本分割、索引构建、查询检索等全流程功能。"
              "LlamaIndex支持多种向量数据库和存储后端。",
        metadata={"source": "LlamaIndex介绍", "category": "框架介绍"}
    ),
    Document(
        text="向量数据库是存储和检索高维向量的专用数据库。"
              "常见的向量数据库包括Milvus、Chroma、Faiss、Pinecone等。"
              "向量检索通过计算向量之间的相似度来找到最相关的内容。",
        metadata={"source": "向量数据库", "category": "技术概述"}
    ),
]
print(f"   创建了 {len(rag_documents)} 个文档")

# Step 2: 构建向量索引
print("Step 2: 构建向量索引")
rag_index = VectorStoreIndex.from_documents(rag_documents)
print("   VectorStoreIndex构建成功")

# Step 3: 创建查询引擎
print("Step 3: 创建查询引擎")
rag_query_engine = rag_index.as_query_engine()
print("   QueryEngine创建成功")

# Step 4: 执行问答
print("Step 4: 执行问答")
questions = [
    "什么是RAG技术？",
    "LlamaIndex有哪些主要功能？",
    "向量数据库的作用是什么？"
]
for question in questions:
    print(f"\n   问题: {question}")
    try:
        answer = rag_query_engine.query(question)
        print(f"   回答: {answer}")
    except Exception as e:
        print(f"   回答（需要配置LLM）: {type(e).__name__}")


# ============================================================
# 第九部分：使用Chroma持久化向量存储
# ============================================================
# Chroma是一个轻量级的向量数据库，适合本地开发和测试
# 通过storage_context可以持久化存储向量数据

print("\n" + "=" * 60)
print("第九部分：Chroma持久化向量存储")
print("=" * 60)

try:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext

    # 创建Chroma客户端和集合
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("llamaindex_demo")

    # 创建Chroma向量存储
    vector_store = ChromaVectorStore(chroma_collection)

    # 创建带有Chroma的StorageContext
    chroma_storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # 使用Chroma存储创建索引
    chroma_index = VectorStoreIndex.from_documents(
        rag_documents,
        storage_context=chroma_storage_context
    )
    print("1. Chroma向量存储创建成功")

    # 从Chroma存储创建检索器
    chroma_retriever = chroma_index.as_retriever(similarity_top_k=2)
    chroma_results = chroma_retriever.retrieve("RAG是什么？")
    print(f"2. Chroma检索到 {len(chroma_results)} 个相关节点")
    for i, result in enumerate(chroma_results):
        print(f"   结果{i+1} - 分数: {result.score:.4f}")

except ImportError:
    print("Chroma未安装，跳过持久化示例")
    print("安装命令: pip install llama-index-vector-stores-chroma chromadb")
except Exception as e:
    print(f"Chroma示例执行异常: {type(e).__name__}")


# ============================================================
# 第十部分：对象关系总结
# ============================================================
print("\n" + "=" * 60)
print("第十部分：LlamaIndex核心对象关系总结")
print("=" * 60)

print("""
LlamaIndex核心对象层次关系：
=========================

1. Document（文档对象）
   - 原始数据的容器
   - 包含text和metadata
   - 是构建索引的输入

2. Node（节点对象）
   - Document分割后的结果
   - 是实际参与索引的最小单元
   - 保留父子关系和元数据

3. StorageContext（存储架构）
   - 管理四种核心存储：
     * VectorStore: 向量嵌入
     * Docstore: 原始文档
     * IndexStore: 索引元数据
     * GraphStore: 图关系
   - 支持持久化和自定义后端

4. VectorStoreIndex（向量索引）
   - 基于向量的语义索引
   - 支持相似度检索
   - 提供查询接口

5. QueryEngine（查询引擎）
   - 封装检索+生成流程
   - 调用LLM生成回答
   - 返回自然语言答案

6. Retriever（检索器）
   - 纯检索，不生成
   - 返回相关节点列表
   - 适合评估和调试

工作流程：
Document -> TextSplitter -> Node -> VectorStoreIndex -> Retriever/QueryEngine
                                          |
                                  StorageContext
                                  (管理底层存储)
""")

print("\n" + "=" * 60)
print("代码演示完成！")
print("=" * 60)
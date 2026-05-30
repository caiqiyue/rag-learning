"""
向量数据库选型与对比示例
==========================

本文件演示主流向量数据库（Chroma、Milvus、Pinecone、FAISS）的特点和选型，
帮助开发者根据业务场景选择合适的向量数据库。

知识点：
- Chroma: 轻量级本地向量库，适合POC和小规模数据
- FAISS: Facebook开源的向量检索库，适合大规模数据本地部署
- Milvus: 开源分布式向量数据库，支持水平扩展
- Pinecone: 云原生向量数据库，提供Serverless服务
"""

# ============================================================
# 第一部分：Chroma向量数据库 - 轻量级本地方案
# ============================================================
"""
【Chroma特点】
- 轻量级、嵌入式向量库，安装简单
- 适合POC验证和小规模数据（<100万向量）
- 支持持久化存储，Python原生API
- 免费开源，适合学习和实验
- 缺点：不支持水平扩展，不适合生产环境大规模部署
"""

def chroma_example():
    """Chroma向量数据库示例"""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    import os

    # 步骤1：创建Chroma持久化客户端
    # Chroma将数据存储在本地指定目录，适合需要快速验证的场景
    db_path = "./chroma_db"
    os.makedirs(db_path, exist_ok=True)

    # 创建持久化客户端，数据会实时写入磁盘
    chroma_client = chromadb.PersistentClient(path=db_path)

    # 步骤2：获取或创建集合（类似于表）
    # 集合名称建议使用有意义的名称，便于管理
    collection_name = "documents_collection"
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    # 步骤3：创建向量存储适配器
    # ChromaVectorStore将LlamaIndex的向量操作转换为Chroma的操作
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 步骤4：创建存储上下文
    # StorageContext定义了文档、向量、索引的存储方式
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 步骤5：创建索引并插入文档
    # from llama_index.core import Document
    # documents = [Document(text="文档内容", metadata={"source": "file1.pdf"})]
    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     show_progress=True
    # )

    print("Chroma示例完成")
    print(f"  - 存储路径: {db_path}")
    print(f"  - 集合名称: {collection_name}")
    print("  - 适用场景: POC验证、小规模数据、快速原型开发")


# ============================================================
# 第二部分：FAISS向量索引 - 高效本地大规模检索
# ============================================================
"""
【FAISS特点】
- Facebook开源的向量检索库，专注高效相似度搜索
- 支持数十亿级向量索引，适合大规模数据
- 提供多种索引类型（IVF、HNSW、PQ等），平衡精度与性能
- 纯本地部署，无额外成本，适合有技术团队的企业
- 缺点：不支持分布式，需要自己管理数据持久化
"""

def faiss_example():
    """FAISS向量数据库示例"""
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    # 步骤1：创建FAISS索引
    # IndexFlatL2是最简单的索引类型，使用L2距离（欧氏距离）
    # 优点：精确检索，缺点：大数据量时查询慢
    dimension = 1536  # 向量维度，应与Embedding模型输出维度一致

    # 创建L2距离的Flat索引
    faiss_index = faiss.IndexFlatL2(dimension)

    # 步骤2：创建向量存储适配器
    # 注意：FAISS只存储向量，原始文档数据需要额外存储（Docstore）
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 步骤3：创建索引
    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     show_progress=True
    # )

    print("FAISS示例完成")
    print(f"  - 向量维度: {dimension}")
    print(f"  - 索引类型: IndexFlatL2 (精确检索)")
    print("  - 适用场景: 大规模本地数据、高精度需求、成本敏感型项目")


def faiss_ivf_example():
    """FAISS IVF索引示例 - 适用于大数据量"""
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore

    dimension = 1536

    # IVF（倒排文件）索引：先将向量聚类，查询时只搜索最近的聚类中心
    # nlist参数：聚类数量，通常设置为sqrt(数据量)
    nlist = 100  # 聚类中心数量

    # 步骤1：创建量化器（用于聚类）
    quantizer = faiss.IndexFlatL2(dimension)

    # 步骤2：创建IVF索引
    # IVF索引将数据分成nlist个聚类，查询时只搜索最近的几个聚类
    faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

    # 步骤3：训练索引（IVF需要先训练）
    # 注意：训练需要代表性的数据样本
    # faiss_index.train(training_vectors)
    # faiss_index.add(vector_data)

    # 设置搜索时的聚类搜索数量，nprobe越大越精确但越慢
    faiss_index.nprobe = 10  # 默认搜索10个最近的聚类

    vector_store = FaissVectorStore(faiss_index=faiss_index)

    print("FAISS IVF索引示例完成")
    print(f"  - 聚类数量: {nlist}")
    print(f"  - 搜索聚类数: {faiss_index.nprobe}")
    print("  - 适用场景: 亿级向量数据、需要平衡精度与性能")


# ============================================================
# 第三部分：Milvus向量数据库 - 开源分布式方案
# ============================================================
"""
【Milvus特点】
- 开源分布式向量数据库，支持水平扩展
- 支持多种索引类型（IVF、HNSW、DiskANN等）
- 提供云原生部署（Kubernetes）和本地Docker部署
- 适合中等规模生产环境（百万到亿级向量）
- 支持混合查询（向量+标量过滤）
- 缺点：运维有一定复杂度，需要专业团队维护
"""

def milvus_example():
    """Milvus向量数据库示例"""
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    # 步骤1：创建Milvus向量存储连接
    # dim: 向量维度，与Embedding模型输出维度一致
    # collection_name: 集合名称，类似于表名
    # uri: Milvus服务器地址，localhost:19530是默认端口
    # overwrite: 是否覆盖已有数据
    vector_store = MilvusVectorStore(
        dim=1536,
        collection_name="milvus_collection",
        uri="http://localhost:19530",  # Milvus默认端口
        overwrite=True
    )

    # 步骤2：创建存储上下文
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 步骤3：创建索引
    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     show_progress=True
    # )

    print("Milvus示例完成")
    print("  - 连接地址: http://localhost:19530")
    print("  - 适用场景: 中等规模生产环境、需要水平扩展的项目")


def milvus_docker_setup():
    """Milvus Docker Compose快速启动说明"""
    setup_guide = """
    【Milvus快速启动步骤】

    1. 下载docker-compose文件:
       curl -L -o docker-compose.yml https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml

    2. 启动Milvus服务:
       docker-compose up -d

    3. 等待服务启动（约30秒），然后连接使用

    4. 停止服务:
       docker-compose down

    【默认配置】
    - Milvus服务端口: 19530
    - Milvus管理界面: http://localhost:9090
    - 数据存储目录: ./volumes
    """
    print(setup_guide)


# ============================================================
# 第四部分：Pinecone向量数据库 - 云原生Serverless
# ============================================================
"""
【Pinecone特点】
- 云原生向量数据库，提供完全托管的Serverless服务
- 无需运维，按使用量付费，适合快速成长的应用
- 全球分布式部署，低延迟高可用
- 支持精确检索和近似检索多种模式
- 适合不确定规模或需要快速扩展的初创公司
- 缺点：需要付费，成本随数据量和查询量增长
"""

def pinecone_example():
    """Pinecone向量数据库示例"""
    import os
    from dotenv import load_dotenv
    from pinecone import Pinecone, ServerlessSpec
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    # 加载环境变量
    load_dotenv()

    # 步骤1：初始化Pinecone客户端
    # 需要先设置PINECONE_API_KEY环境变量
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # 步骤2：定义索引名称和配置
    index_name = "my-first-index"
    dimension = 1536  # 必须与Embedding模型维度一致

    # 步骤3：创建Serverless索引
    # ServerlessSpec定义云端资源配置
    # region: AWS区域
    # cloud: 云服务商 (aws、azure、gcp)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # 距离度量方式：cosine余弦、euclidean欧氏、dotproduct点积
            spec=ServerlessSpec(
                region="us-east-1",
                cloud="aws"
            )
        )

    # 步骤4：获取索引连接
    pinecone_index = pc.Index(index_name)

    # 步骤5：创建LlamaIndex向量存储适配器
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        text_key="text",  # 原始文本字段名
        namespace="",     # 命名空间，用于数据隔离
        insert_kwargs={"batch_size": 100}  # 批量插入配置
    )

    # 步骤6：创建存储上下文
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 步骤7：创建索引
    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     show_progress=True
    # )

    print("Pinecone示例完成")
    print(f"  - 索引名称: {index_name}")
    print(f"  - 区域: us-east-1")
    print("  - 适用场景: Serverless架构、快速扩展、无运维能力团队")


# ============================================================
# 第五部分：向量数据库选型决策指南
# ============================================================
"""
【向量数据库选型决策树】

                          开始
                            │
                            ▼
                    ┌───────────────────┐
                    │  数据规模多大？   │
                    └───────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
        <100万向量      100万-亿级       >亿级
            │               │               │
            ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Chroma/FAISS│  │ Milvus/Pinecone│ │ Milvus分布式│
    └─────────────┘  └─────────────┘  └─────────────┘
            │               │               │
            ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ 本地优先？   │  │ 有运维团队？│  │ 需要成本    │
    └─────────────┘  └─────────────┘  │ 控制？      │
            │               │          └─────────────┘
            ▼               ▼               │
        是 │           是 │               ▼
            ▼               ▼         ┌─────────────┐
    ┌─────────────┐  ┌─────────────┐  │ FAISS集群/  │
    │   Chroma    │  │   Milvus    │  │ Milvus集群  │
    └─────────────┘  └─────────────┘  └─────────────┘
            │               │
            ▼               ▼
            │           否 │
            ▼               ▼
            └─────────────┘
                    │
                    ▼
            ┌─────────────┐
            │  Pinecone   │
            └─────────────┘

【选型建议汇总表】

数据库       适用规模       部署方式       成本          适用场景
--------------------------------------------------------------------------------
Chroma      <100万        本地          免费          POC验证、快速原型、小型项目
FAISS       任意规模      本地          免费          大规模数据、高精度需求、有技术团队
Milvus      百万-亿级     云原生        中等（自建）  中等规模生产、需要水平扩展
Pinecone    任意规模      完全托管      按量付费      快速成长应用、无运维能力、Serverless架构
Qdrant      百万-亿级     云原生/本地   中等          需要高性能HNSW索引
Weaviate    任意规模      云原生/本地   中等          混合查询、多模态数据
"""

def select_vector_db_guide():
    """向量数据库选型指南"""
    guide = """
    【快速选型决策】

    1. 小规模验证/POC → Chroma（免费、快速）
       - 数据量<100万向量
       - 需要快速验证概念
       - 没有专职运维

    2. 大规模本地部署 → FAISS（免费、高精度）
       - 数据量>1000万向量
       - 有技术团队可以运维
       - 成本敏感，不愿使用云服务

    3. 中等规模生产环境 → Milvus（自建）
       - 百万到亿级向量
       - 有Kubernetes运维能力
       - 需要水平扩展

    4. Serverless/快速启动 → Pinecone（托管）
       - 不想管理基础设施
       - 应用可能快速扩展
       - 愿意为便利性付费

    5. 高性能检索需求 → Qdrant/Weaviate
       - 需要HNSW等高性能索引
       - 有混合查询需求

    【关键考虑因素】

    ① 向量维度：大多数模型输出1536维，部分模型需要更大维度
    ② 距离度量：cosine余弦（常用）、euclidean欧氏、dotproduct点积
    ③ 索引类型：Flat（精确）、IVF（聚类）、HNSW（图索引）、PQ（量化压缩）
    ④ 扩展性：数据增长预估，决定选择单机还是分布式方案
    ⑤ 成本：云服务按存储和查询量收费，本地方案有运维成本
    ⑥ 混合过滤：是否需要支持元数据标量过滤（大部分场景需要）
    """
    print(guide)


# ============================================================
# 主函数：运行所有示例
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("向量数据库选型示例")
    print("=" * 60)

    # 注意：各数据库需要先安装对应包和配置环境
    # !pip install llama-index-vector-stores-chroma llama-index-vector-stores-milvus llama-index-vector-stores-pinecone llama-index-vector-stores-faiss chromadb pinecone-client faiss-cpu

    print("\n【1. Chroma示例】")
    # chroma_example()  # 取消注释运行

    print("\n【2. FAISS示例】")
    # faiss_example()  # 取消注释运行

    print("\n【3. FAISS IVF索引示例】")
    # faiss_ivf_example()  # 取消注释运行

    print("\n【4. Milvus示例】")
    # milvus_example()  # 取消注释运行
    milvus_docker_setup()  # 打印Milvus启动说明

    print("\n【5. Pinecone示例】")
    # pinecone_example()  # 取消注释运行

    print("\n【6. 选型决策指南】")
    select_vector_db_guide()

    print("\n" + "=" * 60)
    print("完成！根据业务场景选择合适的向量数据库。")
    print("=" * 60)
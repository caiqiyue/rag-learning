"""
元数据过滤查询实战示例
======================

本文件演示如何在向量数据库中设计元数据结构，
并使用MetadataFilter实现高效的过滤查询。

知识点：
- 元数据结构设计：如何设计有效的元数据字段
- MetadataFilter：LlamaIndex中的元数据过滤器
- 过滤操作符：EQ（等于）、NE（不等于）、GT/GTE/LT/LTE（比较）
- 复合过滤：多条件组合AND/OR逻辑
"""

# ============================================================
# 第一部分：元数据结构设计
# ============================================================
"""
【元数据设计原则】

元数据是与向量关联的附加信息，用于描述文档来源、类型、时间等属性。
良好的元数据设计可以：
1. 支持精确过滤查询，减少无关搜索范围
2. 提高检索效率和精度
3. 支持多维度业务分析

【常见元数据字段】

字段名          类型        说明                    示例值
----------------------------------------------------------------------
source          string      文档来源/文件名          "合同文档.pdf"
page_number     int         页码                   5
element_type    string      元素类型                "Text"/"Table"/"Image"
category        string      文档分类                "技术文档"/"商务文档"
year            int         年份                    2024
author          string      作者                   "张三"
department      string      部门                    "研发部"
confidential    bool        是否机密                true/false
tags            list        标签列表                ["AI", "RAG", "向量"]
language        string      语言                    "中文"/"英文"
"""

def design_metadata_example():
    """元数据结构设计示例"""
    from llama_index.core import Document

    # 业务文档的元数据设计示例
    documents = [
        # 示例1：合同文档
        Document(
            text="本合同有效期限为2024年1月1日至2024年12月31日...",
            metadata={
                "source": "合同文档-A公司.pdf",  # 来源文件
                "page_number": 1,                  # 页码
                "element_type": "Text",            # 文本类型
                "category": "商务合同",             # 文档分类
                "year": 2024,                      # 年份
                "department": "法务部",            # 所属部门
                "confidential": True,              # 机密标识
                "tags": ["合同", "A公司", "有效期"]  # 标签
            }
        ),

        # 示例2：技术文档
        Document(
            text="RAG系统的Embedding模型选择指南...",
            metadata={
                "source": "RAG技术文档.pdf",
                "page_number": 3,
                "element_type": "Text",
                "category": "技术文档",
                "year": 2024,
                "department": "研发部",
                "confidential": False,
                "tags": ["RAG", "Embedding", "选型"]
            }
        ),

        # 示例3：财务报表
        Document(
            text="2024年第一季度营收分析报告...",
            metadata={
                "source": "财务报表-Q1.pdf",
                "page_number": 2,
                "element_type": "Table",           # 表格类型
                "category": "财务报告",
                "year": 2024,
                "department": "财务部",
                "confidential": True,
                "tags": ["财务", "营收", "Q1"]
            }
        ),

        # 示例4：英文文档
        Document(
            text="Best practices for RAG system deployment...",
            metadata={
                "source": "RAG_Best_Practices.pdf",
                "page_number": 1,
                "element_type": "Text",
                "category": "技术文档",
                "year": 2024,
                "department": "研发部",
                "confidential": False,
                "language": "英文",
                "tags": ["RAG", "Best_Practices"]
            }
        )
    ]

    print("元数据结构设计示例")
    print("=" * 50)
    print(f"创建了 {len(documents)} 个带元数据的文档")
    for doc in documents:
        print(f"  - {doc.metadata['source']} (类型:{doc.metadata['element_type']}, 部门:{doc.metadata['department']})")

    return documents


# ============================================================
# 第二部分：MetadataFilter过滤器详解
# ============================================================
"""
【MetadataFilter核心概念】

MetadataFilter用于在向量相似度检索的基础上添加元数据过滤条件，
可以显著缩小检索范围，提高查询精度。

【LlamaIndex中的过滤器类】

1. MetadataFilter - 单个过滤条件
   - key: 元数据字段名
   - value: 过滤值
   - operator: 过滤操作符（可选，默认EQ）

2. MetadataFilters - 多个过滤条件组合
   - filters: 多个MetadataFilter列表
   - condition: 条件逻辑（AND/OR）

3. FilterOperator - 操作符枚举
   - EQ: 等于 (equal)
   - NE: 不等于 (not equal)
   - GT: 大于 (greater than)
   - GTE: 大于等于 (greater than or equal)
   - LT: 小于 (less than)
   - LTE: 小于等于 (less than or equal)
   - CONTAINS: 包含（字符串）
   - IN: 属于列表
"""

from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)


def filter_basics_example():
    """MetadataFilter基础用法"""
    print("【MetadataFilter基础用法】\n")

    # 示例1：相等过滤 - 过滤文档类型
    # 场景：只检索文本类型，不检索表格
    filter_eq = MetadataFilter(
        key="element_type",
        value="Text",
        operator=FilterOperator.EQ  # 等于
    )
    print(f"1. 相等过滤: element_type == 'Text'")
    print(f"   代码: MetadataFilter(key='element_type', value='Text', operator=FilterOperator.EQ)")

    # 示例2：不等过滤 - 排除机密文档
    filter_ne = MetadataFilter(
        key="confidential",
        value=True,
        operator=FilterOperator.NE  # 不等于
    )
    print(f"\n2. 不等过滤: confidential != True")
    print(f"   代码: MetadataFilter(key='confidential', value=True, operator=FilterOperator.NE)")

    # 示例3：比较过滤 - 筛选特定年份后的文档
    filter_gt = MetadataFilter(
        key="year",
        value=2023,
        operator=FilterOperator.GT  # 大于
    )
    print(f"\n3. 大于过滤: year > 2023")
    print(f"   代码: MetadataFilter(key='year', value=2023, operator=FilterOperator.GT)")

    # 示例4：范围过滤 - 筛选年份范围
    filter_range = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),  # >= 2020
            MetadataFilter(key="year", value=2025, operator=FilterOperator.LTE),  # <= 2025
        ],
        condition=FilterCondition.AND  # 同时满足
    )
    print(f"\n4. 范围过滤: 2020 <= year <= 2025")
    print(f"   代码: MetadataFilters(filters=[...], condition=FilterCondition.AND)")

    # 示例5：IN过滤 - 属于列表中任意一个
    filter_in = MetadataFilter(
        key="department",
        value=["研发部", "产品部"],
        operator=FilterOperator.IN  # 属于列表
    )
    print(f"\n5. IN过滤: department in ['研发部', '产品部']")
    print(f"   代码: MetadataFilter(key='department', value=['研发部', '产品部'], operator=FilterOperator.IN)")

    # 示例6：包含过滤 - 字符串包含
    filter_contains = MetadataFilter(
        key="source",
        value="合同",
        operator=FilterOperator.CONTAINS  # 包含字符串
    )
    print(f"\n6. 包含过滤: source contains '合同'")
    print(f"   代码: MetadataFilter(key='source', value='合同', operator=FilterOperator.CONTAINS)")

    return {
        "filter_eq": filter_eq,
        "filter_ne": filter_ne,
        "filter_gt": filter_gt,
        "filter_range": filter_range,
        "filter_in": filter_in,
        "filter_contains": filter_contains
    }


# ============================================================
# 第三部分：过滤查询实战 - Chroma示例
# ============================================================
"""
【Chroma中实现元数据过滤】

Chroma作为嵌入式向量库，原生支持元数据过滤，
LlamaIndex的ChromaVectorStore完整保留了这一特性。
"""

def chroma_filter_example():
    """Chroma元数据过滤查询示例"""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition

    print("【Chroma元数据过滤查询示例】\n")

    # 步骤1：初始化Chroma
    db_path = "./chroma_filter_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="filtered_collection")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 步骤2：创建索引（假设已有文档）
    # from llama_index.core import Document
    # documents = design_metadata_example()
    # index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # 步骤3：过滤查询 - 只查询文本类型的文档
    # filters = MetadataFilters(filters=[
    #     MetadataFilter(key="element_type", value="Text", operator=FilterOperator.EQ)
    # ])
    # retriever = index.as_retriever(filters=filters, similarity_top_k=5)
    # results = retriever.retrieve("RAG系统部署指南")

    print("Chroma过滤查询示例代码:")
    print("""
    # 创建过滤器：只检索文本类型
    filters = MetadataFilters(filters=[
        MetadataFilter(key="element_type", value="Text", operator=FilterOperator.EQ)
    ])

    # 配置检索器
    retriever = index.as_retriever(filters=filters, similarity_top_k=5)

    # 执行检索
    results = retriever.retrieve("RAG系统部署指南")

    # 输出结果
    for node in results:
        print(f"来源: {node.metadata['source']}, 内容: {node.text[:100]}...")
    """)

    # 步骤4：复合过滤示例 - 研发部的非机密文本文档
    composite_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="department", value="研发部", operator=FilterOperator.EQ),
            MetadataFilter(key="confidential", value=False, operator=FilterOperator.EQ),
            MetadataFilter(key="element_type", value="Text", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.AND  # 三个条件同时满足
    )
    print("\n复合过滤示例 - 研发部的非机密文本文档:")
    print(f"  - 部门 = '研发部'")
    print(f"  - 机密 = False")
    print(f"  - 类型 = 'Text'")
    print(f"  - 逻辑关系: AND（全部满足）")

    return storage_context


# ============================================================
# 第四部分：过滤查询实战 - Milvus示例
# ============================================================
"""
【Milvus中实现元数据过滤】

Milvus作为专业向量数据库，支持强大的标量过滤能力。
在创建集合时可以指定可过滤字段，然后使用MetadataFilter进行查询。
"""

def milvus_filter_example():
    """Milvus元数据过滤查询示例"""
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

    print("【Milvus元数据过滤查询示例】\n")

    # 步骤1：创建Milvus向量存储连接
    vector_store = MilvusVectorStore(
        dim=1536,
        collection_name="filtered_milvus_collection",
        uri="http://localhost:19530",
        overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 步骤2：创建索引
    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     show_progress=True
    # )

    # 步骤3：使用MetadataFilter进行过滤查询
    # Milvus支持在检索时直接使用元数据过滤条件

    print("Milvus过滤查询示例代码:")
    print("""
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

    # 示例1：单条件过滤 - 筛选特定年份
    filters = MetadataFilters(filters=[
        MetadataFilter(key="year", value=2024, operator=FilterOperator.EQ)
    ])

    retriever = index.as_retriever(filters=filters, similarity_top_k=5)
    results = retriever.retrieve("合同条款分析")

    # 示例2：多条件过滤 - 按部门和年份
    filters = MetadataFilters(filters=[
        MetadataFilter(key="department", value="法务部", operator=FilterOperator.EQ),
        MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
    ], condition=FilterCondition.AND)

    retriever = index.as_retriever(filters=filters, similarity_top_k=5)
    results = retriever.retrieve("年度合同统计")
    """)

    # 步骤4：高级过滤 - 年份范围+部门+类型组合
    advanced_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),  # 2020年及以后
            MetadataFilter(key="department", value=["法务部", "财务部"], operator=FilterOperator.IN),  # 指定部门
            MetadataFilter(key="confidential", value=False, operator=FilterOperator.EQ),  # 非机密
        ],
        condition=FilterCondition.AND
    )
    print("\n高级过滤示例 - 2020年后法务部或财务部的非机密文档:")
    print(f"  - 年份 >= 2020")
    print(f"  - 部门 in ['法务部', '财务部']")
    print(f"  - 机密 = False")
    print(f"  - 逻辑关系: AND（全部满足）")

    return storage_context


# ============================================================
# 第五部分：过滤查询实战 - Pinecone示例
# ============================================================
"""
【Pinecone中实现元数据过滤】

Pinecone的Serverless架构支持高效的元数据过滤。
在创建索引后可使用where子句进行复杂过滤。
"""

def pinecone_filter_example():
    """Pinecone元数据过滤查询示例"""
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

    print("【Pinecone元数据过滤查询示例】\n")

    print("Pinecone过滤查询示例代码:")
    print("""
    import os
    from dotenv import load_dotenv
    from pinecone import Pinecone, ServerlessSpec
    from llama_index.vector_stores.pinecone import PineconeVectorStore

    # 初始化Pinecone
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # 创建Serverless索引
    index_name = "metadata_filtered_index"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(region="us-east-1", cloud="aws")
        )

    pinecone_index = pc.Index(index_name)

    # 创建向量存储
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        text_key="text",
        namespace="default"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Pinecone过滤查询
    # Pinecone使用filter参数进行元数据过滤
    filters = MetadataFilters(filters=[
        MetadataFilter(key="category", value="技术文档", operator=FilterOperator.EQ),
        MetadataFilter(key="confidential", value=False, operator=FilterOperator.EQ),
    ], condition=FilterCondition.AND)

    retriever = index.as_retriever(filters=filters, similarity_top_k=10)
    results = retriever.retrieve("RAG系统最佳实践")

    for result in results:
        print(f"来源: {result.metadata['source']}, 分数: {result.score}")
    """)


# ============================================================
# 第六部分：过滤查询最佳实践
# ============================================================
"""
【元数据过滤最佳实践】

1. 过滤字段选择
   - 优先选择区分度高的字段（如类别、类型）
   - 避免在高基数字段（如完整文本内容）上过滤
   - 时间范围过滤是最常见场景

2. 组合过滤优化
   - 将最精确的条件放在前面
   - 避免过多AND条件（增加计算量）
   - 考虑使用OR连接相近条件，减少重复

3. 索引策略
   - Chroma: 内存存储，过滤性能一般
   - Milvus: 列式存储，过滤性能优秀
   - Pinecone: 云端优化，过滤性能强

4. 常见业务场景

   场景1：用户反馈分析
   - 过滤条件：year=2024, category="用户反馈"
   - 查询："产品体验问题总结"

   场景2：合同条款检索
   - 过滤条件：source contains "合同", year>=2023
   - 查询："违约金条款"

   场景3：部门知识库
   - 过滤条件：department="研发部", confidential=False
   - 查询："代码规范要求"

   场景4：多语言文档
   - 过滤条件：language="英文", tags CONTAINS "RAG"
   - 查询："RAG deployment guide"
"""

def best_practices_example():
    """元数据过滤最佳实践"""
    print("【元数据过滤最佳实践】\n")

    scenarios = [
        {
            "name": "场景1：用户反馈分析",
            "filters": [
                ("year", "2024", FilterOperator.EQ),
                ("category", "用户反馈", FilterOperator.EQ),
            ],
            "query": "产品体验问题总结"
        },
        {
            "name": "场景2：合同条款检索",
            "filters": [
                ("source", "合同", FilterOperator.CONTAINS),
                ("year", 2023, FilterOperator.GTE),
            ],
            "query": "违约金条款"
        },
        {
            "name": "场景3：研发部知识库",
            "filters": [
                ("department", "研发部", FilterOperator.EQ),
                ("confidential", False, FilterOperator.EQ),
            ],
            "query": "代码规范要求"
        },
        {
            "name": "场景4：多语言文档搜索",
            "filters": [
                ("language", "英文", FilterOperator.EQ),
                ("tags", "RAG", FilterOperator.CONTAINS),
            ],
            "query": "RAG deployment guide"
        }
    ]

    for scenario in scenarios:
        print(f"{scenario['name']}")
        print(f"  查询内容: \"{scenario['query']}\"")
        print(f"  过滤条件:")
        for key, value, op in scenario['filters']:
            op_name = {
                FilterOperator.EQ: "==",
                FilterOperator.GTE: ">=",
                FilterOperator.CONTAINS: "contains"
            }.get(op, str(op))
            print(f"    - {key} {op_name} {value}")
        print()


# ============================================================
# 第七部分：过滤性能对比
# ============================================================
"""
【各数据库过滤性能对比】

数据库       过滤能力       性能           适用场景
---------------------------------------------------------------------------
Chroma       基础          内存级快速     小规模数据、简单条件
Milvus       强大          列式优化       大规模数据、复杂条件
Pinecone     强大          云端优化       任意规模、云原生
FAISS        弱（需过滤）   依赖索引       不推荐元数据过滤

【建议】
- 小规模POC：使用Chroma，配置简单
- 生产环境：优先Milvus或Pinecone，过滤能力强
- 需要复杂过滤：选择Milvus，支持SQL-like过滤语法
"""

def performance_comparison():
    """各数据库过滤性能对比"""
    print("【各数据库过滤性能对比】\n")
    print("数据库       | 过滤能力 | 性能     | 适用场景")
    print("-" * 60)
    print("Chroma      | 基础      | 内存快速  | 小规模、简单条件")
    print("Milvus      | 强大      | 列式优化  | 大规模、复杂条件")
    print("Pinecone    | 强大      | 云端优化  | 任意规模、云原生")
    print("FAISS       | 弱        | 依赖索引  | 不推荐元数据过滤")
    print()
    print("【选型建议】")
    print("  - 小规模POC验证 → Chroma")
    print("  - 生产环境复杂过滤 → Milvus")
    print("  - Serverless架构 → Pinecone")


# ============================================================
# 主函数：运行所有示例
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("元数据过滤查询实战示例")
    print("=" * 60)

    print("\n【1. 元数据结构设计】")
    design_metadata_example()

    print("\n【2. MetadataFilter过滤器详解】")
    filter_basics_example()

    print("\n【3. Chroma过滤查询】")
    chroma_filter_example()

    print("\n【4. Milvus过滤查询】")
    milvus_filter_example()

    print("\n【5. Pinecone过滤查询】")
    pinecone_filter_example()

    print("\n【6. 最佳实践】")
    best_practices_example()

    print("\n【7. 过滤性能对比】")
    performance_comparison()

    print("\n" + "=" * 60)
    print("完成！元数据过滤可以显著提升检索精度和效率。")
    print("=" * 60)
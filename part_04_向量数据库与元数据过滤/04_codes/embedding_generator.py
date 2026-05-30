"""
第04节 向量化模型选型与嵌入生成 - 代码示例

本模块演示如何使用主流Embedding模型进行文本向量化，包括：
1. OpenAI Embedding 云端API的使用
2. HuggingFace Embedding 本地开源模型的使用
3. 文本向量化的基本操作
4. 相似度计算示例

Author: AI Learning
"""

# ============================================================
# 第一部分：导入必要的库
# ============================================================
from llama_index.core.settings import Settings  # LlamaIndex全局设置
import os  # 操作系统库，用于读取环境变量
import numpy as np  # 数值计算库，用于向量运算
from dotenv import load_dotenv  # 环境变量加载库

# ----------------------------------------------------
# OpenAI Embedding 相关导入
# ----------------------------------------------------
from llama_index.embeddings.openai import OpenAIEmbedding

# ----------------------------------------------------
# HuggingFace Embedding 相关导入
# ----------------------------------------------------
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============================================================
# 第二部分：加载环境变量
# ============================================================
# 从.env文件加载环境变量（OPENAI_API_KEY等）
load_dotenv()


# ============================================================
# 第三部分：OpenAI Embedding 云端API使用示例
# ============================================================

def demo_openai_embedding():
    """
    演示如何使用OpenAI Embedding模型进行文本向量化

    OpenAI Embedding特点：
    - 云端API，无需本地部署
    - 支持text-embedding-3-large和text-embedding-3-small两种模型
    - text-embedding-3-small性价比高，适合大规模索引
    - text-embedding-3-large质量更高，适合精确检索场景
    """
    print("=" * 60)
    print("OpenAI Embedding 示例")
    print("=" * 60)

    # 创建OpenAI Embedding模型实例
    # 参数说明：
    # - model: 使用的模型名称，text-embedding-3-small是高性价比选择
    # - api_key: OpenAI API密钥，从环境变量读取
    # - api_base: API地址，支持自定义兼容层
    # - dimensions: 返回向量的维度，可根据需求调整（1536或512）
    # - embed_batch_size: 批量处理时的批次大小，影响吞吐量和内存使用
    # - timeout: 请求超时时间（秒）
    # - max_retries: 最大重试次数
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",  # 模型的维度是1536
        api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量获取API密钥
        api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),  # API地址
        dimensions=1536,  # 控制返回向量的维度，可选512或1536
        embed_batch_size=100,  # 批量处理文本时每个批次的文本数量
        timeout=60,  # 请求超时时间
        max_retries=3  # 批量失败时的重试次数
    )

    # 单文本向量化：使用get_query_embedding对单个文本进行向量化
    # 适用于用户查询的即时向量化
    query_text = "如何提高RAG系统的检索效果？"
    query_embedding = embed_model.get_query_embedding(query_text)
    print(f"\n查询文本: {query_text}")
    print(f"向量化后的维度: {len(query_embedding)}")
    print(f"前5个维度值: {query_embedding[:5]}")

    # 批量文本向量化：使用get_embeddings对多个文本进行批量向量化
    # 适用于文档入库时的批量处理，提高效率
    documents = [
        "RAG是一种结合检索和生成的技术架构",
        "向量数据库用于存储和检索文本向量",
        "Embedding模型将文本转换为向量表示",
        "相似度计算是向量检索的核心"
    ]

    # get_embeddings一次处理多个文本，比循环调用get_query_embedding更高效
    doc_embeddings = embed_model.get_embeddings(documents)
    print(f"\n批量向量化文档数量: {len(doc_embeddings)}")
    print(f"每个向量的维度: {len(doc_embeddings[0])}")

    return embed_model


# ============================================================
# 第四部分：HuggingFace Embedding 本地开源模型使用示例
# ============================================================

def demo_huggingface_embedding():
    """
    演示如何使用HuggingFace Embedding本地开源模型进行文本向量化

    本地开源模型特点：
    - 数据可控，适合对数据安全有要求的场景
    - 长期TCO友好，无需按调用付费
    - 可离线使用，不依赖网络
    - 支持多种模型：BGE、M3E、Jina等

    常用中文Embedding模型：
    - BAAI/bge-large-zh-v1.5: BGE大模型，中文效果好
    - BAAI/bge-small-zh-v1.5: BGE小模型，速度快
    - moka-ai/m3e-base: M3E基础模型，中英双语优
    """
    print("\n" + "=" * 60)
    print("HuggingFace Embedding 本地模型示例")
    print("=" * 60)

    # 创建HuggingFace Embedding模型实例
    # 参数说明：
    # - model_name: 模型名称，支持HuggingFace上的各种嵌入模型
    # - device: 运行环境，"cpu"表示使用CPU，"cuda:0"表示使用第一块GPU
    # - embed_batch_size: 批量处理时的批次大小
    # 注意：首次使用时会自动下载模型，可能需要一些时间
    embed_model = HuggingFaceEmbedding(
        model_name='BAAI/bge-large-zh-v1.5',  # BGE大模型，支持中文
        device="cpu",  # 有GPU的机器可设置为"cuda:0"
        embed_batch_size=64,  # 批量处理时的批次大小
    )

    # 单文本向量化
    query_text = "RAG系统的核心组件有哪些？"
    query_embedding = embed_model.get_query_embedding(query_text)
    print(f"\n查询文本: {query_text}")
    print(f"向量化后的维度: {len(query_embedding)}")
    print(f"前5个维度值: {query_embedding[:5]}")

    # 批量文本向量化
    documents = [
        "大模型RAG应用开发实战课程介绍",
        "向量数据库是RAG系统的核心组件",
        "Embedding模型决定检索质量的上限",
        "文档分割策略影响RAG效果"
    ]

    doc_embeddings = embed_model.get_embeddings(documents)
    print(f"\n批量向量化文档数量: {len(doc_embeddings)}")
    print(f"每个向量的维度: {len(doc_embeddings[0])}")

    return embed_model


# ============================================================
# 第五部分：相似度计算示例
# ============================================================

def calculate_cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    余弦相似度衡量两个向量夹角的余弦值：
    - 值接近1表示两个向量方向相同（语义相似）
    - 值接近0表示两个向量正交（语义无关）
    - 值接近-1表示两个向量方向相反（语义相反）

    参数:
        vec1: 第一个向量（列表或numpy数组）
        vec2: 第二个向量（列表或numpy数组）

    返回:
        float: 余弦相似度值，范围[-1, 1]
    """
    # 将列表转换为numpy数组，便于向量运算
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 计算余弦相似度：cosine = (A · B) / (||A|| × ||B||)
    dot_product = np.dot(vec1, vec2)  # 向量点积
    norm_vec1 = np.linalg.norm(vec1)  # 向量1的模长
    norm_vec2 = np.linalg.norm(vec2)  # 向量2的模长

    # 避免除以零的情况
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity


def calculate_euclidean_distance(vec1, vec2):
    """
    计算两个向量的欧氏距离

    欧氏距离是向量空间的直线距离：
    - 值接近0表示两个向量距离近（语义相似）
    - 值越大表示两个向量距离远（语义不相似）

    参数:
        vec1: 第一个向量（列表或numpy数组）
        vec2: 第二个向量（列表或numpy数组）

    返回:
        float: 欧氏距离值
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 计算欧氏距离：sqrt(sum((a-b)^2))
    euclidean_distance = np.linalg.norm(vec1 - vec2)
    return euclidean_distance


def demo_similarity_calculation(embed_model):
    """
    演示如何使用计算的相似度进行文本匹配

    该函数展示了从向量化到相似度计算的完整流程：
    1. 将多个文档向量化
    2. 将查询向量化
    3. 计算查询与各文档的相似度
    4. 根据相似度排序找出最相关的文档
    """
    print("\n" + "=" * 60)
    print("相似度计算示例")
    print("=" * 60)

    # 待检索的文档集合
    documents = [
        "人工智能技术的发展历程",
        "Python编程语言入门教程",
        "深度学习在图像识别中的应用",
        "机器学习算法原理与实践"
    ]

    # 用户查询
    query_text = "深度学习技术介绍"

    # 获取查询向量
    query_embedding = embed_model.get_query_embedding(query_text)

    # 获取所有文档的向量
    doc_embeddings = embed_model.get_embeddings(documents)

    # 计算查询与每个文档的相似度
    print(f"\n查询文本: {query_text}")
    print("\n文档相似度排序：")
    print("-" * 50)

    similarities = []
    for i, (doc, doc_emb) in enumerate(zip(documents, doc_embeddings)):
        # 计算余弦相似度
        cosine_sim = calculate_cosine_similarity(query_embedding, doc_emb)
        # 计算欧氏距离
        euclidean_dist = calculate_euclidean_distance(query_embedding, doc_emb)

        similarities.append({
            'index': i,
            'document': doc,
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist
        })

    # 按余弦相似度降序排序
    similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)

    # 打印排序后的结果
    for rank, item in enumerate(similarities, 1):
        print(f"\n排名 {rank}:")
        print(f"文档: {item['document']}")
        print(f"余弦相似度: {item['cosine_similarity']:.4f}")
        print(f"欧氏距离: {item['euclidean_distance']:.4f}")

    return similarities


# ============================================================
# 第六部分：LlamaIndex全局Settings配置示例
# ============================================================

def demo_llamaindex_settings():
    """
    演示如何在LlamaIndex中配置全局Embedding模型

    通过Settings.embed_model设置全局默认的Embedding模型，
    这样在后续的索引构建和检索过程中会自动使用该模型
    """
    print("\n" + "=" * 60)
    print("LlamaIndex全局Settings配置示例")
    print("=" * 60)

    # 配置全局默认Embedding模型为OpenAI Embedding
    # 设置后，所有的索引构建和检索都会默认使用这个模型
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        dimensions=1536,
        embed_batch_size=100,
        timeout=60,
        max_retries=3
    )

    print("\n已设置全局默认Embedding模型: OpenAI text-embedding-3-small")
    print(f"模型维度: {Settings.embed_model.dimensions}")

    # 验证配置：进行一个简单的向量化测试
    test_text = "测试全局Embedding模型配置"
    test_embedding = Settings.embed_model.get_query_embedding(test_text)
    print(f"测试文本: {test_text}")
    print(f"向量维度: {len(test_embedding)}")


# ============================================================
# 第七部分：模型对比示例
# ============================================================

def demo_model_comparison():
    """
    演示不同Embedding模型的特点和适用场景

    主要对比维度：
    - 向量维度：维度越高通常表示模型能表达更丰富的语义，但存储成本更高
    - 最大序列长度：影响单次能处理的文本长度
    - 速度：本地模型 vs 云端API
    - 成本：本地模型无需API费用，但有硬件成本
    """
    print("\n" + "=" * 60)
    print("Embedding模型对比")
    print("=" * 60)

    models_info = [
        {
            "name": "OpenAI text-embedding-3-large",
            "type": "云端API",
            "dimensions": 3072,
            "max_tokens": 8192,
            "features": "顶级语义理解、多语言能力强",
            "use_case": "高质量检索、复杂和多语言业务"
        },
        {
            "name": "OpenAI text-embedding-3-small",
            "type": "云端API",
            "dimensions": 1536,
            "max_tokens": 8192,
            "features": "高性价比、性能均衡",
            "use_case": "大规模索引、成本敏感型应用"
        },
        {
            "name": "BAAI/bge-large-zh-v1.5",
            "type": "本地开源",
            "dimensions": 1024,
            "max_tokens": 512,
            "features": "中文语义理解强、长文本支持好",
            "use_case": "中文主导场景、企业内部知识库"
        },
        {
            "name": "moka-ai/m3e-base",
            "type": "本地开源",
            "dimensions": 768,
            "max_tokens": 512,
            "features": "中英双语优、长文本支持",
            "use_case": "多语种检索、中英混合场景"
        }
    ]

    print("\n{:<35} {:<10} {:>10} {:>12} {}".format(
        "模型名称", "类型", "维度", "最大Token", "适用场景"
    ))
    print("-" * 90)

    for model in models_info:
        print("{:<35} {:<10} {:>10} {:>12} {}".format(
            model["name"],
            model["type"],
            model["dimensions"],
            model["max_tokens"],
            model["use_case"]
        ))

    print("\n选型建议：")
    print("1. 合规要求严格、数据不出网 -> 选择本地开源模型（BGE/M3E）")
    print("2. 多语种全球化业务 -> 选择云API（OpenAI/Cohere）")
    print("3. 预算有限、入门尝试 -> text-embedding-3-small 或 jina-embeddings-v2")
    print("4. 对精度要求极高 -> Embedding + Reranker 组合使用")


# ============================================================
# 主函数：运行所有示例
# ============================================================

def main():
    """
    主函数，按顺序执行所有演示代码

    流程：
    1. OpenAI Embedding示例
    2. HuggingFace Embedding示例
    3. 相似度计算示例
    4. LlamaIndex全局配置示例
    5. 模型对比示例
    """
    print("\n" + "#" * 60)
    print("# 第04节 向量化模型选型与嵌入生成 - 代码示例")
    print("#" * 60)

    # 示例1：OpenAI Embedding云端API使用
    openai_model = demo_openai_embedding()

    # 示例2：HuggingFace Embedding本地模型使用
    hf_model = demo_huggingface_embedding()

    # 示例3：使用HuggingFace模型演示相似度计算
    demo_similarity_calculation(hf_model)

    # 示例4：LlamaIndex全局Settings配置
    demo_llamaindex_settings()

    # 示例5：模型对比
    demo_model_comparison()

    print("\n" + "#" * 60)
    print("# 所有示例执行完成！")
    print("#" * 60)


# 当直接运行此脚本时，执行main()函数
# 当作为模块导入时，不会自动执行
if __name__ == "__main__":
    main()
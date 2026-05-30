"""
RAG基础概念代码演示

本代码演示了RAG系统的完整工作流程：
1. 文档加载 - 从Word文档中读取内容
2. 文档分割 - 将长文档切分成小片段
3. 向量化 - 将文本转换为数学向量
4. 向量存储 - 使用FAISS存储向量
5. 相似度检索 - 在向量库中搜索相关内容
6. 答案生成 - 构建RAG链进行智能问答

作者: RAG学习课程
日期: 2026-05-30
"""

# ============================================================
# 第一部分：导入必要的库
# ============================================================
import os

# 设置HuggingFace镜像地址（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# LangChain文档加载器 - 用于加载各种格式的文档
from langchain_community.document_loaders import Docx2txtLoader

# LangChain文本分割器 - 用于将长文档切分成小片段
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain向量数据库 - 用于存储和检索向量
from langchain_community.vectorstores import FAISS

# LangChain嵌入模型 - 用于将文本转为向量
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain提示模板 - 用于组合检索结果和用户问题
from langchain_core.prompts import ChatPromptTemplate

# LangChain输出解析器 - 用于解析LLM输出
from langchain_core.output_parsers import StrOutputParser

# LangChain Runnable组件 - 用于构建LCEL链
from langchain_core.runnables import RunnablePassthrough

# LangChain OpenAI模型 - 用于生成答案
from langchain_openai import ChatOpenAI


# ============================================================
# 第二部分：加载环境变量
# ============================================================
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 获取OpenAI API配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')


# ============================================================
# 第三部分：知识库构建（离线阶段）
# ============================================================

print("=" * 60)
print("【第一阶段】知识库构建（离线阶段）")
print("=" * 60)

# ---------------------------------------------------------
# 3.1 文档加载
# ---------------------------------------------------------
print("\n3.1 文档加载...")
print("-" * 40)

# 初始化Word文档加载器
# Docx2txtLoader专门用于加载Microsoft Word文档(.docx)
loader = Docx2txtLoader("公司假期制度.docx")

# 加载文档，返回Document对象列表
# 每个Document对象包含page_content（文本内容）和metadata（元数据）
documents = loader.load()

# 打印加载结果
print(f"✓ 成功加载 {len(documents)} 个文档")
if documents:
    print(f"✓ 文档来源: {documents[0].metadata.get('source', '未知')}")
    # 只显示前200个字符作为预览
    preview = documents[0].page_content[:200]
    print(f"✓ 内容预览: {preview}...")

# ---------------------------------------------------------
# 3.2 文档分割
# ---------------------------------------------------------
print("\n3.2 文档分割...")
print("-" * 40)

# 创建文本分割器
# RecursiveCharacterTextSplitter是常用的分割器，按层次递归分割
text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size: 每个文本块的最大字符数
    # 设置较小值可以提高检索精度，避免信息过载
    chunk_size=500,

    # chunk_overlap: 相邻文本块之间的重叠字符数
    # 设置重叠可以保持上下文连贯性，避免重要信息被切断
    chunk_overlap=100,

    # separators: 分割符优先级列表
    # 从左到右优先级递减，先尝试双换行，再单换行，再句末标点
    separators=["\n\n", "\n", "。", "！", "？", "，", ""],

    # add_start_index: 是否记录每个块在原文档中的起始位置
    # 开启后可以在检索时追溯原文位置
    add_start_index=True
)

# 执行分割
# split_documents方法会遍历文档列表，逐个分割
all_splits = text_splitter.split_documents(documents)

# 打印分割结果
print(f"✓ 分割完成，共得到 {len(all_splits)} 个文本块")
print(f"✓ 第一个文本块字符数: {len(all_splits[0].page_content)}")
print(f"✓ 第一个文本块起始位置: {all_splits[0].metadata.get('start_index', '未知')}")

# ---------------------------------------------------------
# 3.3 向量化（Embedding）
# ---------------------------------------------------------
print("\n3.3 向量化（Embedding）...")
print("-" * 40)

# 初始化BGE中文嵌入模型
# BGE (BAAI General Embedding) 是国产优秀的Embedding模型
# model_name: 模型名称，BAAI/bge-large-zh-v1.5 是专门针对中文的高性能版本
# model_kwargs: 模型配置，device='cpu'表示使用CPU计算（GPU可改为'cuda'）
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device': 'cpu'}
)

# 验证Embedding模型是否正常工作
test_text = "机器学习是人工智能的重要分支"
test_vector = embeddings.embed_documents([test_text])[0]
print(f"✓ Embedding模型加载成功")
print(f"✓ 向量维度: {len(test_vector)}")
print(f"✓ 测试文本: '{test_text}'")
print(f"✓ 向量预览: {test_vector[:3]}...")

# ---------------------------------------------------------
# 3.4 向量存储
# ---------------------------------------------------------
print("\n3.4 向量存储...")
print("-" * 40)

# 创建FAISS向量数据库
# from_documents方法接收文档列表和嵌入模型，自动完成向量化并存储
# FAISS是Facebook开源的高性能向量检索库，支持海量向量相似度搜索
vectorstore = FAISS.from_documents(all_splits, embeddings)

print(f"✓ 向量数据库创建成功")
print(f"✓ 存储了 {len(all_splits)} 个文本块的向量")

# 保存向量数据库到本地（可选）
# 这样下次使用时无需重新构建，可以直接加载
save_path = "word_doc_faiss_index"
vectorstore.save_local(save_path)
print(f"✓ 已保存到本地目录: {save_path}")


# ============================================================
# 第四部分：问答推理（在线阶段）
# ============================================================

print("\n" + "=" * 60)
print("【第二阶段】问答推理（在线阶段）")
print("=" * 60)

# ---------------------------------------------------------
# 4.1 加载向量数据库
# ---------------------------------------------------------
print("\n4.1 加载向量数据库...")
print("-" * 40)

# 从本地加载已保存的向量数据库
# allow_dangerous_deserialization=True: 允许反序列化操作
# 这是因为LangChain默认禁止加载可能不安全的序列化数据
vectorstore = FAISS.load_local(
    save_path,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"✓ 向量数据库加载成功")

# ---------------------------------------------------------
# 4.2 用户提问
# ---------------------------------------------------------
print("\n4.2 用户提问...")
print("-" * 40)

# 定义用户问题
query = "调休的申请流程是什么？"
print(f"✓ 用户问题: {query}")

# ---------------------------------------------------------
# 4.3 相似度检索
# ---------------------------------------------------------
print("\n4.3 相似度检索...")
print("-" * 40)

# 创建检索器
# as_retriever方法将向量库转换为检索器
# search_kwargs: 检索参数，k=3表示返回最相关的3个文档
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 执行相似度搜索
# similarity_search_with_score返回文档列表和相似度分数
docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

print(f"✓ 检索到 {len(docs_with_scores)} 个相关文档:")
print()

# 遍历检索结果
for i, (doc, score) in enumerate(docs_with_scores):
    print(f"  --- 结果 {i + 1} ---")
    print(f"  相似度分数: {score:.4f}")  # 分数越低越相似（FAISS使用L2距离）
    print(f"  内容: {doc.page_content[:100]}...")
    print()

# ---------------------------------------------------------
# 4.4 提示增强
# ---------------------------------------------------------
print("\n4.4 提示增强...")
print("-" * 40)

# 定义RAG提示词模板
# 模板中包含两个变量：context（上下文）和question（问题）
# context会被替换为检索到的相关文档
# question会被替换为用户的原始问题
template = """请根据以下上下文信息回答问题。如果上下文中有答案，请基于上下文回答；如果没有，请说明。

上下文信息：
{context}

问题：{question}

请给出详细、准确的回答："""

# 创建提示模板对象
prompt = ChatPromptTemplate.from_template(template)

print(f"✓ 提示模板创建成功")
print(f"✓ 模板预览:")
print(f"  {template[:100]}...")

# ---------------------------------------------------------
# 4.5 答案生成
# ---------------------------------------------------------
print("\n4.5 答案生成...")
print("-" * 40)

# 初始化OpenAI语言模型
# model: 使用的模型名称，gpt-3.5-turbo是性价比高的选择
# temperature: 温度参数，控制输出的随机性
#   - 0.0 = 完全确定性输出
#   - 0.7 = 平衡创意性和准确性
#   - 1.0+ = 更多随机性
llm_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# 定义格式化文档的辅助函数
# 将多个文档合并为一个字符串，用双换行分隔
def format_docs(docs):
    """
    将文档列表格式化为单个字符串

    参数:
        docs: Document对象列表

    返回:
        格式化后的字符串
    """
    return "\n\n".join(doc.page_content for doc in docs)

# 构建LCEL链（LangChain Expression Language）
# LCEL是一种声明式的方式来组合LangChain组件
#
# 链的工作流程：
# 1. {"context": retriever | format_docs, "question": RunnablePassthrough()}
#    - retriever检索相关文档
#    - format_docs将文档格式化为字符串
#    - RunnablePassthrough()传递用户原始问题
#    - 结果是 {"context": "...", "question": "..."}
#
# 2. | prompt
#    - 将字典传递给提示模板，填充context和question变量
#
# 3. | llm_model
#    - 将填充后的提示发送给LLM生成答案
#
# 4. | StrOutputParser()
#    - 将LLM输出解析为字符串

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
)

print(f"✓ RAG链构建成功")

# 执行问答
print("\n" + "-" * 40)
print("【最终问答结果】")
print("-" * 40)
print(f"用户问题: {query}")
print()

# 调用RAG链进行问答
response = rag_chain.invoke(query)

print(f"模型回答: {response}")

# ---------------------------------------------------------
# 4.6 完整流程演示
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("【完整流程回顾】")
print("=" * 60)

print("""
RAG系统完整工作流程：

┌─────────────────────────────────────────────────────────────┐
│  阶段一：知识库构建（离线）                                    │
├─────────────────────────────────────────────────────────────┤
│  1. 文档加载 → Docx2txtLoader读取Word文档                      │
│  2. 文档分割 → RecursiveCharacterTextSplitter切分文本           │
│  3. 向量化 → HuggingFaceEmbeddings(BGE模型)                    │
│  4. 存储 → FAISS向量数据库                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段二：问答推理（在线）                                      │
├─────────────────────────────────────────────────────────────┤
│  1. 用户提问 → "调休的申请流程是什么？"                        │
│  2. 问题向量化 → BGE模型将问题转为向量                          │
│  3. 相似度检索 → FAISS中找到最相关的3个文档                     │
│  4. 提示增强 → 将检索结果组合成新Prompt                         │
│  5. 答案生成 → GPT-3.5-turbo生成最终回答                       │
└─────────────────────────────────────────────────────────────┘

关键组件说明：
- Embedding模型：负责将文本转为向量（用户问题和文档必须使用相同模型）
- 向量数据库：负责存储和检索向量（FAISS、ChromaDB、Milvus等）
- LLM模型：负责基于增强后的Prompt生成最终答案
- 提示模板：负责将检索结果和用户问题组合成适合LLM的格式
""")

print("\n" + "=" * 60)
print("【代码演示结束】")
print("=" * 60)
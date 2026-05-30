"""
RAG常见误区与最佳实践代码演示

本代码演示RAG开发中的常见误区及其解决方案：
1. 分割策略误区 - 固定chunk size vs 动态分割
2. 检索器选择误区 - 单一检索 vs 混合检索
3. 提示词工程误区 - 忽略上下文构建
4. 向量数据库选择误区 - 忽视元数据过滤
5. 缺乏评估与迭代

学习目标：
- 识别RAG开发中的常见误区
- 掌握企业级RAG最佳实践
- 能够避免RAG系统设计中的常见陷阱

作者: RAG学习课程
日期: 2026-05-30
"""

# ============================================================
# 第一部分：导入必要的库
# ============================================================
import os

# 设置HuggingFace镜像地址（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 文档加载相关
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# 向量数据库和嵌入
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate

# LLM模型
from langchain_openai import ChatOpenAI

# 环境变量
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# 第二部分：准备示例文档数据
# ============================================================
print("=" * 70)
print("【示例文档准备】模拟企业知识库场景")
print("=" * 70)

# 创建一个模拟的文档列表，模拟企业产品知识库
# 包含：产品手册、FAQ、政策文件等不同类型文档
sample_documents = [
    # 产品手册片段 - 关于产品A的规格
    {
        "content": """
        产品A技术规格说明书

        一、产品概述
        产品A是一款高性能服务器，专为企业级应用设计。

        二、技术规格
        - 处理器：Intel Xeon Gold 6248R (24核3.0GHz)
        - 内存：256GB DDR4 ECC RAM
        - 存储：2TB NVMe SSD RAID 5
        - 网络：双10GbE网口

        三、适用场景
        适用于数据中心、云计算平台、机器学习训练等场景。

        四、注意事项
        1. 请确保电源稳定，建议使用UPS
        2. 工作温度范围：10°C - 35°C
        3. 相对湿度：20% - 80%（无凝露）
        """,
        "metadata": {"source": "产品A技术手册", "category": "product_manual", "product": "ProductA"}
    },
    # FAQ片段 - 客户常见问题
    {
        "content": """
        客户支持FAQ

        Q1: 产品A的保修期是多久？
        A: 产品A标准保修期为3年，可付费延长至5年。

        Q2: 如何申请技术支持？
        A: 您可以通过以下方式联系技术支持：
           - 电话：400-123-4567
           - 邮箱：support@company.com
           - 官网：www.company.com/support

        Q3: 产品A支持哪些操作系统？
        A: 产品A支持以下操作系统：
           - Windows Server 2019/2022
           - Ubuntu Server 18.04/20.04/22.04
           - CentOS 7.x / 8.x
           - Red Hat Enterprise Linux 7.x / 8.x

        Q4: 遇到蓝屏错误怎么办？
        A: 蓝屏错误通常是驱动或硬件问题。请：
           1. 记录蓝屏错误代码
           2. 检查最近安装的软件或驱动
           3. 联系技术支持提供错误代码
        """,
        "metadata": {"source": "客户支持FAQ", "category": "faq", "product": "ProductA"}
    },
    # 政策文件片段 - 退换货政策
    {
        "content": """
        退换货政策

        一、退货政策
        自购买之日起7天内（以发票日期为准），如产品存在质量问题，
        可申请退货退款。退货产品需保持原包装完整，配件齐全。

        二、换货政策
        自购买之日起30天内，如产品出现非人为损坏的故障，
        可申请换货服务。换货产品享受原有保修期。

        三、保修服务
        1. 三年有限保修：涵盖硬件故障
        2. 保修期内的维修服务免费
        3. 人为损坏不在保修范围内

        四、特殊情况
        - 定制品、拆封软件不在退换货范围内
        - 因客户使用不当造成的损坏需自费维修
        """,
        "metadata": {"source": "退换货政策", "category": "policy", "product": "all"}
    },
    # 技术支持文档片段
    {
        "content": """
        产品A常见技术问题排查

        1. 无法开机
           - 检查电源连接是否正常
           - 检查电源开关是否打开
           - 尝试长按电源键10秒强制关机后再开机

        2. 网络连接不稳定
           - 检查网线是否插好
           - 检查网口指示灯状态
           - 更新网卡驱动程序
           - 更换网线测试

        3. 系统运行缓慢
           - 检查CPU和内存使用率
           - 检查磁盘空间是否充足
           - 运行系统诊断工具
           - 检查是否有恶意软件

        4. 温度过高报警
           - 清理风扇和散热片灰尘
           - 检查环境温度是否超标
           - 确认散热系统工作正常
        """,
        "metadata": {"source": "技术排查指南", "category": "troubleshooting", "product": "ProductA"}
    }
]

print(f"✓ 已准备 {len(sample_documents)} 个示例文档")
print(f"  - 产品手册: 技术规格")
print(f"  - FAQ: 客户常见问题")
print(f"  - 政策文件: 退换货政策")
print(f"  - 技术支持: 问题排查指南")


# ============================================================
# 第三部分：误区1 - 分割策略不当
# ============================================================
print("\n" + "=" * 70)
print("【误区1】分割策略不当 - chunk size选择的重要性")
print("=" * 70)

print("""
常见错误做法：
❌ 使用固定过大的chunk size（如2000+字符）
   - 问题：包含过多无关信息，稀释关键知识点
   - 问题：超出LLM上下文窗口的有效信息密度

❌ 使用固定过小的chunk size（如100字符）
   - 问题：丢失完整语义 context
   - 问题：检索时缺乏足够上下文

正确做法：
✓ 根据内容类型选择chunk size
✓ 保持语义完整性（句子、段落级别分割）
✓ 设置适当的overlap保持上下文连贯
✓ 重要内容单独成块
""")

# 演示：错误的分割方式
print("\n--- 演示：chunk_size过大导致的的问题 ---")
wrong_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # 过大的chunk size
    chunk_overlap=0,   # 无overlap
    separators=["\n", "。", "！", "？"]
)

# 演示：正确的分割方式
print("\n--- 演示：正确分割策略 ---")
correct_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # 适中的chunk size
    chunk_overlap=100, # 保持上下文连贯
    separators=["\n\n", "\n", "。", "！", "？"],  # 按语义层级分割
    add_start_index=True
)

# 将示例文档转换为LangChain文档格式
from langchain_core.documents import Document

# 转换文档格式
langchain_docs = [
    Document(page_content=doc["content"], metadata=doc["metadata"])
    for doc in sample_documents
]

# 使用正确的分割策略
correct_splits = correct_text_splitter.split_documents(langchain_docs)

print(f"\n✓ 使用正确策略分割后得到 {len(correct_splits)} 个文本块")
for i, split in enumerate(correct_splits[:3]):
    print(f"\n  文本块 {i+1}:")
    print(f"  内容长度: {len(split.page_content)} 字符")
    print(f"  来源: {split.metadata.get('source', '未知')}")
    # 只显示前80个字符
    preview = split.page_content[:80].replace('\n', ' ')
    print(f"  预览: {preview}...")


# ============================================================
# 第四部分：误区2 - 检索策略单一
# ============================================================
print("\n" + "=" * 70)
print("【误区2】检索策略单一 - 只用向量检索")
print("=" * 70)

print("""
常见错误做法：
❌ 只使用向量检索（语义相似度）
   - 问题：对精确术语匹配效果差
   - 问题：同义词、专业缩写可能导致漏检
   - 问题：专业领域词汇向量化效果不稳定

❌ 只使用关键词检索（BM25）
   - 问题：无法理解语义
   - 问题：用户用不同表达方式提问可能找不到

正确做法：
✓ 使用混合检索策略（向量 + BM25 + 重排序）
✓ 向量检索处理语义匹配
✓ BM25处理精确术语匹配
✓ Reranker进行精排优化
""")

# 初始化嵌入模型
print("\n--- 演示：混合检索实现 ---")
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device': 'cpu'}
)

# 构建向量数据库
vectorstore = FAISS.from_documents(correct_splits, embeddings)
print("✓ 向量数据库构建成功")

# 模拟用户查询
user_query_1 = "产品A的CPU是什么型号？"
user_query_2 = "Server A supports which operating systems?"

print(f"\n用户问题1（中文专业术语）: {user_query_1}")
print(f"用户问题2（英文专业术语）: {user_query_2}")

# 执行向量检索
results_1 = vectorstore.similarity_search_with_score(user_query_1, k=3)
results_2 = vectorstore.similarity_search_with_score(user_query_2, k=3)

print("\n--- 向量检索结果 ---")
print(f"\n问题1 '{user_query_1}' 的检索结果:")
for i, (doc, score) in enumerate(results_1):
    print(f"  {i+1}. [分数:{score:.4f}] {doc.metadata.get('source', '未知')}: {doc.page_content[:50]}...")

print(f"\n问题2 '{user_query_2}' 的检索结果:")
for i, (doc, score) in enumerate(results_2):
    print(f"  {i+1}. [分数:{score:.4f}] {doc.metadata.get('source', '未知')}: {doc.page_content[:50]}...")


# ============================================================
# 第五部分：误区3 - 提示词工程不当
# ============================================================
print("\n" + "=" * 70)
print("【误区3】提示词工程不当 - 缺乏上下文构建")
print("=" * 70)

print("""
常见错误做法：
❌ 直接将检索结果拼接，不做结构化处理
   - 问题：上下文混乱，LLM难以理解
   - 问题：关键信息被稀释

❌ 提示词过于简单，缺乏约束
   - 问题：LLM可能产生幻觉
   - 问题：回答格式不规范

❌ 不指定引用来源
   - 问题：无法验证答案准确性
   - 问题：用户无法追溯原文

正确做法：
✓ 在提示词中明确包含【角色】【任务】【约束】【格式】
✓ 指导LLM基于检索内容回答，而非自身知识
✓ 要求LLM引用具体来源（文档名称、页码等）
✓ 对不确定的内容明确说"未检索到"
""")

# 构建改进的提示词模板
print("\n--- 演示：结构化提示词模板 ---")

# 错误的简单提示词
BAD_PROMPT = """根据以下内容回答问题：
{context}

问题：{question}
回答："""

# 正确的结构化提示词
GOOD_PROMPT = """你是一位专业的产品技术支持工程师。你的职责是基于检索到的知识库内容，
准确、专业地回答客户问题。

【重要规则】
1. 只基于以下检索到的内容回答，不要编造信息
2. 如果检索内容中没有明确答案，必须明确说明"根据现有资料无法确定"
3. 回答时必须引用信息来源，格式：[来源: xxx]
4. 保持回答专业、清晰、易懂

【检索到的知识库内容】
{context}

【客户问题】
{question}

【回答要求】
1. 首先给出直接答案
2. 然后提供详细解释
3. 如有相关注意事项请一并说明
4. 引用具体的文档来源"""

print("错误提示词示例（过于简单）:")
print(BAD_PROMPT[:100] + "...")

print("\n正确提示词示例（结构化、有约束）:")
print(GOOD_PROMPT[:200] + "...")

# 模拟使用提示词
test_context = """
产品A技术规格：
- 处理器：Intel Xeon Gold 6248R (24核3.0GHz)
[来源: 产品A技术手册]
"""
test_question = "产品A用什么CPU？"

print("\n--- 使用提示词模板的示例 ---")
prompt_template = ChatPromptTemplate.from_template(GOOD_PROMPT)
formatted_prompt = prompt_template.format(
    context=test_context,
    question=test_question
)
print("格式化后的提示词预览:")
print(formatted_prompt[:300] + "...")


# ============================================================
# 第六部分：误区4 - 忽视元数据过滤
# ============================================================
print("\n" + "=" * 70)
print("【误区4】忽视元数据过滤 - 降低检索精度")
print("=" * 70)

print("""
常见错误做法：
❌ 所有文档一股脑存入向量库
   - 问题：检索结果混杂，无法精准定位
   - 问题：相似话题的文档相互干扰

❌ 不使用元数据进行预过滤
   - 问题：在无关类别中搜索，浪费计算资源
   - 问题：Top-K结果可能不是真正想要的

正确做法：
✓ 文档入库时丰富元数据（来源、类别、产品、时间等）
✓ 检索前先根据元数据预过滤
✓ 结合向量相似度和元数据筛选
✓ 支持多维度组合查询
""")

# 演示：带元数据的向量检索
print("\n--- 演示：基于元数据的过滤检索 ---")

# 为向量库添加描述符（用于元数据过滤）
# 重新构建带元数据的向量库
doc_with_ids = []
for i, doc in enumerate(correct_splits):
    # 创建唯一ID
    doc.id_ = f"doc_{i}"

# 演示：按产品类别过滤的场景
# 场景：用户只想了解产品A的信息，但向量检索可能返回所有产品

print("模拟场景：用户搜索'保修期'")
print("  - 未使用元数据过滤：可能返回所有产品的保修信息")
print("  - 使用元数据过滤（product=ProductA）：只返回ProductA的保修信息")

# 模拟演示（实际代码中需要向量库支持过滤）
demo_metadata = {
    "doc_1": {"product": "ProductA", "category": "product_manual"},
    "doc_2": {"product": "ProductA", "category": "faq"},
    "doc_3": {"product": "all", "category": "policy"},
    "doc_4": {"product": "ProductA", "category": "troubleshooting"}
}

print("\n模拟检索结果：")
print("  未过滤前：返回4个文档（包含其他产品信息）")
print("  过滤后（product=ProductA）：返回3个文档（更精准）")

for doc_id, meta in demo_metadata.items():
    product = meta.get("product", "unknown")
    category = meta.get("category", "unknown")
    marker = "✓" if product == "ProductA" else "✗"
    print(f"  {marker} {doc_id}: product={product}, category={category}")


# ============================================================
# 第七部分：误区5 - 缺乏评估与迭代
# ============================================================
print("\n" + "=" * 70)
print("【误区5】缺乏评估与迭代 - RAG系统持续优化")
print("=" * 70)

print("""
常见错误做法：
❌ 上线后就认为系统完成
   - 问题：无法发现实际使用中的问题
   - 问题：检索质量无法量化评估

❌ 凭主观感觉判断系统好坏
   - 问题：用户反馈主观性强，难定位问题
   - 问题：无法对比优化前后的效果差异

正确做法：
✓ 建立RAG评估指标体系
  - 检索评估：Precision@K, Recall@K, MRR, NDCG
  - 生成评估：Answer Accuracy, Relevance, Fluency
✓ 定期收集Bad Case
✓ A/B测试不同配置（分割策略、检索器、提示词）
✓ 监控关键指标持续优化
""")

# 演示：简单的RAG系统评估框架
print("\n--- 演示：RAG评估指标框架 ---")

class RAGEvaluator:
    """
    RAG系统评估器

    用于评估RAG系统的检索质量和生成质量
    """

    def __init__(self):
        self.retrieval_metrics = {
            "precision_at_3": 0.0,
            "recall_at_3": 0.0,
            "mrr": 0.0
        }
        self.generation_metrics = {
            "answer_relevance": 0.0,
            "context_utilization": 0.0,
            "hallucination_rate": 0.0
        }
        self.test_cases = []

    def add_test_case(self, query, expected_docs, actual_docs):
        """
        添加测试用例

        参数:
            query: 用户查询
            expected_docs: 期望检索到的文档ID列表
            actual_docs: 实际检索到的文档ID列表
        """
        self.test_cases.append({
            "query": query,
            "expected": expected_docs,
            "actual": actual_docs
        })

    def evaluate_retrieval(self):
        """
        评估检索质量

        使用Precision@K, Recall@K, MRR等指标
        """
        if not self.test_cases:
            return {"error": "No test cases"}

        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0

        for tc in self.test_cases:
            expected = set(tc["expected"])
            actual = tc["actual"][:3]  # Top-3

            # Precision@K
            relevant = len(set(actual) & expected)
            precision = relevant / len(actual) if actual else 0
            total_precision += precision

            # Recall@K
            recall = relevant / len(expected) if expected else 0
            total_recall += recall

            # MRR (Mean Reciprocal Rank)
            for i, doc_id in enumerate(actual):
                if doc_id in expected:
                    total_mrr += 1.0 / (i + 1)
                    break

        n = len(self.test_cases)
        return {
            "precision_at_3": total_precision / n,
            "recall_at_3": total_recall / n,
            "mrr": total_mrr / n
        }

    def print_evaluation_report(self):
        """打印评估报告"""
        print("\n" + "-" * 50)
        print("RAG系统评估报告")
        print("-" * 50)

        retrieval_metrics = self.evaluate_retrieval()
        print(f"\n检索质量指标:")
        print(f"  Precision@3: {retrieval_metrics.get('precision_at_3', 0):.2%}")
        print(f"  Recall@3: {retrieval_metrics.get('recall_at_3', 0):.2%}")
        print(f"  MRR: {retrieval_metrics.get('mrr', 0):.2f}")

        print(f"\n测试用例数量: {len(self.test_cases)}")
        print("-" * 50)


# 使用评估器
evaluator = RAGEvaluator()

# 模拟测试用例：产品保修期查询
evaluator.add_test_case(
    query="产品A的保修期是多久？",
    expected_docs=["doc_2", "doc_3"],  # FAQ和退换货政策
    actual_docs=["doc_2", "doc_3", "doc_1"]  # 实际检索到doc_2排第一
)

# 模拟测试用例：技术支持联系
evaluator.add_test_case(
    query="怎么联系技术支持？",
    expected_docs=["doc_2"],  # FAQ中包含联系方式
    actual_docs=["doc_2", "doc_4", "doc_1"]  # FAQ排第一
)

# 打印评估报告
evaluator.print_evaluation_report()


# ============================================================
# 第八部分：最佳实践总结
# ============================================================
print("\n" + "=" * 70)
print("【最佳实践总结】企业级RAG系统设计规范")
print("=" * 70)

best_practices = """
┌─────────────────────────────────────────────────────────────────────┐
│                    企业级RAG系统最佳实践清单                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  一、文档处理                                                        │
│  ├─ 根据内容类型选择分割策略                                          │
│  ├─ 保持语义完整性（句子/段落级别）                                   │
│  ├─ chunk_size 建议300-800字符                                       │
│  └─ 设置chunk_overlap保持上下文（10-20%）                             │
│                                                                      │
│  二、检索策略                                                        │
│  ├─ 采用混合检索（向量+BM25+Reranker）                                │
│  ├─ 利用元数据进行预过滤和后过滤                                      │
│  ├─ 设置合理的Top-K值（召回与精度平衡）                              │
│  └─ 定期评估和调整检索策略                                            │
│                                                                      │
│  三、提示词工程                                                      │
│  ├─ 明确角色、任务、约束和格式                                       │
│  ├─ 要求引用信息来源                                                 │
│  ├─ 指导LLM基于检索内容回答                                          │
│  └─ 对不确定内容明确说明                                              │
│                                                                      │
│  四、系统评估                                                        │
│  ├─ 建立评估指标体系（Precision, Recall, MRR）                       │
│  ├─ 收集Bad Case持续优化                                            │
│  ├─ A/B测试不同配置                                                 │
│  └─ 监控关键业务指标                                                 │
│                                                                      │
│  五、工程实践                                                        │
│  ├─ 丰富文档元数据                                                   │
│  ├─ 支持增量更新知识库                                               │
│  ├─ 做好日志记录和监控                                               │
│  └─ 考虑成本和性能平衡                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
"""
print(best_practices)


# ============================================================
# 第九部分：完整RAG流程演示（整合最佳实践）
# ============================================================
print("\n" + "=" * 70)
print("【完整RAG流程】整合最佳实践的示例")
print("=" * 70)

print("""
以下展示一个整合了所有最佳实践的完整RAG问答流程：
""")

# 1. 加载文档
print("1. 文档加载...")
print("   - 支持多种格式（PDF、Word、HTML等）")
print("   - 保留文档元数据")

# 2. 智能分割
print("\n2. 文档分割（最佳实践）...")
print("   - 按语义层级分割（段落 > 句子）")
print("   - chunk_size=500, overlap=100")
print("   - 保持上下文连贯性")

# 3. 向量化存储
print("\n3. 向量存储...")
print("   - 使用高质量Embedding模型（BGE）")
print("   - 存入FAISS向量数据库")
print("   - 保留元数据用于后续过滤")

# 4. 检索（混合检索）
print("\n4. 检索（混合检索 + 重排序）...")
print("   - 向量检索：处理语义相似性")
print("   - BM25检索：处理精确术语匹配")
print("   - Reranker：精排优化结果")
print("   - 元数据过滤：精准定位目标文档")

# 5. 答案生成（结构化提示词）
print("\n5. 答案生成（结构化提示词）...")
print("   - 明确角色：技术支持工程师")
print("   - 明确约束：基于检索内容回答")
print("   - 要求引用来源")
print("   - 规范回答格式")

# 演示一个完整的问答
print("\n" + "-" * 50)
print("【完整问答演示】")
print("-" * 50)

# 用户问题
demo_query = "产品A的保修期是多久？出现问题怎么联系技术支持？"
print(f"用户问题: {demo_query}")

# 模拟检索结果
demo_results = [
    {
        "source": "客户支持FAQ",
        "content": "Q1: 产品A的保修期是多久？A: 产品A标准保修期为3年，可付费延长至5年。",
        "relevance": 0.95
    },
    {
        "source": "客户支持FAQ",
        "content": "Q2: 如何申请技术支持？A: 电话：400-123-4567，邮箱：support@company.com",
        "relevance": 0.90
    },
    {
        "source": "退换货政策",
        "content": "三年有限保修：涵盖硬件故障，保修期内的维修服务免费",
        "relevance": 0.85
    }
]

print("\n检索到的相关文档:")
for i, result in enumerate(demo_results):
    print(f"  {i+1}. [{result['source']}] (相关度:{result['relevance']:.0%})")
    print(f"     {result['content'][:60]}...")

# 模拟生成的回答
demo_answer = """
根据检索到的资料，针对您的问题回答如下：

【产品A保修期】
产品A标准保修期为3年，可付费延长至5年。保修期内涵盖硬件故障，维修服务免费。

【联系方式】
- 电话：400-123-4567
- 邮箱：support@company.com
- 官网：www.company.com/support

[来源: 客户支持FAQ, 退换货政策]
"""

print("\n模型生成的回答:")
print(demo_answer)

print("\n" + "=" * 70)
print("【代码演示结束】")
print("=" * 70)
print("""
学习要点回顾：
1. 分割策略要平衡语义完整性和信息密度
2. 混合检索 + 重排序是提升检索质量的关键
3. 提示词需要结构化、规范化、有约束
4. 元数据过滤能显著提升检索精度
5. 持续评估和迭代是RAG系统优化的核心

更多内容请参考：
- part_04: 文档分割策略
- part_09: 混合检索与重排序
- part_10: RAG质量评估体系
""")
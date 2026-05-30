"""
文档分割策略示例代码

本文件演示了LlamaIndex中常用的几种文档分割器：
1. SentenceSplitter - 基于句子的基础分割
2. TokenTextSplitter - 基于token数量的精确分割
3. SemanticSplitterNodeParser - 基于语义的智能分割
4. SentenceWindowNodeParser - 句子窗口分割

使用方法：
    python text_splitter.py
"""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.settings import Settings

# =============================================================================
# 测试用文档
# =============================================================================

# 示例文本：一篇关于文档分割技术的介绍文章
test_text = """
文档分割是RAG系统中至关重要的环节。好的分割策略可以让检索更加精准。
本文将介绍几种常用的文档分割方法。

第一部分是基于字符的分割方法。SentenceSplitter按照自然句子边界进行分割，
它会在句号、问号、感叹号等句子结束符处切分，同时考虑换行符和段落分隔。
这种方法的优点是简单直接，能够保证每个句子都被完整保留。

第二部分是TokenTextSplitter。这种方法按照预设的token数量来切分文档。
它特别适合那些对输入长度有严格限制的模型，比如某些嵌入模型要求输入不超过512个token。
通过精确控制每个chunk的token数量，可以避免超出模型的处理能力。

第三部分是SemanticSplitterNodeParser。这是一种智能的语义分割方法。
它使用嵌入模型计算句子之间的语义相似度，当发现两个相邻句子的相似度突然下降时，
就在此处设置断点。这种方法能够保证每个chunk内的语义都是连贯的。

第四部分是SentenceWindowNodeParser。它的核心思想是将检索单元和上下文窗口分离。
在索引时，每个句子独立编码；检索时，找到相关句子后再用周围的句子作为上下文补充。
这种方法在检索精度和上下文完整性之间取得了很好的平衡。

文档分割的选择应该基于具体的业务场景。简单的应用可以使用SentenceSplitter；
有token限制的场景可以使用TokenTextSplitter；对精度要求高的专业文档
（法律、医疗、学术）建议使用SemanticSplitterNodeParser。
"""

# 创建Document对象
document = Document(text=test_text)
print("=" * 80)
print("文档分割示例")
print("=" * 80)
print(f"\n原始文档长度: {len(test_text)} 字符\n")


# =============================================================================
# 1. SentenceSplitter - 基于句子的基础分割
# =============================================================================

print("-" * 80)
print("1. SentenceSplitter - 基于句子的基础分割")
print("-" * 80)

# 初始化SentenceSplitter分割器
# chunk_size: 每个chunk的目标字符数
# chunk_overlap: 相邻chunk之间的重叠字符数
sentence_splitter = SentenceSplitter(
    chunk_size=100,       # 每个chunk大约100个字符
    chunk_overlap=20,     # 相邻chunk之间重叠20个字符，保持上下文连贯
    separator="\n"        # 优先按换行符分割
)

# 对文档进行分割，返回Node列表
nodes_from_sentence = sentence_splitter.get_nodes_from_documents([document])

print(f"\n分割后的节点数量: {len(nodes_from_sentence)}")
print("\n前3个节点的内容:")
for i, node in enumerate(nodes_from_sentence[:3]):
    print(f"\n--- Node {i} ---")
    print(node.text)


# =============================================================================
# 2. TokenTextSplitter - 基于token数量的精确分割
# =============================================================================

print("\n" + "=" * 80)
print("2. TokenTextSplitter - 基于token数量的精确分割")
print("=" * 80)

# 初始化TokenTextSplitter分割器
# 这种分割器按照token数量来切分，适合有严格token限制的场景
token_splitter = TokenTextSplitter(
    chunk_size=100,       # 每个chunk目标100个token
    chunk_overlap=20,      # 重叠20个token
    separator=" "         # 使用空格作为分隔符
)

# 使用split_text方法直接分割文本字符串
nodes_from_tokens = token_splitter.split_text(test_text)

print(f"\n分割后的文本块数量: {len(nodes_from_tokens)}")
print("\n前2个文本块:")
for i, text in enumerate(nodes_from_tokens[:2]):
    print(f"\n--- Chunk {i} ---")
    print(text)
    print(f"字符数: {len(text)}")


# =============================================================================
# 3. SemanticSplitterNodeParser - 基于语义的智能分割
# =============================================================================

print("\n" + "=" * 80)
print("3. SemanticSplitterNodeParser - 基于语义的智能分割")
print("=" * 80)

# 注意：SemanticSplitterNodeParser需要嵌入模型来计算语义相似度
# 如果没有配置嵌入模型，这部分代码可能会出错

# 定义一个简单的中文句子分割函数（用于处理中文文本）
def split_chinese_sentences(text):
    """
    简单的中文句子分割函数
    按照句号、问号、感叹号和换行符来分割句子
    """
    import re
    # 分割句子：包括中文句号、问号、感叹号和英文句号
    sentences = re.split(r'([。！？；\n]|\.(?=\s|$))', text)
    # 合并句子和分隔符
    merged = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]
        if sentence.strip():
            merged.append(sentence.strip())
    return merged

try:
    # 初始化语义分割器
    # buffer_size: 用于计算相似度的句子缓冲数量
    # breakpoint_percentile_threshold: 相似度阈值，低于此值则视为断点
    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=2,                           # 将2个句子组成一组计算相似度
        breakpoint_percentile_threshold=80,     # 相似度阈值80%
        embed_model=Settings.embed_model,       # 使用配置的嵌入模型
        sentence_splitter=split_chinese_sentences, # 自定义中文分割器
        include_metadata=True,                   # 是否包含元数据
        include_prev_next_rel=True               # 是否包含前后节点关系
    )

    # 对文档进行语义分割
    nodes_from_semantic = semantic_splitter.get_nodes_from_documents([document])

    print(f"\n语义分割后的节点数量: {len(nodes_from_semantic)}")
    print("\n分割结果:")
    for i, node in enumerate(nodes_from_semantic):
        print(f"\n--- Chunk {i} ---")
        print(f"内容: {node.text[:80]}..." if len(node.text) > 80 else f"内容: {node.text}")
        print(f"元数据keys: {list(node.metadata.keys())}")

except Exception as e:
    print(f"\n语义分割需要配置嵌入模型，当前跳过: {e}")
    print("如需使用语义分割，请确保配置了有效的embed_model")


# =============================================================================
# 4. SentenceWindowNodeParser - 句子窗口分割
# =============================================================================

print("\n" + "=" * 80)
print("4. SentenceWindowNodeParser - 句子窗口分割")
print("=" * 80)

# 初始化句子窗口分割器
# window_size: 每个句子前后捕获的句子数量
# window_metadata_key: 存储窗口上下文的元数据键名
# original_text_metadata_key: 存储原始句子的元数据键名
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=1,                          # 每个句子前后各捕获1个句子作为上下文
    window_metadata_key="window",           # 元数据中存储窗口上下文的键名
    original_text_metadata_key="original_sentence",  # 存储原始句子的键名
    include_metadata=True                   # 是否包含元数据
)

# 创建用于演示的短文本（SentenceWindowNodeParser更适合处理完整句子）
demo_text = """
深度学习是机器学习的分支。
它受人脑结构启发。
神经网络是深度学习的基础。
卷积神经网络在图像识别中表现出色。
循环神经网络适合处理序列数据。
注意力机制大幅提升了模型性能。
Transformer架构改变了深度学习格局。
"""

# 创建Document对象
demo_document = Document(text=demo_text)

# 使用句子窗口分割器进行分割
nodes_from_window = node_parser.get_nodes_from_documents([demo_document])

print(f"\n句子窗口分割后的节点数量: {len(nodes_from_window)}")

# 打印每个节点的详细信息
print("\n各节点的详细信息:")
for i, node in enumerate(nodes_from_window):
    print(f"\n--- Node {i} ---")
    print(f"节点文本（句子）: {node.text}")
    print(f"窗口上下文: {node.metadata.get('window', 'N/A')}")
    print(f"原始句子: {node.metadata.get('original_sentence', 'N/A')}")
    # 查看节点关系
    if hasattr(node, 'relationships') and node.relationships:
        print(f"节点关系: {node.relationships}")


# =============================================================================
# 5. 分割策略对比总结
# =============================================================================

print("\n" + "=" * 80)
print("5. 分割策略对比总结")
print("=" * 80)

summary = """
+---------------------------+--------------------------------+---------------------------+
| 分割器                    | 分割依据                       | 适用场景                  |
+---------------------------+--------------------------------+---------------------------+
| SentenceSplitter          | 句子边界、段落分隔符           | 通用文本，保持句子完整    |
| TokenTextSplitter         | 精确的token数量                | 有严格token限制的模型     |
| SemanticSplitterNodeParser| 句子间的语义相似度             | 长文档、学术论文、法律合同|
| SentenceWindowNodeParser  | 单句+周围上下文窗口            | 需要精确检索的场景        |
+---------------------------+--------------------------------+---------------------------+

选择建议：
- 通用场景：SentenceSplitter
- 有token限制：TokenTextSplitter
- 高精度需求：SemanticSplitterNodeParser
- 精确检索+上下文：SentenceWindowNodeParser
"""
print(summary)


# =============================================================================
# 主函数入口
# =============================================================================

if __name__ == "__main__":
    """
    运行本文件将依次演示四种不同的文档分割方法：
    1. SentenceSplitter - 基于句子的基础分割
    2. TokenTextSplitter - 基于token的精确分割
    3. SemanticSplitterNodeParser - 基于语义的智能分割（需要嵌入模型）
    4. SentenceWindowNodeParser - 句子窗口分割
    """
    print("\n" + "=" * 80)
    print("文档分割演示完成！")
    print("=" * 80)
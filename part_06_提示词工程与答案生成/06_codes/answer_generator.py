"""
第08节 答案生成与引用溯源 - 代码案例

本文件演示RAG系统中答案生成和引用溯源的完整流程，包括：
1. 提示词模板的设计与构建
2. 答案生成器（AnswerGenerator）的实现
3. 引用溯源机制的实现
4. QueryEngine响应合成模式演示

核心概念：
- 提示增强：将检索到的文档与用户问题组合成增强提示词
- 引用溯源：在答案中标记信息来源，实现可追溯性
"""

# ==============================================================================
# 第一部分：依赖导入
# ==============================================================================

# 标准库
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# 可选依赖（实际使用时安装）
# pip install llama-index openai

# ==============================================================================
# 第二部分：数据结构定义
# ==============================================================================

@dataclass
class TextChunk:
    """
    文本块数据结构

    用于存储检索到的文档片段及其元信息，是RAG系统的基本数据单元。

    属性说明：
    - content: 文本内容
    - chunk_id: 块唯一标识符，用于引用追踪
    - source: 来源文档名称
    - metadata: 附加元数据（如页码、章节等）
    """
    content: str
    chunk_id: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """
    引用信息数据结构

    记录答案中每个声明的来源，用于实现答案的可溯源性。

    属性说明：
    - chunk_id: 引用的文本块ID
    - source: 来源文档名称
    - text_preview: 引用文本的预览（便于用户理解）
    - relevance_score: 相关性评分（可选）
    """
    chunk_id: str
    source: str
    text_preview: str
    relevance_score: Optional[float] = None


@dataclass
class GeneratedAnswer:
    """
    生成答案的数据结构

    包含最终答案内容及其引用信息，用于返回给用户。

    属性说明：
    - answer: 生成的文本答案
    - citations: 引用列表
    - model_used: 使用的模型名称
    - generation_time: 生成耗时（秒）
    """
    answer: str
    citations: List[Citation]
    model_used: str
    generation_time: float


class ResponseMode(Enum):
    """
    响应合成模式枚举

    定义QueryEngine的不同响应合成策略：
    - COMPACT: 压缩模式，将多个检索结果合并后一次性生成
    - REFINE: 精炼模式，逐步迭代优化答案
    - COMPACT_REFINE: 先压缩再精炼，平衡效率和质量
    - TREE_SUMMARIZE: 树形总结，分层理解后生成
    """
    COMPACT = "compact"
    REFINE = "refine"
    COMPACT_REFINE = "compact_refine"
    TREE_SUMMARIZE = "tree_summarize"


# ==============================================================================
# 第三部分：提示词模板构建
# ==============================================================================

class PromptTemplate:
    """
    提示词模板类

    负责构建用于答案生成的提示词模板，支持自定义系统角色、
    上下文格式、输出格式等。

    模板变量：
    - {context}: 检索到的文档上下文
    - {question}: 用户问题
    - {history}: 对话历史（可选）
    """

    # 默认系统提示词，设定AI的专业能力和行为准则
    DEFAULT_SYSTEM_PROMPT = """你是一位专业的技术文档助手。你的职责是：
1. 根据提供的上下文信息，准确回答用户的问题
2. 在回答中明确标注信息来源，确保答案可溯源
3. 如果上下文中没有足够信息回答问题，请明确说明
4. 回答要简洁、清晰、结构化"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        context_template: Optional[str] = None,
        question_template: Optional[str] = None,
        citation_instruction: Optional[str] = None
    ):
        """
        初始化提示词模板

        参数说明：
        - system_prompt: 系统角色提示，定义AI的专业能力范围
        - context_template: 上下文的格式模板，定义文档如何呈现
        - question_template: 问题的格式模板，定义问题如何呈现
        - citation_instruction: 引用标注指令，告诉模型如何标注来源
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # 上下文模板：定义检索到的文档如何呈现在提示词中
        # 使用编号列表格式，便于模型理解和引用
        self.context_template = context_template or (
            "【上下文信息】\n"
            "{context}\n"
            "【上下文结束】"
        )

        # 问题模板：定义用户问题如何呈现
        self.question_template = question_template or (
            "【用户问题】\n"
            "{question}"
        )

        # 引用标注指令：明确要求模型在答案中标注来源
        self.citation_instruction = citation_instruction or (
            "【重要】请在回答中使用编号标注引用来源，"
            "格式如：[1]、[2]等，并确保每个重要声明都有对应的引用。"
        )

    def build(
        self,
        context_chunks: List[TextChunk],
        question: str,
        include_citation_instruction: bool = True
    ) -> str:
        """
        构建完整的提示词

        参数说明：
        - context_chunks: 检索到的文本块列表
        - question: 用户问题
        - include_citation_instruction: 是否包含引用标注指令

        返回值：
        - 构建好的提示词字符串
        """
        # Step 1: 构建上下文部分
        # 将多个文本块用编号和分隔线组合，便于模型理解每个块的结构
        context_parts = []
        for idx, chunk in enumerate(context_chunks, start=1):
            context_parts.append(
                f"[{idx}] 来源：{chunk.source} | ID：{chunk.chunk_id}\n"
                f"内容：{chunk.content}"
            )
        context_str = "\n---\n".join(context_parts)

        # Step 2: 使用上下文模板格式化
        formatted_context = self.context_template.format(context=context_str)

        # Step 3: 使用问题模板格式化
        formatted_question = self.question_template.format(question=question)

        # Step 4: 组装完整提示词
        parts = [
            self.system_prompt,
            formatted_context,
            formatted_question
        ]

        # Step 5: 如果需要，添加引用标注指令
        if include_citation_instruction:
            parts.append(self.citation_instruction)

        return "\n\n".join(parts)

    def build_for_streaming(self, question: str) -> str:
        """
        构建流式输出的系统提示词（用于流式响应场景）

        参数说明：
        - question: 用户问题

        返回值：
        - 简化的提示词字符串
        """
        return (
            f"{self.system_prompt}\n\n"
            f"用户问题：{question}\n\n"
            f"请直接开始回答，确保引用来源。"
        )


# ==============================================================================
# 第四部分：答案生成器实现
# ==============================================================================

class AnswerGenerator:
    """
    答案生成器类

    核心组件，负责：
    1. 将检索结果和用户问题组装成提示词
    2. 调用LLM生成答案
    3. 解析答案中的引用信息
    4. 返回带有引用溯源的最终答案

    使用流程：
    generator = AnswerGenerator(api_key="your-key", model="gpt-4")
    answer = generator.generate(question="...", retrieved_chunks=[...])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        """
        初始化答案生成器

        参数说明：
        - api_key: OpenAI API密钥
        - model: 使用的模型名称，默认gpt-4
        - temperature: 温度参数，控制随机性（0-1），越低越确定性
        - max_tokens: 最大生成token数

        注意：temperature=0.3在RAG场景中是平衡准确性和多样性的好选择
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = PromptTemplate()

    def generate(
        self,
        question: str,
        retrieved_chunks: List[TextChunk],
        response_mode: ResponseMode = ResponseMode.COMPACT,
        verbose: bool = False
    ) -> GeneratedAnswer:
        """
        生成带引用的答案

        参数说明：
        - question: 用户问题
        - retrieved_chunks: 检索到的文本块列表
        - response_mode: 响应合成模式
        - verbose: 是否打印详细信息

        返回值：
        - GeneratedAnswer对象，包含答案和引用信息
        """
        import time
        start_time = time.time()

        # Step 1: 打印检索结果（调试用）
        if verbose:
            print(f"[AnswerGenerator] 检索到 {len(retrieved_chunks)} 个相关文档")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"  Chunk {i}: {chunk.source} - {chunk.chunk_id}")

        # Step 2: 构建提示词
        prompt = self.prompt_template.build(
            context_chunks=retrieved_chunks,
            question=question,
            include_citation_instruction=True
        )

        if verbose:
            print(f"[AnswerGenerator] 提示词构建完成，长度: {len(prompt)} 字符")

        # Step 3: 调用LLM生成答案（这里使用模拟，实际需要API调用）
        answer_text = self._call_llm(prompt, response_mode)

        # Step 4: 解析答案中的引用
        citations = self._parse_citations(answer_text, retrieved_chunks)

        # Step 5: 清理答案中的引用格式（如果需要）
        # answer_text = self._clean_citation_markers(answer_text)

        generation_time = time.time() - start_time

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            model_used=self.model,
            generation_time=generation_time
        )

    def _call_llm(
        self,
        prompt: str,
        response_mode: ResponseMode
    ) -> str:
        """
        调用LLM生成答案

        参数说明：
        - prompt: 构建好的提示词
        - response_mode: 响应合成模式

        返回值：
        - 生成的答案文本

        注意：这是一个简化实现，实际使用时需要调用OpenAI API
        """
        # ==========================================================================
        # 实际API调用示例（需要取消注释并填入真实API Key）
        # ==========================================================================
        # from openai import OpenAI
        # client = OpenAI(api_key=self.api_key)
        # response = client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": "你是一位专业的技术文档助手..."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens
        # )
        # return response.choices[0].message.content
        # ==========================================================================

        # 模拟返回：在实际环境中替换为真实API调用
        # 这里返回示例答案，展示引用格式
        return (
            "根据检索到的文档信息，RAG系统主要由三个核心模块组成：\n"
            "检索模块负责从向量数据库中找到相关文档 [1]。\n"
            "生成模块使用LLM基于上下文生成答案 [2]。\n"
            "引用溯源机制确保答案的每个声明都有对应的来源 [1][2]。\n\n"
            "这种架构设计使得RAG系统能够在保证答案质量的同时，"
            "提供可验证的答案来源。"
        )

    def _parse_citations(
        self,
        answer_text: str,
        chunks: List[TextChunk]
    ) -> List[Citation]:
        """
        解析答案中的引用信息

        参数说明：
        - answer_text: 生成的答案文本
        - chunks: 原始检索块列表

        返回值：
        - 解析出的引用列表

        工作原理：
        1. 使用正则表达式查找答案中的引用标记（如[1]、[2]）
        2. 建立引用索引到原始chunk的映射
        3. 生成Citation对象列表
        """
        import re

        # 查找所有引用标记 [数字] 格式
        # 正则解释：[(\d+)] 匹配方括号中的数字
        citation_pattern = r'\[(\d+)\]'
        found_numbers = re.findall(citation_pattern, answer_text)

        citations = []
        seen_ids = set()  # 用于去重

        for num_str in found_numbers:
            num = int(num_str)
            # 引用编号从1开始，数组索引从0开始
            if 1 <= num <= len(chunks):
                chunk = chunks[num - 1]
                chunk_id = chunk.chunk_id

                # 避免重复添加同一个引用
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    citations.append(Citation(
                        chunk_id=chunk_id,
                        source=chunk.source,
                        text_preview=chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                        relevance_score=None  # 实际应用中可以从检索阶段传递
                    ))

        return citations


# ==============================================================================
# 第五部分：引用溯源工具函数
# ==============================================================================

def format_citations_for_display(citations: List[Citation]) -> str:
    """
    格式化引用列表用于显示

    参数说明：
    - citations: Citation对象列表

    返回值：
    - 格式化的字符串，适合在答案后展示
    """
    if not citations:
        return "（无引用）"

    lines = ["【参考来源】"]
    for idx, citation in enumerate(citations, start=1):
        lines.append(
            f"[{idx}] {citation.source}\n"
            f"    ID: {citation.chunk_id}\n"
            f"    摘要: {citation.text_preview}"
        )

    return "\n".join(lines)


def verify_answer_quality(
    answer: str,
    question: str,
    citations: List[Citation]
) -> Dict[str, Any]:
    """
    验证答案质量的辅助函数

    参数说明：
    - answer: 生成的答案
    - question: 用户问题
    - citations: 引用列表

    返回值：
    - 包含各项质量指标的字典

    检查项：
    - 答案长度是否合理
    - 是否包含引用标注
    - 引用数量是否足够
    - 是否回应了问题
    """
    quality_report = {
        "answer_length": len(answer),
        "has_citations": len(citations) > 0,
        "citation_count": len(citations),
        "references_all_cited": True,  # 简化检查
        "quality_score": 0.0  # 综合评分
    }

    # 简化评分逻辑
    score = 0.0
    if 50 <= quality_report["answer_length"] <= 2000:
        score += 0.3
    if quality_report["has_citations"]:
        score += 0.4
    score += min(0.3, quality_report["citation_count"] * 0.1)

    quality_report["quality_score"] = min(score, 1.0)

    return quality_report


# ==============================================================================
# 第六部分：QueryEngine响应合成模式演示
# ==============================================================================

class QueryEngineSimulator:
    """
    QueryEngine模拟器

    演示LlamaIndex QueryEngine的响应合成模式在实际中的应用。
    这些模式决定了如何处理多个检索结果并生成最终答案。

    主要模式：
    - Compact: 压缩合并后生成
    - Refine: 迭代精炼
    - Compact-Refine: 先压缩再精炼
    - Tree-Summarize: 分层总结
    """

    def __init__(self, answer_generator: AnswerGenerator):
        """
        初始化QueryEngine模拟器

        参数说明：
        - answer_generator: 答案生成器实例，用于实际生成答案
        """
        self.generator = answer_generator

    def query_compact(
        self,
        question: str,
        chunks: List[TextChunk]
    ) -> GeneratedAnswer:
        """
        Compact模式：压缩合并后生成

        原理：将多个检索结果的内容压缩合并到一个提示词中，
             然后一次性调用LLM生成答案。

        适用场景：检索结果数量较多（>5个），需要减少token消耗
        优点：效率高，API调用少
        缺点：可能丢失部分细节信息
        """
        # 压缩上下文：将多个chunk的内容合并
        compressed_context = self._compress_chunks(chunks)

        # 创建临时的TextChunk用于生成
        compressed_chunk = TextChunk(
            content=compressed_context,
            chunk_id="compressed",
            source="compressed_from_multiple_chunks",
            metadata={"original_chunk_count": len(chunks)}
        )

        return self.generator.generate(
            question=question,
            retrieved_chunks=[compressed_chunk],
            response_mode=ResponseMode.COMPACT
        )

    def query_refine(
        self,
        question: str,
        chunks: List[TextChunk]
    ) -> GeneratedAnswer:
        """
        Refine模式：迭代精炼

        原理：按顺序遍历检索结果，每次基于前一个答案和当前chunk
             优化答案，最终得到精炼的结果。

        适用场景：需要高质量、精细的答案
        优点：答案质量高，充分考虑每个chunk
        缺点：API调用多，耗时长
        """
        current_answer = ""

        for i, chunk in enumerate(chunks):
            # 第一轮：基于第一个chunk生成初始答案
            if i == 0:
                result = self.generator.generate(
                    question=question,
                    retrieved_chunks=[chunk],
                    response_mode=ResponseMode.REFINE
                )
                current_answer = result.answer
            else:
                # 后续轮次：将前一个答案作为上下文传入
                refined_context = TextChunk(
                    content=f"之前的答案：{current_answer}\n\n当前上下文：{chunk.content}",
                    chunk_id=f"refine_step_{i}",
                    source=chunk.source,
                    metadata={"refine_step": i}
                )
                result = self.generator.generate(
                    question=question,
                    retrieved_chunks=[refined_context],
                    response_mode=ResponseMode.REFINE
                )
                current_answer = result.answer

        return GeneratedAnswer(
            answer=current_answer,
            citations=self.generator._parse_citations(current_answer, chunks),
            model_used=self.generator.model,
            generation_time=0.0  # 实际应用中统计
        )

    def query_compact_refine(
        self,
        question: str,
        chunks: List[TextChunk]
    ) -> GeneratedAnswer:
        """
        Compact-Refine模式：先压缩再精炼

        原理：先使用压缩模式得到一个基础答案，
             然后使用refine模式迭代优化。

        适用场景：平衡效率和质量的场景（推荐生产使用）
        优点：兼顾效率和质量
        缺点：实现复杂度较高
        """
        # 第一阶段：Compact模式得到初始答案
        compact_result = self.query_compact(question, chunks[:3])  # 只用前3个
        current_answer = compact_result.answer

        # 第二阶段：Refine模式精炼
        remaining_chunks = chunks[3:]
        for chunk in remaining_chunks:
            refined_context = TextChunk(
                content=f"已有答案：{current_answer}\n\n补充信息：{chunk.content}",
                chunk_id=f"compact_refine_{chunk.chunk_id}",
                source=chunk.source
            )
            result = self.generator.generate(
                question=question,
                retrieved_chunks=[refined_context],
                response_mode=ResponseMode.COMPACT_REFINE
            )
            current_answer = result.answer

        return GeneratedAnswer(
            answer=current_answer,
            citations=self.generator._parse_citations(current_answer, chunks),
            model_used=self.generator.model,
            generation_time=0.0
        )

    def _compress_chunks(self, chunks: List[TextChunk], max_length: int = 3000) -> str:
        """
        压缩多个chunk的内容

        参数说明：
        - chunks: 原始chunk列表
        - max_length: 压缩后的最大长度

        返回值：
        - 压缩后的文本
        """
        total_content = "\n".join([chunk.content for chunk in chunks])
        if len(total_content) <= max_length:
            return total_content

        # 简单截断策略，实际应用中可以使用更智能的压缩
        return total_content[:max_length] + "\n...（内容已截断）"


# ==============================================================================
# 第七部分：完整RAG答案生成流程演示
# ==============================================================================

def demo_complete_rag_pipeline():
    """
    完整的RAG答案生成流程演示

    模拟真实场景：
    1. 模拟检索得到相关文档块
    2. 使用AnswerGenerator生成答案
    3. 解析和展示引用信息
    """
    print("=" * 70)
    print("RAG答案生成与引用溯源 - 完整流程演示")
    print("=" * 70)

    # Step 1: 准备模拟的检索结果
    # 在实际应用中，这些来自向量数据库的检索
    sample_chunks = [
        TextChunk(
            content="RAG（检索增强生成）是一种结合检索系统和生成模型的技术。"
                    "它通过从外部知识库检索相关文档来增强LLM的生成能力。",
            chunk_id="chunk_001",
            source="RAG技术白皮书",
            metadata={"page": 1, "section": "概述"}
        ),
        TextChunk(
            content="RAG系统的主要优势包括：1）减少幻觉生成；2）提供最新信息；"
                    "3）答案可溯源；4）保护隐私数据。",
            chunk_id="chunk_002",
            source="RAG技术白皮书",
            metadata={"page": 2, "section": "优势"}
        ),
        TextChunk(
            content="典型的RAG流程包括：文档加载、文本分割、向量编码、"
                    "相似度检索、提示词增强、答案生成六个步骤。",
            chunk_id="chunk_003",
            source="RAG实战指南",
            metadata={"page": 15, "section": "流程"}
        ),
    ]

    # Step 2: 初始化答案生成器
    # 注意：实际使用时需要填入真实的API Key
    generator = AnswerGenerator(
        api_key="your-openai-api-key",  # 替换为真实API Key
        model="gpt-4",
        temperature=0.3
    )

    # Step 3: 用户问题
    question = "RAG系统有哪些主要优势？"

    print(f"\n【用户问题】\n{question}")
    print(f"\n【检索结果】找到 {len(sample_chunks)} 个相关文档")

    # Step 4: 生成答案
    print("\n正在生成答案...")
    answer = generator.generate(
        question=question,
        retrieved_chunks=sample_chunks,
        response_mode=ResponseMode.COMPACT,
        verbose=True
    )

    # Step 5: 展示答案
    print("\n" + "=" * 70)
    print("【生成的答案】")
    print("=" * 70)
    print(answer.answer)

    # Step 6: 展示引用信息
    print("\n" + "=" * 70)
    print("【引用溯源】")
    print("=" * 70)
    print(format_citations_for_display(answer.citations))

    # Step 7: 质量验证
    print("\n" + "=" * 70)
    print("【答案质量报告】")
    print("=" * 70)
    quality = verify_answer_quality(answer.answer, question, answer.citations)
    for key, value in quality.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("流程演示完成！")
    print("=" * 70)


def demo_query_engine_modes():
    """
    QueryEngine响应合成模式演示

    展示不同响应模式下的答案生成差异
    """
    print("\n" + "=" * 70)
    print("QueryEngine 响应合成模式演示")
    print("=" * 70)

    # 准备测试数据
    test_chunks = [
        TextChunk(content=f"这是第{i+1}个文档块的内容，包含了一些关键信息。",
                  chunk_id=f"test_{i}", source="测试文档")
        for i in range(5)
    ]

    question = "请总结文档的主要内容"

    # 初始化生成器和QueryEngine模拟器
    generator = AnswerGenerator(api_key="your-key", model="gpt-4")
    query_engine = QueryEngineSimulator(generator)

    # 演示Compact模式
    print("\n--- Compact模式 ---")
    result = query_engine.query_compact(question, test_chunks)
    print(f"答案长度: {len(result.answer)} 字符")
    print(f"引用数量: {len(result.citations)}")

    # 演示Refine模式
    print("\n--- Refine模式 ---")
    result = query_engine.query_refine(question, test_chunks)
    print(f"答案长度: {len(result.answer)} 字符")
    print(f"引用数量: {len(result.citations)}")

    # 演示Compact-Refine模式
    print("\n--- Compact-Refine模式 ---")
    result = query_engine.query_compact_refine(question, test_chunks)
    print(f"答案长度: {len(result.answer)} 字符")
    print(f"引用数量: {len(result.citations)}")

    print("\n模式演示完成！")


# ==============================================================================
# 第八部分：主程序入口
# ==============================================================================

if __name__ == "__main__":
    """
    程序入口

    运行本文件将执行两个演示：
    1. 完整的RAG答案生成流程
    2. QueryEngine响应合成模式对比
    """
    # 演示完整流程
    demo_complete_rag_pipeline()

    # 演示QueryEngine模式
    demo_query_engine_modes()
# -*- coding: utf-8 -*-
"""
第14节 RAG 智能体架构与工具调用

本模块实现了一个完整的 RAG 智能体系统，包含：
- 工具基类（BaseTool）和常用工具实现
- 记忆管理模块（Memory）
- 路由模块（Router）
- ReAct 执行器（ReActExecutor）
- RAG 智能体主类（RAGAgent）

作者: RAG Learning Course
版本: 1.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from enum import Enum
import time


# =============================================================================
# 第一部分：工具（Tool）相关类
# =============================================================================

class BaseTool(ABC):
    """
    工具基类

    所有工具都需要继承此类并实现 execute 方法。
    工具是 Agent 执行动作的具体手段，每个工具负责一项具体任务。

    属性:
        name: str - 工具的唯一名称
        description: str - 工具的功能描述，用于 Agent 理解何时使用该工具
    """

    def __init__(self, name: str, description: str):
        """
        初始化工具

        Args:
            name: 工具名称
            description: 工具功能描述
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        执行工具的核心方法

        Args:
            **kwargs: 工具执行所需的参数

        Returns:
            Any: 工具执行的结果
        """
        raise NotImplementedError("子类必须实现 execute 方法")

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


class VectorSearchTool(BaseTool):
    """
    向量检索工具

    该工具封装了向量数据库的检索功能，可以在知识库中搜索相关内容。
    用于 RAG 智能体中的知识检索环节。

    属性:
        name: str - 固定为 "vector_search"
        description: str - 固定为 "在知识库中搜索相关信息"
    """

    def __init__(self, knowledge_base: Dict[str, List[str]]):
        """
        初始化向量检索工具

        Args:
            knowledge_base: 知识库字典，键为文档ID，值为文档内容列表
        """
        super().__init__(
            name="vector_search",
            description="在知识库中搜索相关信息"
        )
        # 知识库数据
        self.knowledge_base = knowledge_base

    def execute(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        执行向量检索

        该方法模拟向量检索过程。在实际应用中，这里会调用向量数据库
        （如 ChromaDB、Milvus 等）进行语义检索。

        Args:
            query: 检索查询语句
            top_k: 返回的最大结果数量，默认为 3

        Returns:
            Dict[str, Any]: 包含检索结果的字典
                - documents: 相关文档列表
                - scores: 相似度分数列表
                - count: 检索到的文档数量
        """
        results = []
        scores = []

        # 简单的关键词匹配模拟向量检索
        # 实际应用中应使用 embedding 模型进行语义匹配
        query_words = set(query.lower().split())

        for doc_id, content_list in self.knowledge_base.items():
            for content in content_list:
                # 计算简单的词重叠分数
                content_words = set(content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = overlap / len(query_words)
                    results.append({
                        "doc_id": doc_id,
                        "content": content,
                        "score": score
                    })
                    scores.append(score)

        # 按分数排序并返回 top_k 个结果
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        return {
            "documents": [r["content"] for r in top_results],
            "scores": [r["score"] for r in top_results],
            "count": len(top_results)
        }


class CalculatorTool(BaseTool):
    """
    计算工具

    该工具用于执行数学计算，是 Agent 调用外部能力的示例。
    在实际应用中，可以替换为 API 调用、数据库查询等工具。

    属性:
        name: str - 固定为 "calculator"
        description: str - 固定为 "执行数学计算"
    """

    def __init__(self):
        """
        初始化计算工具
        """
        super().__init__(
            name="calculator",
            description="执行数学计算"
        )

    def execute(self, expression: str) -> Dict[str, Any]:
        """
        执行数学表达式计算

        Args:
            expression: 数学表达式字符串，如 "2 + 3 * 4"

        Returns:
            Dict[str, Any]: 包含计算结果的字典
                - result: 计算结果（如果表达式有效）
                - error: 错误信息（如果表达式无效）
                - expression: 原表达式
        """
        try:
            # 使用 eval 进行简单计算（注意：实际应用中应使用更安全的计算方法）
            result = eval(expression)
            return {
                "result": result,
                "expression": expression,
                "success": True
            }
        except Exception as e:
            return {
                "result": None,
                "expression": expression,
                "error": str(e),
                "success": False
            }


class AnswerGeneratorTool(BaseTool):
    """
    答案生成工具

    该工具基于检索到的文档内容生成最终回答。
    模拟 LLM 生成答案的过程。

    属性:
        name: str - 固定为 "answer_generator"
        description: str - 固定为 "基于检索结果生成回答"
    """

    def __init__(self):
        """
        初始化答案生成工具
        """
        super().__init__(
            name="answer_generator",
            description="基于检索结果生成回答"
        )

    def execute(self, context: List[str], question: str) -> Dict[str, Any]:
        """
        生成答案

        Args:
            context: 检索到的上下文文档列表
            question: 用户问题

        Returns:
            Dict[str, Any]: 包含生成答案的字典
                - answer: 生成的答案
                - sources: 引用的文档来源
                - has_context: 是否有有效的上下文
        """
        if not context:
            return {
                "answer": "抱歉，我没有找到与您问题相关的知识库内容。",
                "sources": [],
                "has_context": False
            }

        # 简单拼接生成答案（实际应用中应调用 LLM API）
        context_str = "\n\n".join([f"[文档{i+1}] {doc}" for i, doc in enumerate(context)])
        answer = f"根据检索到的信息，我为您解答如下：\n\n{context_str}\n\n基于以上内容，关于「{question}」的回答如上。"

        return {
            "answer": answer,
            "sources": [f"文档{i+1}" for i in range(len(context))],
            "has_context": True
        }


# =============================================================================
# 第二部分：记忆（Memory）相关类
# =============================================================================

@dataclass
class Message:
    """
    消息数据类

    用于存储对话中的单条消息。

    属性:
        role: str - 消息角色，如 "user"、"assistant"、"system"
        content: str - 消息内容
        timestamp: float - 时间戳
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCall:
    """
    工具调用记录数据类

    用于记录 Agent 执行工具调用的详细信息。

    属性:
        tool_name: str - 被调用的工具名称
        parameters: Dict - 调用时传入的参数
        result: Any - 工具执行的结果
        timestamp: float - 调用时间戳
    """
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: float = field(default_factory=time.time)


class Memory:
    """
    记忆管理类

    负责管理对话历史和中间结果，支持多轮对话和工具调用记录。
    是 RAG 智能体的重要组成部分，使智能体能够在多轮交互中保持上下文。

    主要功能:
        - 保存用户和助手的对话历史
        - 记录工具调用历史
        - 维护中间推理结果
    """

    def __init__(self, max_history: int = 100):
        """
        初始化记忆模块

        Args:
            max_history: 最多保存的历史消息数量
        """
        self.messages: List[Message] = []  # 对话历史
        self.tool_calls: List[ToolCall] = []  # 工具调用历史
        self.intermediate_results: Dict[str, Any] = {}  # 中间结果
        self.max_history = max_history

    def add_message(self, role: str, content: str) -> None:
        """
        添加一条对话消息

        Args:
            role: 消息角色，如 "user"、"assistant"、"system"
            content: 消息内容
        """
        message = Message(role=role, content=content)
        self.messages.append(message)

        # 如果超过最大历史，删除最早的消息
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def add_tool_call(self, tool_name: str, parameters: Dict[str, Any], result: Any) -> None:
        """
        记录一次工具调用

        Args:
            tool_name: 被调用的工具名称
            parameters: 调用参数
            result: 调用结果
        """
        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            result=result
        )
        self.tool_calls.append(tool_call)

    def set_intermediate_result(self, key: str, value: Any) -> None:
        """
        保存中间推理结果

        Args:
            key: 结果的键名
            value: 结果值
        """
        self.intermediate_results[key] = value

    def get_intermediate_result(self, key: str) -> Optional[Any]:
        """
        获取中间推理结果

        Args:
            key: 结果的键名

        Returns:
            Optional[Any]: 结果值，如果不存在则返回 None
        """
        return self.intermediate_results.get(key)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史

        Returns:
            List[Dict[str, str]]: 对话历史列表，每个元素包含 role 和 content
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def get_recent_messages(self, n: int = 5) -> List[Dict[str, str]]:
        """
        获取最近 n 条消息

        Args:
            n: 返回的消息数量

        Returns:
            List[Dict[str, str]]: 最近的消息列表
        """
        recent = self.messages[-n:] if len(self.messages) >= n else self.messages
        return [{"role": msg.role, "content": msg.content} for msg in recent]

    def get_tool_call_history(self) -> List[ToolCall]:
        """
        获取工具调用历史

        Returns:
            List[ToolCall]: 工具调用记录列表
        """
        return self.tool_calls

    def clear(self) -> None:
        """
        清空所有记忆内容
        """
        self.messages.clear()
        self.tool_calls.clear()
        self.intermediate_results.clear()


# =============================================================================
# 第三部分：路由（Router）相关类
# =============================================================================

class QueryType(Enum):
    """
    查询类型枚举

    用于分类用户查询的类型，决定如何处理该查询。
    """
    # 需要检索知识库的问题
    KNOWLEDGE_QUERY = "knowledge_query"
    # 可以直接回答的问题（如问候、闲聊）
    GENERAL_QUERY = "general_query"
    # 需要执行计算的问题
    CALCULATION_QUERY = "calculation_query"
    # 多跳复杂问题，需要多次检索
    COMPLEX_QUERY = "complex_query"


class Router:
    """
    路由模块

    负责理解用户意图，对查询进行分类，并决定是否需要检索以及使用哪个知识库。

    主要功能:
        - 判断查询类型
        - 决定是否需要检索
        - 选择合适的处理策略
    """

    def __init__(self, knowledge_bases: Dict[str, Dict[str, List[str]]]):
        """
        初始化路由模块

        Args:
            knowledge_bases: 知识库字典，键为知识库名称，值为知识库内容
        """
        self.knowledge_bases = knowledge_bases

    def classify_query(self, query: str) -> QueryType:
        """
        对用户查询进行分类

        Args:
            query: 用户查询语句

        Returns:
            QueryType: 查询类型
        """
        query_lower = query.lower()

        # 简单的规则分类（实际应用中可使用 LLM 或分类模型）
        calculation_keywords = ["计算", "加", "减", "乘", "除", "+", "-", "*", "/", "等于", "多少"]
        knowledge_keywords = ["什么", "如何", "为什么", "怎样", "哪个", "哪些", "定义", "原理"]
        complex_keywords = ["首先", "然后", "最后", "步骤", "流程", "原因和结果"]

        # 检查是否需要计算
        if any(kw in query_lower for kw in calculation_keywords):
            return QueryType.CALCULATION_QUERY

        # 检查是否是多跳复杂问题
        if any(kw in query_lower for kw in complex_keywords):
            return QueryType.COMPLEX_QUERY

        # 检查是否需要知识检索
        if any(kw in query_lower for kw in knowledge_keywords):
            return QueryType.KNOWLEDGE_QUERY

        # 默认为通用查询
        return QueryType.GENERAL_QUERY

    def select_knowledge_base(self, query: str) -> Optional[str]:
        """
        选择适合当前查询的知识库

        Args:
            query: 用户查询语句

        Returns:
            Optional[str]: 选中的知识库名称，如果没有合适的选择则返回 None
        """
        query_lower = query.lower()

        # 简单的知识库选择逻辑
        # 实际应用中可以根据查询的语义相似度选择知识库
        if "技术" in query or "代码" in query or "编程" in query:
            return "technical_kb"
        elif "产品" in query or "规格" in query:
            return "product_kb"
        elif "政策" in query or "规定" in query:
            return "policy_kb"

        # 默认返回第一个知识库
        if self.knowledge_bases:
            return list(self.knowledge_bases.keys())[0]

        return None

    def should_retrieve(self, query: str) -> bool:
        """
        判断是否需要进行检索

        Args:
            query: 用户查询语句

        Returns:
            bool: 是否需要检索
        """
        query_type = self.classify_query(query)
        return query_type in [QueryType.KNOWLEDGE_QUERY, QueryType.COMPLEX_QUERY]


# =============================================================================
# 第四部分：ReAct 执行器
# =============================================================================

@dataclass
class Thought:
    """
    思考步骤数据类

    用于记录 ReAct 推理过程中的单个思考步骤。

    属性:
        step: int - 步骤编号
        thought: str - 思考内容
        action: str - 采取的动作
        action_input: Dict - 动作输入参数
        observation: Any - 观察结果
    """
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Any = None


class ReActExecutor:
    """
    ReAct 执行器

    实现 ReAct（Reasoning + Acting）范式，通过思考-行动-观察的循环
    让 Agent 能够自主决策并执行动作。

    工作流程:
        1. Thought: 分析当前状态，产生推理
        2. Action: 根据推理决定执行哪个动作
        3. Observation: 观察动作结果
        4. 重复直到得到最终答案

    属性:
        tools: Dict[str, BaseTool] - 可用工具字典
        max_iterations: int - 最大迭代次数，防止无限循环
    """

    def __init__(self, tools: Dict[str, BaseTool], max_iterations: int = 10):
        """
        初始化 ReAct 执行器

        Args:
            tools: 可用工具字典，键为工具名称，值为工具实例
            max_iterations: 最大迭代次数，默认 10
        """
        self.tools = tools
        self.max_iterations = max_iterations
        self.thought_history: List[Thought] = []

    def execute(self, question: str, memory: Memory) -> Dict[str, Any]:
        """
        执行 ReAct 推理循环

        Args:
            question: 用户问题
            memory: 记忆模块实例

        Returns:
            Dict[str, Any]: 执行结果，包含最终答案和推理过程
        """
        self.thought_history = []
        current_context = []

        for iteration in range(self.max_iterations):
            # ----- Thought 阶段 -----
            thought_content = self._think(question, current_context, iteration)

            # ----- Action 阶段 -----
            action_result = self._act(question, current_context, thought_content)

            if action_result is None:
                # 无法决定动作，结束推理
                break

            action, action_input, observation = action_result
            current_context.append(observation)

            # 记录思考步骤
            thought = Thought(
                step=iteration + 1,
                thought=thought_content,
                action=action,
                action_input=action_input,
                observation=observation
            )
            self.thought_history.append(thought)

            # 如果动作是生成答案，结束循环
            if action == "answer_generator":
                break

        # 返回最终结果
        final_thought = self.thought_history[-1] if self.thought_history else None
        return {
            "answer": final_thought.observation.get("answer", "无法生成答案") if final_thought else "无结果",
            "thought_history": self.thought_history,
            "iterations": len(self.thought_history)
        }

    def _think(self, question: str, context: List[Any], iteration: int) -> str:
        """
        思考阶段：分析当前状态，决定下一步行动

        Args:
            question: 用户问题
            context: 当前上下文（之前的观察结果）
            iteration: 当前迭代次数

        Returns:
            str: 思考内容
        """
        # 根据是否有上下文和迭代次数产生思考
        if iteration == 0:
            # 第一次迭代，判断是否需要检索
            if context:
                return f"我已经有了一些上下文，现在需要基于这些内容生成答案。"
            else:
                return f"用户问题是「{question}」，我需要先检索知识库获取相关信息。"

        # 后续迭代，检查检索结果是否足够
        if len(context) > 0:
            return f"我已经检索到 {len(context)} 条相关内容，评估后决定是否需要继续检索。"
        else:
            return f"继续检索以获取更多信息。"

    def _act(self, question: str, context: List[Any], thought: str) -> Optional[tuple]:
        """
        行动阶段：根据思考决定并执行动作

        Args:
            question: 用户问题
            context: 当前上下文
            thought: 思考内容

        Returns:
            Optional[tuple]: (动作名, 动作输入, 观察结果) 元组，如果无法决定动作则返回 None
        """
        # 决定使用哪个工具
        if "检索" in thought or "继续检索" in thought:
            # 使用向量检索工具
            tool = self.tools.get("vector_search")
            if tool:
                action_input = {"query": question, "top_k": 3}
                observation = tool.execute(**action_input)
                return ("vector_search", action_input, observation)

        # 评估是否需要生成答案
        if len(context) >= 2 or "生成答案" in thought:
            tool = self.tools.get("answer_generator")
            if tool:
                # 将之前的观察结果作为上下文
                docs = []
                for obs in context:
                    if isinstance(obs, dict) and "documents" in obs:
                        docs.extend(obs["documents"])
                action_input = {"context": docs, "question": question}
                observation = tool.execute(**action_input)
                return ("answer_generator", action_input, observation)

        # 默认返回第一个可用工具
        if self.tools:
            tool_name = list(self.tools.keys())[0]
            tool = self.tools[tool_name]
            action_input = {"query": question}
            observation = tool.execute(**action_input)
            return (tool_name, action_input, observation)

        return None


# =============================================================================
# 第五部分：RAG 智能体主类
# =============================================================================

class RAGAgent:
    """
    RAG 智能体主类

    整合所有组件，构建完整的 RAG 智能体系统。
    支持自主决策、工具调用和记忆管理。

    主要功能:
        - 理解用户意图
        - 自主决策是否需要检索
        - 调用多种工具
        - 维护对话历史
        - 生成最终答案
    """

    def __init__(
        self,
        knowledge_bases: Dict[str, Dict[str, List[str]]],
        llm_provider: Optional[Callable] = None
    ):
        """
        初始化 RAG 智能体

        Args:
            knowledge_bases: 知识库字典
            llm_provider: LLM 提供者（可选，用于增强推理能力）
        """
        # 初始化组件
        self.router = Router(knowledge_bases)
        self.memory = Memory()

        # 初始化工具
        self.tools: Dict[str, BaseTool] = {}

        # 注册知识库检索工具
        for kb_name, kb_content in knowledge_bases.items():
            self.tools["vector_search"] = VectorSearchTool(kb_content)

        # 注册其他工具
        self.tools["calculator"] = CalculatorTool()
        self.tools["answer_generator"] = AnswerGeneratorTool()

        # 初始化 ReAct 执行器
        self.executor = ReActExecutor(tools=self.tools)

        # LLM 提供者（可选）
        self.llm_provider = llm_provider

    def add_tool(self, tool: BaseTool) -> None:
        """
        注册新工具

        Args:
            tool: 工具实例
        """
        self.tools[tool.name] = tool

    def process(self, question: str) -> Dict[str, Any]:
        """
        处理用户问题

        完整的工作流程：
        1. 理解意图（Router）
        2. 决定是否检索
        3. 执行 ReAct 循环
        4. 返回答案

        Args:
            question: 用户问题

        Returns:
            Dict[str, Any]: 处理结果
        """
        # 保存用户问题到记忆
        self.memory.add_message("user", question)

        # ----- 路由阶段 -----
        query_type = self.router.classify_query(question)
        should_retrieve = self.router.should_retrieve(question)

        # ----- ReAct 执行阶段 -----
        result = self.executor.execute(question, self.memory)

        # 保存助手回答到记忆
        self.memory.add_message("assistant", result["answer"])

        # 添加工具调用记录
        for thought in result.get("thought_history", []):
            self.memory.add_tool_call(
                tool_name=thought.action,
                parameters=thought.action_input,
                result=thought.observation
            )

        return {
            "question": question,
            "query_type": query_type.value,
            "should_retrieve": should_retrieve,
            "answer": result["answer"],
            "thought_history": [
                {
                    "step": t.step,
                    "thought": t.thought,
                    "action": t.action,
                    "observation": t.observation
                }
                for t in result.get("thought_history", [])
            ],
            "iterations": result.get("iterations", 0)
        }

    def clear_memory(self) -> None:
        """
        清空记忆

        用于开始新的对话会话。
        """
        self.memory.clear()


# =============================================================================
# 第六部分：演示代码
# =============================================================================

def demo():
    """
    RAG 智能体演示函数

    展示如何创建和使用 RAG 智能体。
    """
    print("=" * 70)
    print("RAG 智能体演示")
    print("=" * 70)

    # ----- 准备知识库数据 -----
    knowledge_bases = {
        "technical_kb": {
            "doc1": [
                "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。",
                "RAG 可以让大语言模型基于外部知识库回答问题，减少幻觉。",
                "ReAct 是一种推理框架，让智能体能够思考和行动相结合。"
            ],
            "doc2": [
                "向量数据库用于存储和检索文本的向量表示。",
                "常见的向量数据库包括 ChromaDB、Milvus、Pinecone 等。",
                "向量检索通过计算余弦相似度找到最相关的内容。"
            ],
            "doc3": [
                "Agent（智能体）是一种能够自主决策和执行动作的系统。",
                "智能体由感知、决策、执行三部分组成。",
                "工具调用是智能体扩展能力的重要方式。"
            ]
        }
    }

    # ----- 创建 RAG 智能体 -----
    print("\n[1] 初始化 RAG 智能体...")
    agent = RAGAgent(knowledge_bases=knowledge_bases)
    print(f"    - 已注册工具: {list(agent.tools.keys())}")
    print(f"    - 知识库数量: {len(knowledge_bases)}")

    # ----- 测试用例 1：知识查询 -----
    print("\n" + "-" * 70)
    print("[2] 测试知识查询")
    print("-" * 70)

    question1 = "什么是 RAG？它的工作原理是什么？"
    print(f"\n用户问题: {question1}")

    result1 = agent.process(question1)
    print(f"\n查询类型: {result1['query_type']}")
    print(f"需要检索: {result1['should_retrieve']}")
    print(f"\nReAct 推理过程:")
    for thought in result1["thought_history"]:
        print(f"  步骤 {thought['step']}:")
        print(f"    - 思考: {thought['thought']}")
        print(f"    - 动作: {thought['action']}")
        print(f"    - 观察: {thought['observation']}")
    print(f"\n最终回答:\n{result1['answer']}")

    # ----- 测试用例 2：多步查询 -----
    print("\n" + "-" * 70)
    print("[3] 测试复杂查询")
    print("-" * 70)

    question2 = "RAG 和 Agent 有什么关系？"
    print(f"\n用户问题: {question2}")

    result2 = agent.process(question2)
    print(f"\n查询类型: {result2['query_type']}")
    print(f"\nReAct 推理过程:")
    for thought in result2["thought_history"]:
        print(f"  步骤 {thought['step']}:")
        print(f"    - 思考: {thought['thought']}")
        print(f"    - 动作: {thought['action']}")
        print(f"    - 观察: {thought['observation']}")
    print(f"\n最终回答:\n{result2['answer']}")

    # ----- 查看记忆状态 -----
    print("\n" + "-" * 70)
    print("[4] 记忆状态")
    print("-" * 70)

    conversation_history = agent.memory.get_conversation_history()
    print(f"\n对话历史 ({len(conversation_history)} 条消息):")
    for msg in conversation_history:
        print(f"  - [{msg['role']}]: {msg['content'][:50]}...")

    tool_history = agent.memory.get_tool_call_history()
    print(f"\n工具调用历史 ({len(tool_history)} 次):")
    for tc in tool_history:
        print(f"  - {tc.tool_name} at {time.ctime(tc.timestamp)}")

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    # 运行演示
    demo()
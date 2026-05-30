# -*- coding: utf-8 -*-
"""
第07节 提示词工程核心原理 - 代码案例

本文件演示RAG系统中提示词工程的核心原理，包括：
1. 三种提示词模板的定义（系统提示词、用户提示词、上下文提示词）
2. 增强提示词的组装函数
3. LlamaIndex QueryEngine集成演示
4. 企业级RAG提示词模板的完整实现

依赖安装：
    pip install llama-index llama-index-llms-openai openai python-dotenv

环境配置：
    在当前目录创建 .env 文件，配置 OPENAI_API_KEY 和 OPENAI_BASE_URL
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# ============================================================
# 第一部分：提示词相关的数据类定义
# ============================================================

@dataclass
class Document:
    """
    文档数据类，用于存储检索到的文档信息

    属性说明：
    - content: 文档的文本内容
    - title: 文档标题
    - source: 文档来源（文件名、URL等）
    - page_num: 文档页码（可选）
    - metadata: 其他元数据信息（可选）
    """
    content: str
    title: str
    source: str
    page_num: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UserQuery:
    """
    用户查询数据类，用于封装用户的问题信息

    属性说明：
    - question: 用户的原始问题
    - chat_history: 对话历史（可选，用于多轮对话）
    """
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None


# ============================================================
# 第二部分：提示词模板类定义
# ============================================================

class PromptTemplate:
    """
    提示词模板类，用于构建和管理提示词

    提示词模板通过占位符（如 {question}、{context}）来动态插入内容，
    使得同一套模板可以适用于不同的用户问题和文档内容。

    使用方式：
        template = PromptTemplate("你好，{name}！今天是{date}。")
        result = template.format(name="小明", date="2024-01-01")
    """

    def __init__(self, template: str):
        """
        初始化提示词模板

        参数：
            template: 模板字符串，包含占位符 {placeholder_name}
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """
        格式化模板，用传入的值替换占位符

        参数：
            **kwargs: 要替换的占位符及其对应的值

        返回：
            格式化后的字符串

        示例：
            template = PromptTemplate("你好，{name}！")
            template.format(name="小明")  # 返回 "你好，小明！"
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            # 如果有未填充的占位符，抛出明确的错误信息
            raise ValueError(f"提示词模板缺少必需参数: {e}")


class SystemPromptTemplate(PromptTemplate):
    """
    系统提示词模板类，继承自PromptTemplate

    系统提示词用于定义AI的角色、行为规范和回答风格。
    它是整个提示词的基础设定期望，通常放在最前面。

    设计要点：
    1. 明确AI的角色定位（如：技术支持工程师、产品顾问）
    2. 设定回答的范围和边界（如：只回答文档范围内的问题）
    3. 规范输出的格式（如：使用Markdown、标注来源）
    4. 强调知识运用（要求必须基于提供的上下文回答）
    """

    def __init__(self, role: str, domain: str, rules: List[str], output_format: str):
        """
        初始化系统提示词模板

        参数：
            role: AI的角色描述，如"技术支持工程师"
            domain: 专业领域，如"网络设备"
            rules: 行为规则列表
            output_format: 输出格式要求
        """
        self.role = role
        self.domain = domain
        self.rules = rules
        self.output_format = output_format

        # 构建系统提示词模板
        template = self._build_template()
        super().__init__(template)

    def _build_template(self) -> str:
        """
        构建系统提示词模板的内部方法

        模板结构：
        1. 角色设定：以【角色】开头定义AI身份
        2. 专业领域：明确AI的专业范围
        3. 指令要求：列出具体的行为规则
        4. 输出格式：定义回答的结构要求
        5. 安全边界：明确不能回答的问题类型
        """
        # 构建规则列表字符串
        rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(self.rules)])

        template = f"""【角色】你是{self.role}
【专业领域】{self.domain}

【指令要求】
{rules_text}

【输出格式】
{self.output_format}

【重要提醒】
- 必须基于提供的参考文档回答，不要依赖自身知识
- 如果文档中没有相关信息，明确告知用户
- 回答时引用具体文档内容，并标注来源"""

        return template


class ContextPromptTemplate(PromptTemplate):
    """
    上下文提示词模板类，用于将检索到的文档内容格式化为提示词的一部分

    上下文提示词是RAG系统的"资料库"，为LLM提供回答问题所需的参考信息。

    设计要点：
    1. 文档来源标注：每段文档都要标注来源，便于用户追溯
    2. 内容组织：使用层级结构（如 ### 文档1）组织多个文档
    3. 分隔清晰：不同文档之间用分隔符区分
    4. 长度控制：避免上下文过长导致信息迷失
    """

    def __init__(self):
        """初始化上下文提示词模板"""
        super().__init__(self._build_template())

    def _build_template(self) -> str:
        """
        构建上下文提示词模板

        模板结构：
        1. 标题：## 参考文档
        2. 文档列表：每个文档包含标题、来源、内容
        3. 分隔符：使用 --- 分隔不同文档
        """
        template = """【参考文档】
{documents}

---"""
        return template

    def format_documents(self, documents: List[Document]) -> str:
        """
        将文档列表格式化为上下文提示词

        参数：
            documents: Document对象列表，包含检索到的相关文档

        返回：
            格式化后的上下文提示词字符串

        处理逻辑：
        1. 遍历每个文档，构建 "### 文档N：标题\n来源：...\n内容：..." 格式
        2. 多个文档之间用分隔符 --- 分隔
        3. 如果没有文档，返回提示信息
        """
        if not documents:
            return "（暂无参考文档）"

        doc_parts = []
        for i, doc in enumerate(documents, 1):
            # 构建单个文档的格式字符串
            # 来源信息包含文档标题和来源，页码可选
            source_info = f"{doc.title}"
            if doc.page_num:
                source_info += f"（第{doc.page_num}页）"
            source_info += f"\n来源：{doc.source}"

            doc_part = f"""### 文档{i}：{source_info}

{doc.content}"""
            doc_parts.append(doc_part)

        # 使用分隔符连接多个文档
        documents_text = "\n\n---\n\n".join(doc_parts)
        return self.template.format(documents=documents_text)


class UserPromptTemplate(PromptTemplate):
    """
    用户提示词模板类，用于将用户问题格式化为提示词的一部分

    用户提示词是用户实际输入的问题，需要清晰、明确地表达用户的意图。

    设计要点：
    1. 问题聚焦：直接提出核心问题，不绕弯子
    2. 结构清晰：用【用户问题】标签标注问题位置
    3. 历史感知：如果有多轮对话，包含历史记录
    """

    def __init__(self):
        """初始化用户提示词模板"""
        super().__init__(self._build_template())

    def _build_template(self) -> str:
        """
        构建用户提示词模板

        模板结构：
        【用户问题】：具体问题内容
        【对话历史】：如有，列出之前的对话（可选）
        """
        template = """【用户问题】
{question}

【对话历史】
{chat_history}

请基于以上参考文档，回答用户的问题。"""
        return template

    def format_question(self, query: UserQuery) -> str:
        """
        将用户查询格式化为用户提示词

        参数：
            query: UserQuery对象，包含用户问题

        返回：
            格式化后的用户提示词字符串

        处理逻辑：
        1. 格式化问题部分
        2. 如果有对话历史，追加历史记录
        3. 如果没有对话历史，标记为"无"
        """
        question_text = query.question

        # 处理对话历史
        if query.chat_history:
            # 将对话历史格式化为可读文本
            history_parts = []
            for i, turn in enumerate(query.chat_history, 1):
                role = "用户" if turn.get("role") == "user" else "助理"
                content = turn.get("content", "")
                history_parts.append(f"第{i}轮 - {role}：{content}")
            chat_history_text = "\n".join(history_parts)
        else:
            chat_history_text = "无"

        return self.template.format(
            question=question_text,
            chat_history=chat_history_text
        )


# ============================================================
# 第三部分：增强提示词组装器
# ============================================================

class EnhancedPromptAssembler:
    """
    增强提示词组装器，将系统提示词、上下文提示词、用户提示词组装成完整的提示词

    这是RAG系统中提示增强的核心组件，负责：
    1. 组合三种提示词模板
    2. 管理模板之间的组合顺序
    3. 处理不同格式的输入
    """

    def __init__(
        self,
        system_template: SystemPromptTemplate,
        context_template: ContextPromptTemplate,
        user_template: UserPromptTemplate
    ):
        """
        初始化增强提示词组装器

        参数：
            system_template: 系统提示词模板
            context_template: 上下文提示词模板
            user_template: 用户提示词模板
        """
        self.system_template = system_template
        self.context_template = context_template
        self.user_template = user_template

    def assemble(
        self,
        query: UserQuery,
        documents: List[Document]
    ) -> str:
        """
        组装完整的增强提示词

        参数：
            query: 用户查询对象
            documents: 检索到的文档列表

        返回：
            完整的增强提示词字符串

        组装顺序：
        1. 系统提示词（定义基础角色和行为）
        2. 上下文提示词（提供参考资料）
        3. 用户提示词（呈现用户问题）

        这样的顺序确保LLM先了解自己的角色和约束，
        再看到参考资料，最后面对具体问题。
        """
        # 1. 获取系统提示词
        system_prompt = self.system_template.template

        # 2. 获取上下文提示词（将文档列表格式化）
        context_prompt = self.context_template.format_documents(documents)

        # 3. 获取用户提示词（将用户问题格式化）
        user_prompt = self.user_template.format_question(query)

        # 4. 按顺序组合三个部分
        # 使用双换行符分隔不同部分，形成清晰的层次结构
        enhanced_prompt = f"""{system_prompt}

{context_prompt}

{user_prompt}

【回答】："""

        return enhanced_prompt

    def assemble_with_separator(
        self,
        query: UserQuery,
        documents: List[Document],
        separator: str = "\n\n" + "="*50 + "\n\n"
    ) -> str:
        """
        使用自定义分隔符组装增强提示词（用于调试和可视化）

        参数：
            query: 用户查询对象
            documents: 检索到的文档列表
            separator: 各部分之间的分隔符

        返回：
            完整的增强提示词字符串
        """
        system_prompt = self.system_template.template
        context_prompt = self.context_template.format_documents(documents)
        user_prompt = self.user_template.format_question(query)

        parts = [
            "【系统提示词】",
            system_prompt,
            "【上下文提示词】",
            context_prompt,
            "【用户提示词】",
            user_prompt
        ]

        return separator.join(parts) + "\n\n【回答】："


# ============================================================
# 第四部分：企业级提示词模板工厂
# ============================================================

class EnterprisePromptFactory:
    """
    企业级提示词模板工厂，用于快速创建符合企业标准的提示词模板

    工厂模式封装了模板创建的细节，用户只需指定领域即可获得完整的提示词模板。

    使用方式：
        factory = EnterprisePromptFactory("网络设备")
        assembler = factory.create_assembler()
        prompt = assembler.assemble(query, documents)
    """

    # 预定义的角色和专业领域模板
    ROLE_TEMPLATES = {
        "技术支持": "专业技术支持工程师",
        "产品顾问": "专业产品顾问",
        "客服代表": "专业客服代表",
        "研发专家": "研发技术专家"
    }

    # 预定义的规则模板
    RULE_TEMPLATES = {
        "技术支持": [
            "必须基于提供的参考文档回答，不依赖自身知识",
            "如果文档中没有相关信息，明确告知用户",
            "回答时引用具体文档内容，并标注来源",
            "使用专业术语，确保回答准确无误",
            "涉及操作步骤时，列出清晰的编号步骤"
        ],
        "产品顾问": [
            "必须基于提供的参考文档回答，不依赖自身知识",
            "如果文档中没有相关信息，明确告知用户",
            "站在客户角度回答问题，提供专业建议",
            "回答要简洁明了，避免过多技术细节",
            "如需进一步确认，引导用户补充信息"
        ],
        "客服代表": [
            "必须基于提供的参考文档回答，不依赖自身知识",
            "如果文档中没有相关信息，明确告知用户",
            "使用友好、亲切的语言风格",
            "回答要耐心、细致，不要假设用户具有专业知识",
            "如遇到复杂问题，引导用户联系专业人员"
        ],
        "研发专家": [
            "必须基于提供的参考文档回答，不依赖自身知识",
            "如果文档中没有相关信息，明确告知用户",
            "使用严谨的技术语言，确保描述准确",
            "涉及代码时，提供完整的、可运行的示例",
            "深入分析问题，提供多种可能的解决方案"
        ]
    }

    # 预定义的输出格式模板
    OUTPUT_FORMAT_TEMPLATES = {
        "技术支持": """
1. 直接回答问题，开门见山
2. 重要信息用**加粗**标注
3. 步骤性内容用编号列表：
   1) 第一步
   2) 第二步
   3) 第三步
4. 引用文档内容时用引用格式（> 引用内容）
5. 回答末尾注明参考文档来源
6. 如有补充说明，用*斜体*标注""",
        "产品顾问": """
1. 直接回答问题，简洁明了
2. 根据问题类型调整回答深度：
   - 简单问题：一两句话直接回答
   - 复杂问题：用结构化方式详细说明
3. 使用要点列表组织内容
4. 如有产品对比，提供对比表格
5. 回答末尾注明参考文档来源""",
        "客服代表": """
1. 先表示理解和关心，再回答问题
2. 使用简单易懂的语言，避免专业术语
3. 回答要耐心、详细，必要时举例说明
4. 如需操作步骤，用清晰的分步骤说明
5. 回答末尾询问是否需要进一步帮助""",
        "研发专家": """
1. 直接回答问题，技术细节清晰
2. 提供完整的分析过程和推理逻辑
3. 涉及代码时：
   - 用代码块展示
   - 添加必要的注释
   - 解释关键逻辑
4. 提供多种解决方案时，比较各自的优缺点
5. 引用文档中的具体技术细节作为依据"""
    }

    def __init__(self, domain: str, role_type: str = "技术支持"):
        """
        初始化企业级提示词模板工厂

        参数：
            domain: 专业领域，如"网络设备"、"企业软件"
            role_type: 角色类型，支持"技术支持"、"产品顾问"、"客服代表"、"研发专家"
        """
        self.domain = domain
        self.role_type = role_type

        # 验证角色类型是否有效
        if role_type not in self.ROLE_TEMPLATES:
            raise ValueError(f"不支持的角色类型：{role_type}，可选值：{list(self.ROLE_TEMPLATES.keys())}")

    def create_assembler(self) -> EnhancedPromptAssembler:
        """
        创建增强提示词组装器

        返回：
            配置好的EnhancedPromptAssembler实例

        处理逻辑：
        1. 根据角色类型获取对应的角色名称
        2. 根据角色类型获取对应的规则模板
        3. 根据角色类型获取对应的输出格式模板
        4. 创建三个模板实例
        5. 创建并返回组装器
        """
        role = self.ROLE_TEMPLATES[self.role_type]
        rules = self.RULE_TEMPLATES[self.role_type]
        output_format = self.OUTPUT_FORMAT_TEMPLATES[self.role_type]

        # 创建三个模板
        system_template = SystemPromptTemplate(
            role=f"{role}（{self.domain}领域）",
            domain=self.domain,
            rules=rules,
            output_format=output_format
        )

        context_template = ContextPromptTemplate()
        user_template = UserPromptTemplate()

        # 创建组装器
        assembler = EnhancedPromptAssembler(
            system_template=system_template,
            context_template=context_template,
            user_template=user_template
        )

        return assembler


# ============================================================
# 第五部分：LlamaIndex QueryEngine集成演示
# ============================================================

def demo_llamaindex_query_engine():
    """
    LlamaIndex QueryEngine 集成演示函数

    本函数演示如何在LlamaIndex框架中使用自定义提示词模板构建QueryEngine。
    注意：本函数需要完整的LlamaIndex环境才能运行。

    QueryEngine工作流程：
    1. 用户输入问题
    2. QueryEngine调用检索器获取相关文档
    3. 将检索结果传入提示模板
    4. 组装成增强提示词
    5. 发送给LLM生成答案
    6. 返回最终回答
    """
    print("=" * 60)
    print("LlamaIndex QueryEngine 集成演示")
    print("=" * 60)

    # 检查环境变量是否配置
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[跳过] 未配置 OPENAI_API_KEY 环境变量，跳过LlamaIndex演示")
        print("请在项目根目录创建 .env 文件配置以下环境变量：")
        print("  OPENAI_API_KEY=your_api_key")
        print("  OPENAI_BASE_URL=https://api.openai.com/v1")
        return

    try:
        # 导入LlamaIndex相关模块
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.core import QueryEngine as LlamaQueryEngine
        from llama_index.core import get_response_synthesizer
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.llms.openai import OpenAI

        print("\n[成功] LlamaIndex 模块导入成功")
        print("\n--- LlamaIndex QueryEngine 工作流程 ---")
        print("""
QueryEngine 是 LlamaIndex 提供的端到端问答引擎，它整合了：
1. 检索器（Retriever）：根据问题检索相关文档
2. 提示模板（Prompt Template）：组合检索结果与问题
3. LLM（大语言模型）：基于增强提示生成答案

QueryEngine 内部流程：
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  用户问题   │ --> │   检索器    │ --> │  检索结果   │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   最终答案  │ <-- │     LLM     │ <-- │  增强提示词  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              ▲
                                        ┌─────────────┐
                                        │  提示模板   │
                                        └─────────────┘
""")

        print("\n--- 使用自定义提示词模板 ---")
        print("""
# 导入LlamaIndex的PromptTemplate
from llama_index.core import PromptTemplate

# 定义自定义提示模板，{context_str} 和 {query_str} 是LlamaIndex的保留占位符
custom_prompt = PromptTemplate(
    \"\"\"【角色】你是专业的网络设备技术支持工程师
【任务】基于参考文档回答用户问题
【要求】必须引用文档内容，标注来源

【参考文档】
{context_str}

【用户问题】：{query_str}

【回答】：\"\"\"
)

# 创建QueryEngine时指定提示模板
query_engine = index.as_query_engine(
    text_qa_template=custom_prompt,
    llm=llm
)

# 执行查询
response = query_engine.query("如何重置路由器密码？")
print(response)
""")

    except ImportError as e:
        print(f"\n[跳过] LlamaIndex 模块导入失败：{e}")
        print("请先安装依赖：pip install llama-index llama-index-llms-openai")


# ============================================================
# 第六部分：完整演示
# ============================================================

def main():
    """
    主函数：演示提示词工程的完整流程

    本函数展示：
    1. 如何使用EnterprisePromptFactory创建企业级提示词模板
    2. 如何组装增强提示词
    3. 查看组装后的完整提示词内容
    """
    print("=" * 60)
    print("第07节 提示词工程核心原理 - 代码演示")
    print("=" * 60)

    # ---------------------------------------------------------
    # 演示1：创建企业级提示词模板
    # ---------------------------------------------------------
    print("\n【演示1】创建企业级提示词模板")
    print("-" * 40)

    # 创建一个针对"网络设备"领域的"技术支持"角色提示词工厂
    factory = EnterprisePromptFactory(
        domain="网络设备",
        role_type="技术支持"
    )
    assembler = factory.create_assembler()
    print(f"已创建提示词模板工厂（领域：网络设备，角色：技术支持）")

    # ---------------------------------------------------------
    # 演示2：准备示例数据
    # ---------------------------------------------------------
    print("\n【演示2】准备示例数据")
    print("-" * 40)

    # 创建模拟的用户问题
    query = UserQuery(
        question="如何重置XX型号路由器的管理员密码？",
        chat_history=[
            {"role": "user", "content": "路由器无法登录了"},
            {"role": "assistant", "content": "请问是忘记了登录密码吗？"}
        ]
    )
    print(f"用户问题：{query.question}")
    print(f"对话历史：{len(query.chat_history)}轮")

    # 创建模拟的检索到的文档
    documents = [
        Document(
            content="重置管理员密码的标准流程：\n1. 在路由器背面找到reset小孔\n2. 使用回形针长按10秒\n3. 等待指示灯闪烁后松开\n4. 使用默认密码admin登录",
            title="路由器快速入门指南",
            source="router_guide.pdf",
            page_num=3
        ),
        Document(
            content="如遇密码遗忘，可通过以下方式重置：\n1. 硬件重置：在设备通电状态下，长按reset键15秒\n2. 恢复出厂设置后，所有配置将丢失\n3. 默认用户名和密码均为admin",
            title="产品技术手册",
            source="technical_manual.pdf",
            page_num=15
        )
    ]
    print(f"检索到 {len(documents)} 篇相关文档：")
    for i, doc in enumerate(documents, 1):
        print(f"  文档{i}：{doc.title}（{doc.source}）")

    # ---------------------------------------------------------
    # 演示3：组装增强提示词
    # ---------------------------------------------------------
    print("\n【演示3】组装增强提示词")
    print("-" * 40)

    # 组装增强提示词
    enhanced_prompt = assembler.assemble(query, documents)
    print("增强提示词已组装完成")
    print(f"提示词总长度：{len(enhanced_prompt)} 字符")

    # ---------------------------------------------------------
    # 演示4：查看带分隔符的提示词详情
    # ---------------------------------------------------------
    print("\n【演示4】查看提示词详细内容")
    print("-" * 40)

    # 使用带分隔符的版本，便于查看结构
    detailed_prompt = assembler.assemble_with_separator(
        query,
        documents,
        separator="\n\n" + "-" * 30 + "\n\n"
    )
    print(detailed_prompt)

    # ---------------------------------------------------------
    # 演示5：不同角色类型的提示词对比
    # ---------------------------------------------------------
    print("\n\n【演示5】不同角色类型的提示词对比")
    print("-" * 40)

    role_types = ["技术支持", "产品顾问", "客服代表", "研发专家"]
    domains = ["网络设备", "企业软件", "硬件产品"]

    for domain in domains:
        print(f"\n>>> 领域：{domain}")
        for role_type in role_types:
            factory = EnterprisePromptFactory(domain=domain, role_type=role_type)
            assembler = factory.create_assembler()
            prompt_preview = assembler.system_template.template[:50]
            print(f"  {role_type}：{prompt_preview}...")

    # ---------------------------------------------------------
    # 演示6：LlamaIndex QueryEngine集成
    # ---------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("【演示6】LlamaIndex QueryEngine 集成")
    print("=" * 60)

    demo_llamaindex_query_engine()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
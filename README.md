# 企业级 RAG 应用学习课程

基于多智能体协作构建的企业级 RAG（检索增强生成）系统学习内容，涵盖从基础概念到高级实践的完整知识体系。

---

## 课程概述

本课程旨在帮助学习者快速入门企业级 RAG 应用，掌握 RAG 所有核心知识点。课程结合原始学习资料与最新 RAG 技术发展，提供系统性的理论讲解和企业级代码实践。

**学习目标：**
- 理解 RAG 的本质是"查资料后再回答问题"的开卷考试模式
- 掌握企业级 RAG 系统的各个组件和工作流程
- 具备独立设计和实现生产级 RAG 系统的能力

---

## 学习路径

课程共 18 节，按认知递进规律分为四个阶段：

### 基础阶段（Lesson 01-04）
| 节次 | 标题 | 学习目标 |
|------|------|----------|
| 01 | RAG基础概念与工作原理 | 理解RAG核心定义、工作流程、四代演进 |
| 02 | 文档加载解析与预处理 | 掌握PDF/Word/HTML等格式解析方法 |
| 03 | 文档分割策略与实现 | 掌握多种分割方法和策略选择 |
| 04 | 向量化模型选型与嵌入生成 | 理解Embedding原理和模型选型 |

### 进阶阶段（Lesson 05-09）
| 节次 | 标题 | 学习目标 |
|------|------|----------|
| 05 | 向量数据库选型与元数据过滤 | 掌握Chroma/FAISS/Milvus/Pinecone选型 |
| 06 | 检索器类型与相似度计算 | 掌握余弦相似度、Top-K检索实现 |
| 07 | 提示词工程核心原理 | 掌握系统提示词、上下文提示词设计 |
| 08 | 答案生成与引用溯源 | 掌握带引用答案生成方法 |
| 09 | LlamaIndex核心对象与存储架构 | 掌握Document/Node/StorageContext |

### 企业级阶段（Lesson 10-14）
| 节次 | 标题 | 学习目标 |
|------|------|----------|
| 10 | 高级检索策略 | 掌握Sentence Window/HyDE/Self-RAG/GraphRAG |
| 11 | 混合检索与重排序策略 | 掌握BM25+向量混合+Reranker精排 |
| 12 | RAG质量评估体系 | 掌握Faithfulness/NDCG等评估指标 |
| 13 | 模块化RAG架构设计 | 掌握Dataflow/Router/Query Pipeline |
| 14 | RAG智能体架构与工具调用 | 掌握ReAct范式和Agent工具调用 |

### 高级阶段（Lesson 15-18）
| 节次 | 标题 | 学习目标 |
|------|------|----------|
| 15 | 复杂查询处理与上下文压缩 | 掌握Query Decomposition/CRAG/压缩技术 |
| 16 | 多模态RAG与最新技术 | 理解图文检索、Visual QA等最新发展 |
| 17 | 企业级RAG架构设计与实践 | 掌握高可用、多租户、监控告警设计 |
| 18 | RAG常见误区与最佳实践 | 识别常见误区，掌握最佳实践 |

---

## 课程结构

```
rag-learning/
│
├── raw-material/                   # 原始学习资料（不进行Git跟踪）
│   ├── Part 1. 大模型RAG入门基础架构与实战/
│   ├── Part 2. 大模型RAG进阶多格式文档解析实战/
│   ├── Part 3. 大模型RAG文档切分进阶实战/
│   ├── Part 4. 大模型RAG嵌入向量数据库实战/
│   └── Part 5. 大模型RAG检索生成和评估实战/
│
├── part_01_RAG基础概念与工作原理/
│   ├── README.md                   # 课程介绍文档
│   └── 01_codes/
│       └── rag_basic_concept.py   # RAG基础流程代码
│
├── part_02_文档加载解析与预处理/
│   ├── README.md
│   └── 02_codes/
│       └── document_loader.py     # 文档解析代码
│
├── part_03_文档分割与向量化/
│   ├── 03_文档分割策略与实现.md    # 课程文档
│   └── 03_codes/
│       └── text_splitter.py       # 分割策略代码
│
├── part_04_向量数据库与元数据过滤/
│   ├── 04_向量化模型选型与嵌入生成.md
│   ├── 05_向量数据库选型与元数据过滤.md
│   └── 04_codes/
│       └── embedding_generator.py # Embedding代码
│       └── vector_store_select.py # 向量库选型代码
│       └── metadata_filter.py     # 元数据过滤代码
│
├── part_05_检索器与相似度计算/
│   ├── 06_检索器类型与相似度计算.md
│   └── 05_codes/
│       └── retriever_types.py     # 检索器代码
│
├── part_06_提示词工程与答案生成/
│   ├── 07_提示词工程核心原理.md
│   ├── 08_答案生成与引用溯源.md
│   └── 06_codes/
│       ├── prompt_engineering.py  # 提示词工程代码
│       └── answer_generator.py    # 答案生成代码
│
├── part_07_LlamaIndex核心对象与存储架构/
│   ├── 07_LlamaIndex核心对象与存储架构.md
│   └── 07_codes/
│       └── llamaindex_core.py     # LlamaIndex核心代码
│
├── part_08_高级检索策略/
│   ├── 10_高级检索策略.md
│   └── 08_codes/
│       └── advanced_retrieval.py  # 高级检索代码
│
├── part_09_混合检索与重排序策略/
│   ├── README.md
│   └── 09_codes/
│       ├── hybrid_search.py       # 混合检索代码
│       └── reranking.py           # 重排序代码
│
├── part_10_RAG质量评估体系/
│   ├── 12_RAG质量评估体系.md
│   └── 10_codes/
│       └── rag_evaluation.py      # 评估代码
│
├── part_11_模块化RAG架构设计/
│   ├── README.md
│   └── 11_codes/
│       └── modular_rag.py         # 模块化架构代码
│
├── part_12_RAG智能体架构与工具调用/
│   ├── README.md
│   └── 12_codes/
│       └── rag_agent.py           # 智能体代码
│
├── part_13_复杂查询处理与上下文压缩/
│   ├── 15_复杂查询处理与上下文压缩.md
│   └── 13_codes/
│       ├── complex_query.py       # 复杂查询代码
│       └── context_compression.py # 上下文压缩代码
│
├── part_14_多模态RAG与最新技术/
│   └── 14_codes/
│       └── multimodal_rag.py       # 多模态RAG代码
│
├── part_15_企业级RAG架构设计与实践/
│   └── 15_codes/
│       └── enterprise_rag_architecture.py  # 企业级架构代码
│
├── part_16_RAG常见误区与最佳实践/
│   └── 16_codes/
│       └── rag_best_practices.py  # 最佳实践代码
│
├── knowledge_framework.json       # 知识框架
├── learning_structure.json        # 课程结构
├── course_structure.json          # 节次结构
└── README.md                      # 本文件
```

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# 安装核心依赖
pip install llama-index llama-index-embeddings-openai llama-index-embeddings-huggingface
pip install chromadb faiss-cpu sentence-transformers python-dotenv
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. 开始学习

```bash
# 查看第01节内容
cat part_01_RAG基础概念与工作原理/README.md

# 运行第01节代码示例
python part_01_RAG基础概念与工作原理/01_codes/rag_basic_concept.py
```

---

## 知识框架

课程覆盖以下知识模块：

| 模块ID | 模块名称 | 级别 | 覆盖知识点数 |
|--------|----------|------|--------------|
| mod_001 | RAG基础概念与工作原理 | 基础 | 3 |
| mod_002 | 文档加载解析与预处理 | 基础 | 3 |
| mod_003 | 文档分割与向量化 | 基础 | 2 |
| mod_004 | 向量数据库选型与元数据过滤 | 进阶 | 2 |
| mod_005 | 检索器与相似度检索 | 进阶 | 2 |
| mod_006 | 混合检索与重排序策略 | 企业级 | 2 |
| mod_007 | 提示词工程与答案生成 | 基础 | 2 |
| mod_008 | LlamaIndex核心对象与存储架构 | 进阶 | 2 |
| mod_009 | RAG质量评估体系 | 企业级 | 3 |
| mod_010 | 高级检索策略 | 进阶 | 3 |
| mod_011 | 模块化RAG与智能体架构 | 企业级 | 2 |
| mod_012 | 复杂查询处理与上下文压缩 | 企业级 | 3 |
| mod_013 | 多模态RAG与最新技术 | 企业级 | 1 |
| mod_014 | 企业级RAG架构设计 | 企业级 | 2 |
| mod_015 | RAG常见误区与最佳实践 | 中级 | 5 |

**总计：15个模块，32个核心知识点**

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 文档解析 | unstructured.io, PyMuPDF, python-docx |
| Embedding | OpenAI text-embedding-3, BGE, M3E |
| 向量数据库 | Chroma, FAISS, Milvus, Pinecone |
| RAG 框架 | LangChain, LlamaIndex |
| LLM | OpenAI GPT, Claude, 国产模型 |
| 评估框架 | RAGAS, DeepEval, Trulens |

---

## .gitignore 说明

本项目配置的 `.gitignore` 会忽略以下文件：

| 类型 | 规则 | 原因 |
|------|------|------|
| 压缩包 | `*.zip` | 无法直接在 Git 查看 |
| Python 缓存 | `__pycache__/`, `*.py[cod]` | 可随时重新生成 |
| 虚拟环境 | `venv/`, `.venv/` | 个人环境配置 |
| 向量数据库 | `chroma/`, `milvus_data/` | 可重新构建 |
| IDE | `.vscode/`, `.idea/` | 个人配置 |
| 日志 | `*.log`, `logs/` | 可重新生成 |
| 系统文件 | `.DS_Store`, `Thumbs.db` | 操作系统文件 |

---

## 构建状态

| 阶段 | 状态 |
|------|------|
| Phase 0: 环境准备 | ✅ 完成 |
| Phase 1: 知识框架设计 | ✅ 完成 |
| Phase 2: 课程结构设计 | ✅ 完成 |
| Phase 3: 内容构建（18节） | ✅ 完成 |
| Phase 4: 最终审核与修复 | ✅ 完成 |
| Phase 5: 生成配置文件 | ✅ 完成 |

---

## 参与贡献

本课程的学习内容由 AI 多智能体协作构建，结合原始学习资料和最新 RAG 技术发展。

---

## 许可

MIT License
# RAG 实战课程

## 课程简介

企业级 RAG（检索增强生成）快速入门课程，涵盖从基础认知到企业级实战的完整学习路径。

**学习时长：** 2-3周
**难度等级：** 企业级快速入门
**前置要求：** Python 基础、了解大模型基本概念（Prompt、Token、Chat Completions API）

---

## 需求解析

| 参数 | 值 |
|------|-----|
| **主题** | RAG（检索增强生成）企业级实战 |
| **受众** | 初学者 - 具备 Python 基础、了解大模型基本概念 |
| **级别** | 企业级快速入门 |
| **学习时长** | 2-3周 |
| **技术栈** | Python, LangChain, LlamaIndex, 向量数据库 |

---

## 原始资料

本课程的原始学习资料保存在项目根目录的 `raw-material/` 目录：

```
raw-material/
├── Part 1. 大模型RAG入门基础架构介绍/
│   ├── *.ipynb
│   └── 演示资料.zip
├── Part 1. 大模型RAG入门基础架构与实战/
├── Part 2. 大模型RAG进阶多格式文档解析实战/
├── Part 3. 大模型RAG文档切分进阶实战/
├── Part 4. 大模型RAG嵌入向量数据库实战/
└── Part 5. 大模型RAG检索生成和评估实战/
```

**说明：**
- 所有原始课件已迁移到 `raw-material/` 目录
- 保持原始目录结构和文件名
- 原始资料仅作为构建学习内容的参考
- 压缩包文件（*.zip）已配置不进行 Git 跟踪

---

## 知识框架

### 框架概述

本课程面向具备 Python 基础、了解大模型基本概念的开发者，采用"为什么需要 → 是什么 → 怎么工作 → 亲手搭建 → 工程化落地"的主线，涵盖 RAG 全流程核心技术。

### 基础阶段

#### 模块 1.1: RAG 基础认知

**学习目标：**
- 理解大模型的四大知识局限
- 掌握三种知识注入方案的差异
- 理解 RAG 的核心思想（检索-增强-生成）

**核心内容：**
- 大模型的知识困境：幻觉、知识截止、领域盲区、上下文限制
- 三种方案对比：Prompt Engineering vs Fine-tuning vs RAG
- RAG 三步走：Retrieve → Augment → Generate

#### 模块 1.2: RAG 系统架构

**学习目标：**
- 掌握 RAG 六大核心组件
- 理解数据流转全过程
- 了解主流开发框架

**核心内容：**
- 文档加载器、文本切分器、Embedding 模型
- 向量数据库、检索器、生成器
- LangChain 和 LlamaIndex 框架对比

### 进阶阶段

#### 模块 2.1: 多格式文档解析

**学习目标：**
- 掌握 PDF、Word、Excel 等格式解析
- 理解不同文档类型的解析挑战
- 能够处理多格式混合文档

**核心内容：**
- PDF 布局分析与文本提取（PyMuPDF / Pdfplumber）
- Word 段落和表格提取（python-docx）
- Excel 结构化数据提取（pandas）
- PPT 内容提取（python-pptx）
- OCR 文字识别（RapidOCR / pytesseract）

#### 模块 2.2: 文档切分策略

**学习目标：**
- 掌握数据清洗与预处理
- 理解多种切分策略
- 能够设计合理的切分方案

**核心内容：**
- 数据清洗：去除噪声、标准化格式
- 固定大小切分、递归切分、语义切分
- chunk_size 和 overlap 参数优化

#### 模块 2.3: Embedding 与向量数据库

**学习目标：**
- 理解 Embedding 原理
- 掌握主流向量数据库
- 能够搭建向量存储检索系统

**核心内容：**
- Embedding 模型原理和选型（OpenAI、BGE、M3E）
- 向量数据库对比（Chroma、Milvus、Pinecone、Weaviate、Faiss）
- 相似度度量：余弦相似度、点积、欧氏距离

### 高级阶段

#### 模块 3.1: 高级检索策略

**学习目标：**
- 理解检索流程四步
- 掌握检索优化技术
- 能够优化检索效果

**核心内容：**
- Query 向量化 → 初步召回 → 精排 → 过滤合并
- 混合检索（向量 + 关键词）
- Reranker 重排序
- 上下文压缩

#### 模块 3.2: RAG 评估体系

**学习目标：**
- 理解 RAG 评估指标
- 掌握主流评估工具
- 能够系统性评估 RAG 系统

**核心内容：**
- 检索质量指标：召回率、准确率、MRR
- 生成质量指标：相关性、完整性、幻觉率
- RAGAS、Trulens 评估框架
- Bad Case 分析方法

#### 模块 3.3: 企业级 RAG 实战

**学习目标：**
- 理解从原型到生产的迁移
- 掌握工程化最佳实践
- 能够设计企业级 RAG 系统

**核心内容：**
- 架构演进：原型 → 开发 → 生产 → 持续优化
- 工程化实践：多级缓存、并发控制、熔断器
- 安全合规：敏感过滤、权限控制、审计日志

---

## 内容结构

### 课程列表

| 课ID | 标题 | 类型 | 核心知识点 | 配套文件 |
|------|------|------|-----------|---------|
| 01 | RAG基础：为什么需要检索增强 | 理论 | 大模型四大局限、三种方案对比 | content.md |
| 02 | RAG核心思想：检索-增强-生成 | 理论 | RAG三步走、开卷考试 | content.md |
| 03 | RAG系统架构与组件 | 理论+实战 | 六大组件、数据流转 | content.md |
| 04 | 多格式文档解析 | 实战 | PDF/Word/Excel/PPT解析 | content.md, code/doc_parser.py |
| 05 | 文档切分策略 | 实战 | 数据清洗、递归切分 | content.md, code/text_splitter.py |
| 06 | Embedding与向量数据库 | 实战 | 向量库使用、相似度计算 | content.md, code/vector_store.py |
| 07 | 高级检索策略 | 实战 | 混合检索、Reranker | content.md, code/retriever.py |
| 08 | RAG评估体系 | 实战 | RAGAS、Bad Case分析 | content.md, code/evaluator.py |
| 09 | 企业级RAG实战 | 实战 | 架构设计、工程化 | content.md, code/rag_system.py |

---

## 课程目录

```
RAG实战课程/
├── README.md                      # 本文件
├── COURSE_META.json               # 课程元信息
├── KNOWLEDGE_FRAMEWORK.json       # 知识框架
├── CONTENT_STRUCTURE.json         # 内容结构
├── .gitignore                    # Git 忽略文件
├── lessons/                       # 课程内容
│   ├── 01-rag-basics-why-retrieval-augmentation/
│   │   └── content.md
│   ├── 02-rag-core-concept-retrieve-augment-generate/
│   │   └── content.md
│   ├── 03-rag-system-architecture/
│   │   └── content.md
│   ├── 04-multi-format-document-parsing/
│   │   ├── content.md
│   │   └── code/
│   │       └── doc_parser.py
│   ├── 05-document-chunking-strategies/
│   │   ├── content.md
│   │   └── code/
│   │       ├── text_splitter.py
│   │       └── data_cleaner.py
│   ├── 06-embedding-and-vector-database/
│   │   ├── content.md
│   │   └── code/
│   │       ├── vector_store.py
│   │       └── embedding_demo.py
│   ├── 07-advanced-retrieval-strategies/
│   │   ├── content.md
│   │   └── code/
│   │       └── retriever.py
│   ├── 08-rag-evaluation/
│   │   ├── content.md
│   │   └── code/
│   │       └── evaluator.py
│   └── 09-enterprise-rag-practice/
│       ├── content.md
│       └── code/
│           ├── rag_system.py
│           └── config.py
└── assets/
    ├── diagrams/
    └── images/
```

---

## 构建状态

- [x] 阶段1: 需求解析
- [x] 阶段2: 资料迁移
- [x] 阶段3: 框架设计
- [x] 阶段4: 结构规划
- [x] 阶段5: 框架审核
- [x] 阶段6: 任务拆分
- [x] 阶段7: 并行构建
- [x] 阶段8: 最终审核
- [x] 阶段9: 生成产物

### 已完成课程

| 课ID | 标题 | 状态 |
|------|------|------|
| 01 | RAG基础：为什么需要检索增强 | ✅ |
| 02 | RAG核心思想：检索-增强-生成 | ✅ |
| 03 | RAG系统架构与组件 | ✅ |
| 04 | 多格式文档解析 | ✅ |
| 05 | 文档切分策略 | ✅ |
| 06 | Embedding与向量数据库 | ✅ |
| 07 | 高级检索策略 | ✅ |
| 08 | RAG评估体系 | ✅ |
| 09 | 企业级RAG实战 | ✅ |

---

## 核心技术点

### 核心概念

| 概念 | 说明 |
|------|------|
| **RAG** | Retrieval-Augmented Generation，检索增强生成 |
| **Embedding** | 将文本转换为向量表示的技术 |
| **向量数据库** | 存储文本向量，支持高效相似度搜索 |
| **幻觉 (Hallucination)** | 大模型生成看似合理但与事实不符的内容 |
| **知识截止 (Knowledge Cutoff)** | 训练数据的截止日期 |
| **Chunk** | 被切分后的文档片段 |

### 技术栈

| 领域 | 技术 |
|------|------|
| 文档解析 | PyMuPDF, Pdfplumber, python-docx, pandas, python-pptx |
| Embedding | OpenAI text-embedding-3-small, BGE, M3E |
| 向量数据库 | Chroma, Milvus, Pinecone, Weaviate, Faiss |
| RAG框架 | LangChain, LlamaIndex |
| 评估 | RAGAS, Trulens |

---

## 开始学习

### 环境准备

```bash
# 克隆项目
git clone <repo>
cd rag-learning

# 安装 Python 依赖（按需）
pip install langchain langchain-community chromadb openai
pip install pymupdf python-docx openpyxl python-pptx
pip install pandas numpy
```

### 学习路径

1. **基础阶段（课程 01-03）**：理解 RAG 核心概念和系统架构
2. **进阶阶段（课程 04-06）**：掌握文档处理、Embedding、向量检索
3. **高级阶段（课程 07-09）**：高级检索策略、评估体系、工程化

---

## 项目结构

```
rag-learning/
├── RAG实战课程/                # 学习内容
│   ├── README.md
│   ├── lessons/               # 9节课程内容
│   └── assets/
│
├── raw-material/              # 原始资料（不进行 Git 跟踪）
│   ├── Part 1. xxx/
│   ├── Part 2. xxx/
│   └── ...
│
├── .gitignore                 # Git 忽略规则
└── README.md                  # 项目入口（本文件）
```

---

## .gitignore 说明

本项目配置的 `.gitignore` 规则：

**不跟踪的文件：**
- 压缩包（*.zip）：无法直接在 Git 查看
- Claude 配置（.claude/）：工作区特定
- 缓存（__pycache__/, .ruff_cache/）：可重新生成
- 向量数据库（chroma/, milvus/）：可重新构建
- 临时文件（*.tmp, *.log）

**跟踪的文件：**
- RAG实战课程/ 目录下的所有内容
- 本 README.md

---

## 参与贡献

本课程由 AI 多智能体协作构建。如有内容问题，欢迎提交 Issue 或 Pull Request。

---

## 许可

MIT License

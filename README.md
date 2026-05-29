# RAG Learning

基于多智能体协作的 RAG（检索增强生成）学习内容构建系统。

---

## 项目结构

```
rag-learning/
│
├── RAG实战课程/                 # 🎓 构建完成的学习内容
│   ├── README.md               # 课程详情
│   ├── COURSE_META.json        # 课程元信息
│   ├── KNOWLEDGE_FRAMEWORK.json # 知识框架
│   ├── CONTENT_STRUCTURE.json   # 内容结构
│   ├── lessons/                 # 9 节课程
│   │   ├── 01-xxx/content.md
│   │   ├── 02-xxx/content.md
│   │   └── ...
│   └── assets/                 # 资源文件
│
├── raw-material/               # 📚 原始学习资料（不进行 Git 跟踪）
│   ├── Part 1. 大模型RAG入门基础架构与实战/
│   ├── Part 2. 大模型RAG进阶多格式文档解析实战/
│   ├── Part 3. 大模型RAG文档切分进阶实战/
│   ├── Part 4. 大模型RAG嵌入向量数据库实战/
│   └── Part 5. 大模型RAG检索生成和评估实战/
│
├── .gitignore                  # Git 忽略规则
├── .claude/                    # Claude Code 配置
│   └── skills/
│       └── content-builder/     # 内容构建技能
│
└── README.md                   # 本文件（项目入口）
```

---

## 目录说明

### RAG实战课程/

基于原始课件构建的企业级 RAG 学习内容，包含：

| 课程 | 主题 | 内容 |
|------|------|------|
| 01 | RAG 基础认知 | 大模型四大局限、三种知识注入方案 |
| 02 | RAG 核心思想 | 检索-增强-生成三步走 |
| 03 | RAG 系统架构 | 六大组件、数据流转、框架对比 |
| 04 | 多格式文档解析 | PDF/Word/Excel/PPT/OCR |
| 05 | 文档切分策略 | 数据清洗、递归切分、语义切分 |
| 06 | Embedding 与向量数据库 | 向量化和相似度检索 |
| 07 | 高级检索策略 | 混合检索、Reranker、上下文压缩 |
| 08 | RAG 评估体系 | RAGAS、Trulens、Bad Case 分析 |
| 09 | 企业级 RAG 实战 | 架构设计、工程化最佳实践 |

### raw-material/

原始学习资料（原始课件、notebook、压缩包），**已配置不进行 Git 跟踪**。

### .claude/skills/content-builder/

内容构建技能（Skill），用于自动化构建学习内容。

---

## .gitignore 说明

本项目配置的 `.gitignore` 会忽略以下文件：

| 类型 | 规则 | 原因 |
|------|------|------|
| 压缩包 | `*.zip`, `*.7z`, `*.tar.gz` 等 | 无法直接在 Git 查看 |
| Claude 配置 | `.claude/` | 工作区特定配置 |
| 缓存 | `__pycache__/`, `.ruff_cache/` | 可随时重新生成 |
| Python | `*.pyc`, `venv/`, `*.so` | 构建产物 |
| 向量数据库 | `chroma/`, `milvus/` | 可重新构建 |
| IDE | `.vscode/`, `.idea/` | 个人配置 |
| 系统文件 | `.DS_Store`, `Thumbs.db` | 操作系统文件 |

---

## 快速开始

### 查看学习内容

```bash
# 进入学习内容目录
cd RAG实战课程/

# 查看课程列表
cat lessons/*/content.md | head -100
```

### 使用 content-builder 构建新内容

```bash
# 使用 Claude Code 调用 skill
/skill content-builder "你的学习主题"
```

---

## 构建状态

| 模块 | 状态 |
|------|------|
| RAG实战课程 | ✅ 完成 |
| raw-material 迁移 | ✅ 完成 |
| .gitignore 配置 | ✅ 完成 |

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 学习内容构建 | Claude Code + Multi-Agent |
| 文档解析 | PyMuPDF, python-docx, pandas |
| Embedding | OpenAI text-embedding-3-small, BGE |
| 向量数据库 | Chroma, Milvus, Pinecone |
| RAG 框架 | LangChain, LlamaIndex |

---

## 参与贡献

本项目的学习内容由 AI 多智能体协作构建。

---

## 许可

MIT License

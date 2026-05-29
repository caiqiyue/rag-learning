# 第五课：文档切分策略

## 本节概述

本节课将学习如何将长文档合理地分割成小块（Chunk），这是 RAG 系统中至关重要的一步。你将掌握：

- 数据清洗与预处理技术
- 多种文档切分策略
- 切分参数的优化方法
- 如何根据业务场景设计切分方案

**学习时长**：约 2-3 小时

---

## 5.1 为什么需要文档切分？

### 5.1.1 切分的原因

```
不切分的问题：
┌────────────────────────────────────────────────────┐
│  假设我们有 10000 页的文档                          │
│                                                     │
│  问题1：上下文窗口限制                               │
│  GPT-4o 上下文：128K tokens ≈ 70 页                 │
│  10000 页 >> 70 页，根本塞不进去！                   │
│                                                     │
│  问题2：检索精度低                                   │
│  用户问"第三章第5节的内容"                           │
│  整篇返回，包含大量无关的第一章、第二章               │
│                                                     │
│  问题3：计算成本高                                   │
│  向量化成本 = Token 数 × 单价                        │
│  整篇向量化 = $$$

└────────────────────────────────────────────────────┘
```

### 5.1.2 切分的目标

```
好的切分应该：
✓ 每个块有独立语义（可以单独理解）
✓ 块之间有上下文连贯性（不割裂语义）
✓ 块大小适中（适配检索和生成）
✓ 保留关键信息（不丢失重要内容）
```

---

## 5.2 数据清洗与预处理

### 5.2.1 为什么需要清洗？

原始文档往往包含各种"噪声"：

```
文档中的噪声：
├── HTML/XML 标签：<div>, <span>, <p> 等
├── Markdown 符号：**粗体**, [链接], #标题
├── 特殊字符：\n, \t, 多个连续空格
├── 代码片段：无意义的程序代码块
├── 页眉页脚：每页都有的重复内容
└── 水印、注释：不应进入检索的内容
```

### 5.2.2 文本清洗代码

```python
import re

def clean_text(text: str) -> str:
    """
    清洗文本中的噪声内容
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    # 1. 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 移除 Markdown 格式符号
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **粗体** → 粗体
    text = re.sub(r'\*(.+?)\*', r'\1', text)        # *斜体* → 斜体
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text) # [链接](url) → 链接
    text = re.sub(r'#+\s*', '', text)               # # 标题 → 标题
    
    # 3. 标准化空白字符
    text = re.sub(r'\s+', ' ', text)               # 多个空格 → 单个空格
    text = re.sub(r'\n\s*\n', '\n\n', text)          # 多个换行 → 两个换行
    
    # 4. 修复断开的单词（PDF 转换常见问题）
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # 5. 标准化引号
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def clean_pdf_text(text: str) -> str:
    """
    专门针对 PDF 转换文本的清洗
    
    PDF 转文本常见问题：
    - 单词被断开：comp- any → company
    - 页码混入正文：123
    - 页眉页脚重复
    """
    # 1. 修复断开的单词
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 2. 移除孤立的大写字母（通常是页眉页脚）
    # 例如：每行开头/结尾的单个大写字母
    text = re.sub(r'\n[A-Z]\s+', '\n', text)
    
    # 3. 移除纯数字行（通常是页码）
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # 4. 清洗后再次标准化空白
    text = re.sub(r'\s+', ' ', text)
    
    return text


# 使用示例
raw_text = """
<div class="content">
  <h1>**RAG基础教程**</h1>
  <p>这是一段关于RAG的    介绍文本。
  </p>
  <span class="footer">第 1 页</span>
</div>
"""

cleaned = clean_text(raw_text)
print(cleaned)
# 输出: RAG基础教程 这是一段关于RAG的介绍文本。
```

### 5.2.3 文档预处理完整流程

```python
from pathlib import Path
from typing import List, Callable

class DocumentPreprocessor:
    """
    文档预处理器
    包含多种清洗和标准化操作
    """
    
    def __init__(self):
        self.cleaners: List[Callable] = []
        self._setup_default_cleaners()
    
    def _setup_default_cleaners(self):
        """设置默认的清洗函数"""
        self.cleaners = [
            self._remove_html_tags,
            self._normalize_whitespace,
            self._fix_hyphenation,
            self._normalize_quotes,
            self._remove_special_chars,
        ]
    
    def preprocess(self, text: str) -> str:
        """执行所有清洗步骤"""
        for cleaner in self.cleaners:
            text = cleaner(text)
        return text
    
    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r'<[^>]+>', '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _fix_hyphenation(self, text: str) -> str:
        """修复断字（PDF 转换常见）"""
        return re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    def _normalize_quotes(self, text: str) -> str:
        replacements = {
            '"': '"', '"': '"',
            ''': "'", ''': "'",
            '«': '"', '»': '"'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符（可自定义）"""
        # 保留中文、英文、数字、常用标点
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:()（）【】\[\]、，。！？；：]', '', text)


# 使用示例
preprocessor = DocumentPreprocessor()
cleaned_text = preprocessor.preprocess(raw_text)
```

---

## 5.3 切分策略

### 5.3.1 固定大小切分

最简单的切分方式，按字符或 Token 数切分。

```python
def split_by_size(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    固定大小切分
    
    Args:
        text: 要切分的文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
    
    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 确保在句子边界切分（可选）
        if end < len(text):
            # 尝试找到最近的句子结束符
            for sep in ['。', '！', '？', '.', '!', '?', '\n']:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + 1
                    break
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 下一个块的起始位置（考虑重叠）
        start = end - chunk_overlap
    
    return chunks


# 使用示例
text = "这是第一段文本。" * 100  # 模拟长文本
chunks = split_by_size(text, chunk_size=50, chunk_overlap=10)
print(f"切成 {len(chunks)} 个块")
print(f"第一个块: {chunks[0][:30]}...")
```

### 5.3.2 递归切分（推荐）

按层次尝试不同的分隔符，确保在好的边界切分。

```python
from typing import List, Callable

class RecursiveTextSplitter:
    """
    递归文本切分器
    
    原理：先尝试用大的分隔符（如\n\n）切分，
    如果块太大，再用小的分隔符（如\n）切分，
    以此类推，直到块大小合适。
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分隔符（优先级从高到低）
        if separators is None:
            self.separators = [
                "\n\n",      # 段落分隔
                "\n",        # 换行
                "。",        # 中文句号
                "！",        # 中文感叹
                "？",        # 中文问号
                ". ",        # 英文句号+空格
                "! ",        # 英文感叹+空格
                "? ",        # 英文问号+空格
                "；",        # 中文分号
                "，",        # 中文逗号
                " ",         # 空格
                ""           # 字符级别（最后兜底）
            ]
        else:
            self.separators = separators
    
    def split(self, text: str) -> List[str]:
        """递归切分文本"""
        return self._split_recursive(text, self.separators)
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """递归切分"""
        if not separators:
            # 最后一层：按字符数硬切
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # 用当前分隔符分割
        if separator:
            parts = text.split(separator)
        else:
            # 最后一个分隔符是空字符串，意味着按字符切
            parts = list(text)
        
        # 检查每部分的大小
        good_parts = []
        current_part = ""
        
        for part in parts:
            test_part = current_part + separator + part if current_part else part
            
            if len(test_part) <= self.chunk_size:
                current_part = test_part
            else:
                # 当前部分太大了
                if current_part:
                    good_parts.append(current_part)
                
                # 递归处理超长的部分
                if len(part) > self.chunk_size:
                    # 继续用更小的分隔符切
                    sub_parts = self._split_recursive(part, remaining_separators)
                    good_parts.extend(sub_parts)
                    current_part = ""
                else:
                    current_part = part
        
        # 处理最后一部分
        if current_part:
            if len(current_part) > self.chunk_size and remaining_separators:
                sub_parts = self._split_recursive(current_part, remaining_separators)
                good_parts.extend(sub_parts)
            else:
                good_parts.append(current_part)
        
        # 处理重叠
        return self._add_overlap(good_parts)
    
    def _split_by_size(self, text: str) -> List[str]:
        """硬切分（不考虑语义）"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠"""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            # 取上一个块的末尾部分作为重叠
            prev_chunk = result[-1]
            overlap_text = prev_chunk[-self.chunk_overlap:]
            
            # 新的块 = 重叠部分 + 新内容
            new_chunk = overlap_text + chunks[i]
            result[-1] = overlap_text  # 更新上一个块的末尾
            result.append(new_chunk)
        
        return result


# 使用示例
splitter = RecursiveTextSplitter(chunk_size=200, chunk_overlap=30)

sample_text = """
第一章 RAG 基础

RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写。
它由三个核心步骤组成：检索（Retrieve）、增强（Augment）和生成（Generate）。

RAG 的核心思想是让大模型从"闭卷考试"变成"开卷考试"。
通过先检索相关文档，再基于检索结果生成回答，
RAG 能够显著提高回答的准确性和可靠性。

第二章 Embedding

Embedding 是将文本转换为向量表示的技术。
它能够捕捉文本的语义信息，使得语义相似的文本在向量空间中相近。
"""

chunks = splitter.split(sample_text)
for i, chunk in enumerate(chunks):
    print(f"块 {i+1} ({len(chunk)} 字): {chunk[:50]}...")
```

### 5.3.3 语义切分（高级）

按语义单元（句子、段落、章节）切分，保留完整语义。

```python
import re

class SemanticTextSplitter:
    """
    语义切分器
    按句子/段落/章节等语义单元切分
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_by_sentences(self, text: str) -> List[str]:
        """按句子分割"""
        # 中文句号、感叹、问号 + 英文句号、感叹、问号
        sentence_split = re.compile(r'[。！？.!?]+')
        sentences = sentence_split.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_by_chapters(self, text: str) -> List[tuple]:
        """
        按章节分割
        返回 (章节标题, 章节内容) 的元组列表
        """
        # 匹配章节标题：第X章、第一章、1. 等格式
        chapter_pattern = re.compile(r'(^第[一二三四五六七八九十]+章|^第[0-9]+章|^[0-9]+\.[^\n]+)', re.MULTILINE)
        
        chapters = []
        current_title = "前言"
        current_content = ""
        
        lines = text.split('\n')
        for line in lines:
            if chapter_pattern.match(line.strip()):
                # 保存上一章
                if current_content:
                    chapters.append((current_title, current_content))
                current_title = line.strip()
                current_content = ""
            else:
                current_content += line + "\n"
        
        # 保存最后一章
        if current_content:
            chapters.append((current_title, current_content))
        
        return chapters
    
    def split_semantic(
        self,
        text: str,
        mode: str = "paragraph"
    ) -> List[str]:
        """
        语义切分
        
        Args:
            text: 文本
            mode: 切分模式
                - "sentence": 按句子
                - "paragraph": 按段落
                - "chapter": 按章节
        """
        if mode == "sentence":
            units = self.split_by_sentences(text)
        elif mode == "paragraph":
            units = self.split_by_paragraphs(text)
        elif mode == "chapter":
            chapters = self.split_by_chapters(text)
            return [f"{title}\n\n{content}" for title, content in chapters]
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 将小单元合并成大块
        chunks = []
        current_chunk = ""
        
        for unit in units:
            if len(current_chunk) + len(unit) <= self.chunk_size:
                current_chunk += unit + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 如果单个 unit 就超过 chunk_size，需要硬切
                if len(unit) > self.chunk_size:
                    chunks.extend(self._split_by_size_hard(unit))
                else:
                    current_chunk = unit + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_size_hard(self, text: str) -> List[str]:
        """硬切分"""
        return [
            text[i:i+self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]


# 使用示例
splitter = SemanticTextSplitter(chunk_size=200)

# 按章节切分
chapters = splitter.split_semantic(sample_text, mode="chapter")
for title, content in chapters:
    print(f"章节: {title}")
    print(f"内容预览: {content[:100]}...")
    print()
```

---

## 5.4 切分参数优化

### 5.4.1 关键参数

```
切分参数：
├── chunk_size: 每个块的最大字符/Token 数
├── chunk_overlap: 相邻块之间的重叠大小
└── separator: 分隔符（用于确定切分边界）

参数选择的影响：
├── chunk_size 太小 → 每个块语义不完整
├── chunk_size 太大 → 包含过多噪声，检索精度下降
├── overlap 太小 → 跨块的关键信息可能丢失
├── overlap 太大 → 冗余增加，计算成本上升
```

### 5.4.2 参数选择指南

```python
"""
切分参数选择指南
"""

# 基于 Embedding 模型的 Token 限制选择 chunk_size
CHUNK_SIZE_GUIDE = {
    # OpenAI 的 token 计算是 1 token ≈ 4 字符（中文）
    "gpt-4": {
        "max_tokens": 8192,
        "recommended_chunk_size": 1000,  # 约 4000 字符
        "for": "长文档优先"
    },
    "text-embedding-3-small": {
        "max_tokens": 8191,
        "recommended_chunk_size": 1000,
        "for": "通用场景"
    },
    "BGE-large-zh": {
        "max_tokens": 512,
        "recommended_chunk_size": 256,  # 约 1000 字符
        "for": "中文向量模型"
    }
}

# 基于文档类型选择策略
STRATEGY_GUIDE = {
    "技术文档": {
        "chunk_size": 500-800,
        "overlap": 50-100,
        "strategy": "recursive",
        "reason": "技术文档结构清晰，按段落切分效果好"
    },
    "法律合同": {
        "chunk_size": 300-500,
        "overlap": 30-50,
        "strategy": "semantic",
        "reason": "法律文本需要精确，块太小可能丢失法律语义"
    },
    "财务报表": {
        "chunk_size": 200-400,
        "overlap": 20-30,
        "strategy": "semantic+table",
        "reason": "表格和数字密集，需要特殊处理"
    },
    "聊天记录": {
        "chunk_size": 1000-2000,
        "overlap": 100-200,
        "strategy": "session",
        "reason": "按会话切分，保留完整对话上下文"
    }
}

def suggest_chunk_params(document_type: str, model: str = "text-embedding-3-small") -> dict:
    """根据文档类型推荐切分参数"""
    guide = STRATEGY_GUIDE.get(document_type, STRATEGY_GUIDE["技术文档"])
    model_guide = CHUNK_SIZE_GUIDE.get(model, CHUNK_SIZE_GUIDE["text-embedding-3-small"])
    
    return {
        "chunk_size": guide["chunk_size"],
        "chunk_overlap": guide["overlap"],
        "strategy": guide["strategy"],
        "reason": guide["reason"],
        "max_tokens": model_guide["max_tokens"],
        "recommended": model_guide["recommended_chunk_size"]
    }
```

---

## 5.5 统一切分接口

```python
from typing import List, Dict

class TextSplitter:
    """
    统一文本切分器
    支持多种切分策略
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "recursive"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # 初始化各策略的切分器
        self.recursive_splitter = RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.semantic_splitter = SemanticTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split(self, text: str) -> List[str]:
        """根据策略切分文本"""
        if self.strategy == "recursive":
            return self.recursive_splitter.split(text)
        elif self.strategy == "semantic":
            return self.semantic_splitter.split_semantic(text, mode="paragraph")
        elif self.strategy == "sentence":
            return self.semantic_splitter.split_semantic(text, mode="sentence")
        elif self.strategy == "chapter":
            return self.semantic_splitter.split_semantic(text, mode="chapter")
        elif self.strategy == "fixed":
            return self._split_fixed(text)
        else:
            raise ValueError(f"不支持的策略: {self.strategy}")
    
    def _split_fixed(self, text: str) -> List[str]:
        """固定大小切分"""
        return [
            text[i:i+self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            if i + self.chunk_size <= len(text) or i == 0
        ]


def create_chunks(
    documents: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive"
) -> List[Dict]:
    """
    为文档列表创建切分块
    
    Args:
        documents: 文档列表，每个文档包含 content 和 metadata
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        strategy: 切分策略
    
    Returns:
        块列表，每个块包含 content, metadata, chunk_index
    """
    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy
    )
    
    chunks = []
    chunk_id = 0
    
    for doc in documents:
        text = doc["content"]
        metadata = doc.get("metadata", {})
        
        text_chunks = splitter.split(text)
        
        for chunk in text_chunks:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "content": chunk,
                "char_count": len(chunk),
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_id
                }
            })
            chunk_id += 1
    
    return chunks


# 使用示例
if __name__ == "__main__":
    docs = [
        {
            "content": "第一篇文档的内容..." * 50,
            "metadata": {"source": "doc1.pdf", "page": 1}
        },
        {
            "content": "第二篇文档的内容..." * 50,
            "metadata": {"source": "doc2.pdf", "page": 1}
        }
    ]
    
    chunks = create_chunks(docs, chunk_size=200, overlap=20)
    print(f"生成了 {len(chunks)} 个块")
```

---

## 本节总结

### 核心要点

1. **数据清洗**：移除 HTML 标签、标准化空白、修复断字
2. **切分策略**：固定大小、递归切分、语义切分
3. **参数选择**：chunk_size 根据模型和场景调整，overlap 保持上下文连贯

### 代码文件

- `code/text_splitter.py`: 多种切分策略实现
- `code/data_cleaner.py`: 数据清洗工具

### 思考题

1. 如果一个块太小（比如只有一句话），会有什么影响？
2. 为什么需要 chunk_overlap？overlap 设置太大或太小会有什么后果？
3. 对于包含表格的 PDF 文档，应该如何切分？

### 下节预告

下一节课我们将学习 Embedding 与向量数据库，掌握如何将文本转换为向量并存储到向量数据库中。

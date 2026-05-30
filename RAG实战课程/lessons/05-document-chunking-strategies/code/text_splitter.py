"""
文本切分器
支持多种切分策略：固定大小、递归、语义切分
"""

import re
from typing import List, Callable, Dict, Tuple


# ========== 基础切分器 ==========


def split_by_size(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
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

        # 尝试在句子边界切分
        if end < len(text):
            for sep in ["。", "！", "？", ".", "!", "?", "\n"]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + 1
                    break

        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def split_by_sentences(text: str) -> List[str]:
    """按句子分割"""
    sentence_split = re.compile(r"[。！？.!?]+")
    sentences = sentence_split.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_paragraphs(text: str) -> List[str]:
    """按段落分割"""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


# ========== 递归切分器 ==========


class RecursiveTextSplitter:
    """
    递归文本切分器

    先用大分隔符切分，如果块太大，再用小分隔符切分。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            self.separators = [
                "\n\n",  # 段落
                "\n",  # 换行
                "。",  # 中文句号
                "！",  # 感叹
                "？",  # 问号
                ". ",  # 英文句号
                "! ",  # 英文感叹
                "? ",  # 英文问号
                "；",  # 分号
                "，",  # 逗号
                " ",  # 空格
                "",  # 字符
            ]
        else:
            self.separators = separators

    def split(self, text: str) -> List[str]:
        """切分文本"""
        chunks = self._split_recursive(text, self.separators)
        return self._add_overlap(chunks)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归切分"""
        if not separators:
            return self._split_by_size_hard(text)

        separator = separators[0]
        remaining = separators[1:]

        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        good_parts = []
        current = ""

        for part in parts:
            test = current + separator + part if current else part

            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    good_parts.append(current)

                if len(part) > self.chunk_size:
                    sub = self._split_recursive(part, remaining)
                    good_parts.extend(sub)
                    current = ""
                else:
                    current = part

        if current:
            if len(current) > self.chunk_size and remaining:
                sub = self._split_recursive(current, remaining)
                good_parts.extend(sub)
            else:
                good_parts.append(current)

        return good_parts

    def _split_by_size_hard(self, text: str) -> List[str]:
        """硬切分"""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            if i + self.chunk_size <= len(text) or i == 0
        ]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠"""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = result[-1]
            overlap = prev[-self.chunk_overlap :]
            new_chunk = overlap + chunks[i]
            result[-1] = overlap
            result.append(new_chunk)

        return result


# ========== 语义切分器 ==========


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
        return split_by_sentences(text)

    def split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割"""
        return split_by_paragraphs(text)

    def split_by_chapters(self, text: str) -> List[Tuple[str, str]]:
        """按章节分割"""
        chapter_pattern = re.compile(
            r"(^第[一二三四五六七八九十]+章|^第[0-9]+章|^[0-9]+\.[^\n]+)", re.MULTILINE
        )

        chapters = []
        current_title = "前言"
        current_content = ""

        lines = text.split("\n")
        for line in lines:
            if chapter_pattern.match(line.strip()):
                if current_content:
                    chapters.append((current_title, current_content))
                current_title = line.strip()
                current_content = ""
            else:
                current_content += line + "\n"

        if current_content:
            chapters.append((current_title, current_content))

        return chapters

    def split_semantic(self, text: str, mode: str = "paragraph") -> List[str]:
        """语义切分"""
        if mode == "sentence":
            units = self.split_by_sentences(text)
        elif mode == "paragraph":
            units = self.split_by_paragraphs(text)
        elif mode == "chapter":
            chapters = self.split_by_chapters(text)
            return [f"{title}\n\n{content}" for title, content in chapters]
        else:
            raise ValueError(f"不支持的模式: {mode}")

        # 合并小单元成大块
        chunks = []
        current = ""

        for unit in units:
            if len(current) + len(unit) <= self.chunk_size:
                current += unit + " "
            else:
                if current:
                    chunks.append(current.strip())
                if len(unit) > self.chunk_size:
                    chunks.extend(self._split_by_size_hard(unit))
                else:
                    current = unit + " "

        if current:
            chunks.append(current.strip())

        return chunks

    def _split_by_size_hard(self, text: str) -> List[str]:
        """硬切分"""
        return [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]


# ========== 统一切分器 ==========


class TextSplitter:
    """
    统一文本切分器
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        self.recursive = RecursiveTextSplitter(chunk_size, chunk_overlap)
        self.semantic = SemanticTextSplitter(chunk_size, chunk_overlap)

    def split(self, text: str) -> List[str]:
        """根据策略切分"""
        if self.strategy == "recursive":
            return self.recursive.split(text)
        elif self.strategy == "semantic":
            return self.semantic.split_semantic(text, mode="paragraph")
        elif self.strategy == "sentence":
            return self.semantic.split_semantic(text, mode="sentence")
        elif self.strategy == "chapter":
            return self.semantic.split_semantic(text, mode="chapter")
        elif self.strategy == "fixed":
            return split_by_size(text, self.chunk_size, self.chunk_overlap)
        else:
            raise ValueError(f"不支持的策略: {self.strategy}")


# ========== 文档块创建 ==========


def create_chunks(
    documents: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> List[Dict]:
    """
    为文档列表创建切分块

    Args:
        documents: 文档列表 [{"content": "...", "metadata": {...}}]
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        strategy: 切分策略

    Returns:
        块列表 [{"chunk_id": "...", "content": "...", "metadata": {...}}]
    """
    splitter = TextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy
    )

    chunks = []
    chunk_id = 0

    for doc in documents:
        text = doc["content"]
        metadata = doc.get("metadata", {})

        text_chunks = splitter.split(text)

        for chunk in text_chunks:
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id}",
                    "content": chunk,
                    "char_count": len(chunk),
                    "metadata": {**metadata, "chunk_index": chunk_id},
                }
            )
            chunk_id += 1

    return chunks


# ========== 参数建议 ==========

CHUNK_SIZE_GUIDE = {
    "gpt-4": {"max_tokens": 8192, "recommended": 1000},
    "text-embedding-3-small": {"max_tokens": 8191, "recommended": 1000},
    "BGE-large-zh": {"max_tokens": 512, "recommended": 256},
}

STRATEGY_GUIDE = {
    "技术文档": {
        "chunk_size": (500, 800),
        "overlap": (50, 100),
        "strategy": "recursive",
    },
    "法律合同": {"chunk_size": (300, 500), "overlap": (30, 50), "strategy": "semantic"},
    "财务报表": {"chunk_size": (200, 400), "overlap": (20, 30), "strategy": "semantic"},
    "聊天记录": {
        "chunk_size": (1000, 2000),
        "overlap": (100, 200),
        "strategy": "session",
    },
}


if __name__ == "__main__":
    sample = """
    第一章 RAG 基础
    
    RAG 是检索增强生成的缩写。它由三个核心步骤组成。
    
    第二章 Embedding
    
    Embedding 是将文本转换为向量表示的技术。
    """

    splitter = TextSplitter(chunk_size=100, overlap=20, strategy="recursive")
    chunks = splitter.split(sample)

    print(f"切成 {len(chunks)} 个块:")
    for i, c in enumerate(chunks):
        print(f"  块{i + 1}: {len(c)}字 - {c[:30]}...")

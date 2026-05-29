"""
数据清洗器
用于清洗 PDF、HTML、Markdown 等格式文档中的噪声
"""

import re
from typing import Callable, List


class TextCleaner:
    """
    文本清洗器
    包含多种清洗函数
    """

    def __init__(self):
        self.cleaners: List[Callable] = []
        self._setup_default()

    def _setup_default(self):
        """设置默认清洗函数"""
        self.cleaners = [
            self.remove_html_tags,
            self.remove_markdown_format,
            self.normalize_whitespace,
            self.fix_hyphenation,
            self.normalize_quotes,
            self.remove_special_chars,
            self.remove_page_numbers,
        ]

    def clean(self, text: str) -> str:
        """执行所有清洗"""
        for cleaner in self.cleaners:
            text = cleaner(text)
        return text

    def remove_html_tags(self, text: str) -> str:
        """移除 HTML 标签"""
        # 移除所有 HTML 标签
        text = re.sub(r"<[^>]+>", "", text)
        # 移除 HTML 实体
        text = re.sub(r"&[a-zA-Z]+;", "", text)
        text = re.sub(r"&#\d+;", "", text)
        return text

    def remove_markdown_format(self, text: str) -> str:
        """移除 Markdown 格式"""
        # **粗体** → 粗体
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        # *斜体* → 斜体
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        # __粗体__ → 粗体
        text = re.sub(r"__(.+?)__", r"\1", text)
        # _斜体_ → 斜体
        text = re.sub(r"_(.+?)_", r"\1", text)
        # [链接](url) → 链接
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        # 图片 ![alt](url) → alt
        text = re.sub(r"!\[.*?\]\(.+?\)", "", text)
        # # 标题 → 标题
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        # > 引用 → 引用内容
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
        # --- 分割线
        text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
        # `代码` → 代码
        text = re.sub(r"`(.+?)`", r"\1", text)
        # ```代码块``` → 代码块
        text = re.sub(r"```[\s\S]*?```", "", text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 多个空格 → 单个空格
        text = re.sub(r"[ \t]+", " ", text)
        # 多个换行 → 两个换行（保持段落分隔）
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 行首行尾空格
        text = re.sub(r"^ +| +$", "", text, flags=re.MULTILINE)
        return text.strip()

    def fix_hyphenation(self, text: str) -> str:
        """修复断字（PDF 转换常见问题）"""
        # word- → word (跨行断字)
        text = re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2", text)
        # word- → word (行尾连字符)
        text = re.sub(r"(\w+)-\s+$", r"\1", text, flags=re.MULTILINE)
        return text

    def normalize_quotes(self, text: str) -> str:
        """标准化引号"""
        replacements = {
            '"': '"',
            '"': '"',
            """: "'", """: "'",
            "«": '"',
            "»": '"',
            "『": '"',
            "』": '"',
            "「": '"',
            "」": '"',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def remove_special_chars(self, text: str) -> str:
        """移除特殊字符（可选）"""
        # 保留中文、英文、数字、常用标点
        # 移除控制字符
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def remove_page_numbers(self, text: str) -> str:
        """移除页码"""
        # 移除单独的页码（多出现在行首或行尾）
        text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n\d+\n", "\n", text)
        return text


class PDFTextCleaner(TextCleaner):
    """
    专门针对 PDF 转换文本的清洗器
    """

    def __init__(self):
        super().__init__()
        self.cleaners = [
            self.fix_hyphenation,
            self.remove_page_numbers,
            self.remove_isolated_letters,
            self.remove_header_footer,
            self.remove_html_tags,
            self.normalize_whitespace,
            self.normalize_quotes,
        ]

    def remove_isolated_letters(self, text: str) -> str:
        """移除孤立的单字母（通常是 PDF 转换错误）"""
        # 移除行首/行尾的孤立大写字母
        text = re.sub(r"\n[A-Z]\s+", "\n", text)
        text = re.sub(r"\s+[A-Z]\n", "\n", text)
        return text

    def remove_header_footer(self, text: str) -> str:
        """
        移除页眉页脚（需要已知格式）
        如果页眉页脚有固定格式，可以在这里处理
        """
        # 例如：每页都有的 "内部资料 - 禁止外传"
        text = re.sub(r"内部资料\s*-\s*禁止外传", "", text)
        return text


class HTMLCleaner(TextCleaner):
    """
    专门针对 HTML 的清洗器
    """

    def __init__(self):
        super().__init__()
        self.cleaners = [
            self.remove_html_tags,
            self.remove_scripts,
            self.remove_styles,
            self.normalize_whitespace,
        ]

    def remove_scripts(self, text: str) -> str:
        """移除 JavaScript"""
        text = re.sub(r"<script[\s\S]*?</script>", "", text)
        text = re.sub(r"<noscript[\s\S]*?</noscript>", "", text)
        return text

    def remove_styles(self, text: str) -> str:
        """移除 CSS"""
        text = re.sub(r"<style[\s\S]*?</style>", "", text)
        return text


# ========== 清洗流程 ==========


def clean_document(text: str, doc_type: str = "auto") -> str:
    """
    自动识别文档类型并清洗

    Args:
        text: 原始文本
        doc_type: 文档类型 ("pdf", "html", "markdown", "auto")

    Returns:
        清洗后的文本
    """
    if doc_type == "auto":
        # 根据文本特征自动判断
        if "<html" in text.lower() or "<div" in text.lower():
            doc_type = "html"
        elif "<!--" in text:
            doc_type = "html"
        else:
            doc_type = "pdf"  # 默认当 PDF 处理

    if doc_type == "pdf":
        cleaner = PDFTextCleaner()
    elif doc_type == "html":
        cleaner = HTMLCleaner()
    else:
        cleaner = TextCleaner()

    return cleaner.clean(text)


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 测试 PDF 清洗
    raw_pdf = """
    <div class="page">
    <p>**RAG基础教程**</p>
    <p>这是第一 comp-
    any 段落。</p>
    123
    <p>这是第二段落。</p>
    </div>
    456
    """

    print("原始文本:")
    print(raw_pdf)
    print("\n清洗后:")
    print(clean_document(raw_pdf, "pdf"))

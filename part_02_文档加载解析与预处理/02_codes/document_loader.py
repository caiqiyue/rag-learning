# -*- coding: utf-8 -*-
"""
第02节 文档加载解析与预处理 - 代码案例

本文件演示如何使用unstructured.io库进行多格式文档解析。

主要功能：
1. 使用partition函数自动检测并解析任意文档格式
2. 使用专用解析函数解析PDF、Word、HTML、Markdown、Excel、CSV、图片等格式
3. 解析结果的结构分析（Element元素类型、文本内容、元数据）
4. 批量文档解析功能
5. 数据清洗与预处理示例

依赖安装：
    pip install unstructured python-magic-bin pdf2image pytesseract poppler

可选依赖（用于OCR和高精度PDF解析）：
    pip install pytesseract

Tesseract OCR安装（Windows）：
    https://github.com/UB-Mannheim/tesseract/wiki/Download

Poppler安装（Windows，用于PDF转图像）：
    https://github.com/oschwartz10612/Poppler-Release/releases
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


# =============================================================================
# 第一部分：unstructured核心组件导入
# =============================================================================

# 从unstructured.partition.auto导入partition函数 - 这是自动检测文件类型的核心函数
# partition函数会根据文件扩展名或内容自动选择合适的解析策略
from unstructured.partition.auto import partition

# 导入各专用解析函数，针对不同文档格式
from unstructured.partition.pdf import partition_pdf       # PDF文档解析
from unstructured.partition.docx import partition_docx   # Word .docx文档解析
from unstructured.partition.doc import partition_doc     # Word .doc文档解析（老格式）
from unstructured.partition.html import partition_html     # HTML网页解析
from unstructured.partition.md import partition_md       # Markdown文档解析
from unstructured.partition.xlsx import partition_xlsx   # Excel .xlsx文档解析
from unstructured.partition.csv import partition_csv      # CSV文档解析
from unstructured.partition.image import partition_image  # 图片OCR解析

# 导入Element元素类型 - 解析结果由多个Element对象组成
from unstructured.documents.elements import Element


# =============================================================================
# 第二部分：数据结构定义
# =============================================================================

@dataclass
class DocumentInfo:
    """
    文档解析结果的数据类

    用于存储解析后的文档信息，便于后续处理和分析。

    属性说明：
    - file_path: 原始文件路径
    - file_extension: 文件扩展名（小写）
    - total_elements: 解析出的Element元素总数
    - element_types: 各类型元素的数量统计（字典）
    - text_content: 提取的纯文本内容
    - total_characters: 文本总字符数
    """
    file_path: str
    file_extension: str
    total_elements: int
    element_types: Dict[str, int]
    text_content: str
    total_characters: int


# =============================================================================
# 第三部分：核心解析函数
# =============================================================================

def parse_document_auto(file_path: str, strategy: str = "auto") -> DocumentInfo:
    """
    使用partition函数自动检测文件类型并解析文档

    这是最常用的通用解析函数，partition会自动：
    1. 根据文件扩展名或内容检测文件类型
    2. 选择合适的解析策略
    3. 返回Element元素列表

    Args:
        file_path: 待解析文件的完整路径
        strategy: 解析策略，可选值：
            - "auto": 自动选择最佳策略（默认）
            - "fast": 快速策略，跳过复杂元素，速度快
            - "hi_res": 高精度策略，使用图像模型解析（需安装相关依赖）

    Returns:
        DocumentInfo: 包含解析结果的数据类

    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 不支持的文件类型时抛出

    示例:
        >>> info = parse_document_auto("example.pdf")
        >>> print(f"提取文本长度: {info.total_characters}")
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 获取文件扩展名，用于统计
    file_extension = Path(file_path).suffix.lower()

    # 打印解析信息
    print(f"\n{'='*60}")
    print(f"开始解析文件: {file_path}")
    print(f"文件扩展名: {file_extension}")
    print(f"解析策略: {strategy}")
    print(f"{'='*60}")

    try:
        # =====================================
        # 核心解析操作：使用partition自动解析
        # =====================================
        # partition函数是unstructured的核心函数，它会根据文件类型
        # 自动选择合适的解析器，并将解析结果以Element列表形式返回

        elements: List[Element] = partition(
            filename=file_path,
            strategy=strategy  # 解析策略，auto自动选择，fast快速模式，hi_res高精度模式
        )

        # 统计元素类型数量
        element_types: Dict[str, int] = {}
        for element in elements:
            # 获取元素类型名称
            element_type = type(element).__name__
            element_types[element_type] = element_types.get(element_type, 0) + 1

        # 提取文本内容 - 将所有Element的text属性拼接起来
        text_parts = []
        for element in elements:
            if hasattr(element, 'text') and element.text:
                text_parts.append(element.text)

        text_content = "\n\n".join(text_parts)

        # 构建返回结果
        doc_info = DocumentInfo(
            file_path=file_path,
            file_extension=file_extension,
            total_elements=len(elements),
            element_types=element_types,
            text_content=text_content,
            total_characters=len(text_content)
        )

        # 打印解析结果统计
        print(f"解析完成!")
        print(f"  - 元素总数: {doc_info.total_elements}")
        print(f"  - 元素类型: {doc_info.element_types}")
        print(f"  - 文本字符数: {doc_info.total_characters}")

        return doc_info

    except Exception as e:
        print(f"解析文件时出错: {e}")
        raise


def parse_document_by_type(file_path: str, **kwargs) -> List[Element]:
    """
    根据文件类型使用对应的专用解析函数

    当你需要更精细地控制解析过程时，可以直接调用对应的解析函数。
    例如：解析PDF时需要提取图片，就需要使用partition_pdf而非partition。

    Args:
        file_path: 待解析文件的完整路径
        **kwargs: 传递给解析函数的其他参数，如：
            - strategy: 解析策略
            - languages: OCR语言列表，如["eng", "zho"]
            - encoding: 文件编码，如"utf-8"
            - extract_images_in_pdf: 是否提取PDF中的图片
            - include_page_breaks: 是否包含分页符

    Returns:
        List[Element]: Element元素列表

    示例:
        >>> # 解析PDF，提取图片，使用高精度模式
        >>> elements = parse_document_by_type("report.pdf",
        ...     strategy="hi_res",
        ...     extract_images_in_pdf=True,
        ...     languages=["eng", "zho"]
        ... )
    """
    file_extension = Path(file_path).suffix.lower()

    print(f"\n使用专用解析函数: {file_extension} 类型")

    # =====================================
    # 根据文件扩展名选择对应的解析函数
    # =====================================

    if file_extension == ".pdf":
        # PDF文档解析
        # 参数说明：
        # - strategy: "hi_res"使用高精度模式（需要模型支持），"fast"快速模式
        # - extract_images_in_pdf: 是否提取PDF中的图片
        # - extract_image_block_types: 指定要提取的元素类型
        # - languages: OCR识别语言
        elements = partition_pdf(
            filename=file_path,
            strategy=kwargs.get("strategy", "auto"),
            extract_images_in_pdf=kwargs.get("extract_images_in_pdf", True),
            extract_image_block_types=kwargs.get("extract_image_block_types", ["Table", "Image"]),
            languages=kwargs.get("languages", ["eng", "zho"])
        )

    elif file_extension in [".docx", ".doc"]:
        # Word文档解析
        # 参数说明：
        # - include_page_breaks: 是否标识分页符
        elements = partition_docx(
            filename=file_path,
            include_page_breaks=kwargs.get("include_page_breaks", True)
        )

    elif file_extension in [".html", ".htm"]:
        # HTML网页解析
        # 参数说明：
        # - url: 直接从URL解析（此时filename应为None或空）
        # - headers: HTTP请求头
        # - ssl_verify: 是否验证SSL证书
        elements = partition_html(
            filename=file_path if os.path.exists(file_path) else None,
            url=kwargs.get("url"),
            headers=kwargs.get("headers", {"User-Agent": "MyBot"}),
            ssl_verify=kwargs.get("ssl_verify", False)
        )

    elif file_extension == ".md":
        # Markdown文档解析
        # 参数说明：
        # - languages: 语言列表，用于更好地识别文本
        # - include_page_breaks: 是否标识页面断点
        elements = partition_md(
            filename=file_path,
            languages=kwargs.get("languages", ["zho", "eng"]),
            include_page_breaks=kwargs.get("include_page_breaks", False)
        )

    elif file_extension in [".xlsx", ".xls"]:
        # Excel文档解析
        elements = partition_xlsx(
            filename=file_path,
            languages=kwargs.get("languages", ["zho", "eng"])
        )

    elif file_extension == ".csv":
        # CSV文档解析
        elements = partition_csv(
            filename=file_path,
            encoding=kwargs.get("encoding", "utf-8")
        )

    elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        # 图片OCR解析
        # 参数说明：
        # - strategy: "ocr_only"使用OCR识别文字
        # - languages: OCR识别语言
        elements = partition_image(
            filename=file_path,
            strategy=kwargs.get("strategy", "ocr_only"),
            languages=kwargs.get("languages", ["eng", "chi_sim"])
        )

    else:
        # 其他类型文件，使用auto自动检测
        elements = partition(filename=file_path, strategy="auto")

    return elements


def analyze_elements(elements: List[Element]) -> Dict[str, Any]:
    """
    分析Element列表，提取结构和统计信息

    这个函数帮助我们理解解析结果的结构，便于后续处理。

    Args:
        elements: Element元素列表

    Returns:
        Dict: 包含分析结果的字典，包括：
            - total_count: 元素总数
            - type_summary: 各类型数量统计
            - category_summary: 按category分类的数量
            - sample_texts: 每种类型的示例文本（前100字符）

    示例:
        >>> elements = partition(filename="document.pdf")
        >>> analysis = analyze_elements(elements)
        >>> print(f"共有 {analysis['total_count']} 个元素")
    """
    analysis = {
        "total_count": len(elements),
        "type_summary": {},      # 按Python类型统计
        "category_summary": {}, # 按category统计
        "sample_texts": {}      # 每种类型的示例文本
    }

    for element in elements:
        # 获取元素类型名称
        element_type = type(element).__name__
        analysis["type_summary"][element_type] = analysis["type_summary"].get(element_type, 0) + 1

        # 获取元素category
        if hasattr(element, 'category'):
            category = element.category
            analysis["category_summary"][category] = analysis["category_summary"].get(category, 0) + 1

        # 保存示例文本（每种类型只保存一个示例）
        if element_type not in analysis["sample_texts"]:
            if hasattr(element, 'text') and element.text:
                analysis["sample_texts"][element_type] = element.text[:100]

    return analysis


# =============================================================================
# 第四部分：批量文档解析
# =============================================================================

def batch_parse_documents(directory: str, extensions: Optional[List[str]] = None) -> List[DocumentInfo]:
    """
    批量解析目录下指定类型的文档

    Args:
        directory: 目录路径
        extensions: 要处理的文件扩展名列表，如[".pdf", ".docx", ".txt"]
                   如果为None，则处理所有支持的文档类型

    Returns:
        List[DocumentInfo]: 解析结果列表

    示例:
        >>> results = batch_parse_documents("./documents", [".pdf", ".docx"])
        >>> for info in results:
        ...     print(f"{info.file_path}: {info.total_characters} 字符")
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")

    # 默认支持的文件扩展名
    if extensions is None:
        extensions = [".pdf", ".docx", ".doc", ".html", ".htm", ".md",
                      ".xlsx", ".xls", ".csv", ".txt", ".png", ".jpg", ".jpeg"]

    # 收集所有待处理文件
    files_to_process = []
    for ext in extensions:
        # 使用glob搜索匹配的文件
        files_to_process.extend(Path(directory).glob(f"*{ext}"))
        # 同时搜索大写扩展名
        files_to_process.extend(Path(directory).glob(f"*{ext.upper()}"))

    print(f"在目录 {directory} 中找到 {len(files_to_process)} 个文档")

    # 批量解析
    results = []
    for file_path in files_to_process:
        try:
            info = parse_document_auto(str(file_path), strategy="auto")
            results.append(info)
        except Exception as e:
            print(f"  警告：解析失败 {file_path}: {e}")
            continue

    return results


# =============================================================================
# 第五部分：数据清洗示例
# =============================================================================

def clean_text(text: str) -> str:
    """
    文本清洗：去除多余空白字符

    清洗规则：
    1. 将多个连续空格替换为单个空格
    2. 将多个连续换行替换为两个换行（保留段落分隔）
    3. 去除行首行尾空白

    Args:
        text: 原始文本

    Returns:
        str: 清洗后的文本

    示例:
        >>> text = "这是   文本\\n\\n\\n有很多   空格"
        >>> cleaned = clean_text(text)
        >>> print(cleaned)
        "这是 文本\\n\\n有很多 空格"
    """
    import re

    # 去除行首行尾空白
    text = text.strip()

    # 将多个空格替换为单个空格
    text = re.sub(r' +', ' ', text)

    # 将多个换行替换为两个换行（最多保留两个换行作为段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 将\t制表符替换为空格
    text = re.sub(r'\t', ' ', text)

    return text


def remove_html_tags(text: str) -> str:
    """
    去除HTML标签残留

    有些文档解析后可能还残留一些HTML标签，这个函数用于清理。

    Args:
        text: 可能包含HTML标签的文本

    Returns:
        str: 去除HTML标签后的文本

    示例:
        >>> text = "这是<b>粗体</b>文本"
        >>> cleaned = remove_html_tags(text)
        >>> print(cleaned)
        "这是粗体文本"
    """
    import re

    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 清理多余的空白
    text = clean_text(text)

    return text


def normalize_text_for_embedding(text: str) -> str:
    """
    文本规范化，用于向量化前的预处理

    规范化操作：
    1. 转换为小写（对于英文文本）
    2. 去除多余空白
    3. 标准化引号和括号

    Args:
        text: 原始文本

    Returns:
        str: 规范化后的文本
    """
    import re

    # 去除行首行尾空白
    text = text.strip()

    # 标准化空白字符
    text = re.sub(r'[\t\n\r\f\v]+', ' ', text)

    # 标准化引号（将各种引号统一为英文引号）
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # 标准化括号
    text = text.replace('（', '(').replace('）', ')')

    # 去除多余空格
    text = re.sub(r' +', ' ', text)

    return text


# =============================================================================
# 第六部分：元素过滤和处理
# =============================================================================

def filter_elements_by_type(elements: List[Element],
                           include_types: Optional[List[str]] = None,
                           exclude_types: Optional[List[str]] = None) -> List[Element]:
    """
    根据元素类型过滤Element列表

    Args:
        elements: Element列表
        include_types: 要保留的元素类型列表（如["Title", "NarrativeText"]）
        exclude_types: 要排除的元素类型列表

    Returns:
        List[Element]: 过滤后的Element列表

    示例:
        >>> elements = partition(filename="document.pdf")
        >>> # 只保留标题和正文，过滤掉图片和表格
        >>> filtered = filter_elements_by_type(elements,
        ...     include_types=["Title", "NarrativeText"])
    """
    filtered = []

    for element in elements:
        element_type = type(element).__name__

        # 如果指定了包含类型，且当前类型不在包含列表中，则跳过
        if include_types and element_type not in include_types:
            continue

        # 如果指定了排除类型，且当前类型在排除列表中，则跳过
        if exclude_types and element_type in exclude_types:
            continue

        filtered.append(element)

    return filtered


def get_text_by_element_types(elements: List[Element],
                              type_priority: Optional[List[str]] = None) -> str:
    """
    按优先级顺序提取文本内容

    这个函数按照指定的优先级顺序处理元素，便于生成结构化的文本输出。
    例如：先输出所有标题，再输出正文。

    Args:
        elements: Element列表
        type_priority: 类型优先级列表，如["Title", "NarrativeText", "Table"]
                      排在前面的类型会先输出

    Returns:
        str: 按优先级组织的文本内容

    示例:
        >>> elements = partition(filename="document.pdf")
        >>> # 按标题、段落、表格的顺序组织文本
        >>> text = get_text_by_element_types(elements,
        ...     type_priority=["Title", "NarrativeText", "Table"])
    """
    if type_priority is None:
        # 默认优先级
        type_priority = ["Title", "SectionHeader", "NarrativeText", "List", "Table"]

    # 按类型分组元素
    elements_by_type: Dict[str, List[Element]] = {}
    for element in elements:
        element_type = type(element).__name__
        if element_type not in elements_by_type:
            elements_by_type[element_type] = []
        elements_by_type[element_type].append(element)

    # 按优先级顺序组织文本
    text_parts = []
    for element_type in type_priority:
        if element_type in elements_by_type:
            for element in elements_by_type[element_type]:
                if hasattr(element, 'text') and element.text:
                    text_parts.append(element.text)

    # 添加其他未指定的类型
    for element_type, type_elements in elements_by_type.items():
        if element_type not in type_priority:
            for element in type_elements:
                if hasattr(element, 'text') and element.text:
                    text_parts.append(element.text)

    return "\n\n".join(text_parts)


# =============================================================================
# 第七部分：LlamaIndex集成示例
# =============================================================================

def load_documents_with_llamaindex(file_path: str):
    """
    使用LlamaIndex的UnstructuredReader加载文档

    LlamaIndex提供了与unstructured的集成，可以通过UnstructuredReader
    方便地加载文档到LlamaIndex的Document对象中。

    Args:
        file_path: 文件路径

    Returns:
        List[Document]: LlamaIndex的Document列表

    安装依赖:
        pip install llama-index readers

    示例:
        >>> from llama_index.readers.file.unstructured import UnstructuredReader
        >>> reader = UnstructuredReader()
        >>> documents = reader.load_data(file=Path("document.pdf"))
    """
    try:
        from llama_index.readers.file.unstructured import UnstructuredReader
        from pathlib import Path

        # 创建UnstructuredReader实例
        reader = UnstructuredReader()

        # 加载文档
        documents = reader.load_data(file=Path(file_path))

        print(f"LlamaIndex成功加载文档，共 {len(documents)} 个Document对象")
        print(f"第一个Document文本长度: {len(documents[0].text)} 字符")

        return documents

    except ImportError:
        print("错误：LlamaIndex未安装。请运行: pip install llama-index readers")
        return None
    except Exception as e:
        print(f"加载文档时出错: {e}")
        return None


# =============================================================================
# 主函数：演示代码
# =============================================================================

def main():
    """
    主函数：演示各种文档解析功能

    运行前请确保：
    1. 安装了必要的依赖
    2. 在当前目录或指定路径放置了测试文档
    """
    print("=" * 70)
    print("第02节 文档加载解析与预处理 - 代码演示")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 示例1：使用partition自动解析（最常用的方式）
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("示例1：使用partition函数自动解析文档")
    print("=" * 70)

    # 注意：请将此路径替换为你的实际文件路径
    # 这里使用一个示例文件名，实际运行时需要替换为真实文件
    example_file = "example.pdf"  # 可以替换为 .md, .docx, .html 等任何支持的文件

    # 检查文件是否存在，如果存在则解析
    if os.path.exists(example_file):
        doc_info = parse_document_auto(example_file, strategy="auto")
        print(f"\n成功解析 {example_file}")
        print(f"提取文本预览（前200字符）: {doc_info.text_content[:200]}")
    else:
        print(f"\n示例文件 {example_file} 不存在，跳过自动解析示例")
        print("请替换为你的实际文件路径进行测试")

    # -------------------------------------------------------------------------
    # 示例2：使用专用解析函数解析不同格式的文档
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("示例2：使用专用解析函数")
    print("=" * 70)

    # 这里演示如何针对不同文件类型调用专用解析函数
    # 实际使用时取消注释并替换为真实文件路径

    # PDF解析示例
    # pdf_file = "document.pdf"
    # if os.path.exists(pdf_file):
    #     elements = parse_document_by_type(pdf_file, strategy="hi_res", languages=["eng", "zho"])
    #     analysis = analyze_elements(elements)
    #     print(f"PDF解析结果: {analysis['total_count']} 个元素")

    # Word解析示例
    # docx_file = "document.docx"
    # if os.path.exists(docx_file):
    #     elements = parse_document_by_type(docx_file)
    #     analysis = analyze_elements(elements)
    #     print(f"Word解析结果: {analysis['total_count']} 个元素")

    # HTML解析示例（支持从URL直接解析）
    # html_file = "page.html"
    # if os.path.exists(html_file):
    #     elements = parse_document_by_type(html_file)
    # else:
    #     # 也可以直接从URL解析
    #     elements = parse_document_by_type("", url="https://example.com")

    # Markdown解析示例
    # md_file = "document.md"
    # if os.path.exists(md_file):
    #     elements = parse_document_by_type(md_file)
    #     analysis = analyze_elements(elements)
    #     print(f"Markdown解析结果: {analysis['total_count']} 个元素")

    print("提示：请取消注释上述代码中的示例，并替换为真实文件路径进行测试")

    # -------------------------------------------------------------------------
    # 示例3：元素类型分析
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("示例3：元素类型分析")
    print("=" * 70)

    print("""
    Element元素类型说明：
    - Title: 标题元素
    - NarrativeText: 段落文本
    - Table: 表格元素
    - List: 列表元素
    - Image: 图片元素
    - PageBreak: 分页符
    - Formula: 公式元素

    使用analyze_elements函数可以统计各类型元素的数量。
    """)

    # -------------------------------------------------------------------------
    # 示例4：数据清洗
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("示例4：数据清洗示例")
    print("=" * 70)

    # 演示文本清洗
    dirty_text = """

    这是   一段  有很多   空白字符   的文本。


    第二个段落    有多余的    空格。

    """

    cleaned_text = clean_text(dirty_text)
    print(f"原始文本:\n{dirty_text}")
    print(f"\n清洗后:\n{cleaned_text}")

    # 演示HTML标签去除
    html_text = "这是<b>粗体</b>和<i>斜体</i>文本"
    cleaned_html = remove_html_tags(html_text)
    print(f"\nHTML清理示例:")
    print(f"原始: {html_text}")
    print(f"清理后: {cleaned_html}")

    # -------------------------------------------------------------------------
    # 示例5：元素过滤
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("示例5：元素过滤")
    print("=" * 70)

    print("""
    filter_elements_by_type函数可以：
    - include_types: 只保留指定的元素类型
    - exclude_types: 排除指定的元素类型

    get_text_by_element_types函数可以：
    - 按优先级顺序组织文本输出

    示例：只保留标题和正文，过滤掉图片和表格
    filtered = filter_elements_by_type(elements,
                                     include_types=["Title", "NarrativeText"])
    """)

    # -------------------------------------------------------------------------
    # 示例6：LlamaIndex集成
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("示例6：LlamaIndex集成")
    print("=" * 70)

    print("""
    使用LlamaIndex的UnstructuredReader可以方便地将解析结果
    转换为LlamaIndex的Document对象，便于后续的文档分割和向量化。

    示例代码：

    from llama_index.readers.file.unstructured import UnstructuredReader
    from pathlib import Path

    reader = UnstructuredReader()
    documents = reader.load_data(file=Path("document.pdf"))

    安装依赖: pip install llama-index readers
    """)

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    # 调用主函数
    main()
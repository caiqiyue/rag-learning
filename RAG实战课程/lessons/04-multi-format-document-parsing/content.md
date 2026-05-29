# 第四课：多格式文档解析

## 本节概述

本节课将学习企业级 RAG 系统中的第一步：如何将各种格式的文档（PDF、Word、Excel、PPT、图片等）解析为可处理的文本。你将掌握：

- 常见文档格式的解析方法
- PDF 布局分析与文本提取
- OCR 文字识别技术
- 多格式文档的统一处理方案

**学习时长**：约 3-4 小时

---

## 4.1 文档解析概述

### 4.1.1 为什么需要文档解析？

RAG 系统的第一步是"消化"各种来源的文档。企业环境中的文档格式多种多样：

```
企业文档格式分布（典型）：
├── PDF (40%) - 官方文档、合同、报告
├── Word (25%) - 内部制度、操作手册
├── Excel (15%) - 数据报表、财务报表
├── PPT (10%) - 演示文稿、培训材料
├── 图片 (5%) - 扫描件、截图
└── 其他 (5%) - HTML、Markdown、邮件
```

**文档解析的挑战**：

| 格式 | 主要挑战 |
|------|----------|
| PDF | 布局复杂、表格嵌套、图片混排 |
| Word | 样式多样、目录结构、修订痕迹 |
| Excel | 公式、多Sheet、合并单元格 |
| PPT | 幻灯片顺序、动画内容、备注 |
| 图片 | 需要 OCR 识别 |

### 4.1.2 解析技术全景图

```
┌─────────────────────────────────────────────────────────────┐
│                      文档解析技术栈                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PDF ──→ PyMuPDF/Pdfplumber ──→ 布局分析 ──→ 文本提取      │
│                                                             │
│  Word ──→ python-docx ──────────→ 段落/表格提取            │
│                                                             │
│  Excel ──→ openpyxl/pandas ──────→ 结构化数据提取          │
│                                                             │
│  PPT ───→ python-pptx ──────────→ 幻灯片内容提取           │
│                                                             │
│  图片 ──→ OCR (pytesseract/       文字识别                 │
│             RapidOCR)                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4.2 PDF 文档解析

### 4.2.1 PDF 解析基础

PDF 是企业文档最常见的格式。PDF 解析的难点在于：

```
PDF 解析难点：
1. 文本提取 ≠ 布局保留：PDF 保存的是"如何渲染"，不是"文本内容"
2. 表格识别难：PDF 中的表格是线条和文字的组合，不是结构化数据
3. 图片混排：文字和图片交错，提取时容易混乱
4. 扫描件：本质是图片，需要 OCR 处理
```

### 4.2.2 PyMuPDF（推荐）

**特点**：速度快、功能全、支持布局分析

```python
import fitz  # PyMuPDF

def extract_text_pdf_pymupdf(pdf_path: str) -> list:
    """
    使用 PyMuPDF 提取 PDF 文本
    返回每页的文本内容
    """
    doc = fitz.open(pdf_path)
    pages_content = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 方法1：纯文本提取（忽略布局）
        text = page.get_text("text")  # "text" | "blocks" | "dict"
        
        # 方法2：提取文本块（保留位置信息）
        blocks = page.get_text("blocks")
        
        # 方法3：提取为字典（包含样式信息）
        page_dict = page.get_text("dict")
        
        pages_content.append({
            "page_num": page_num + 1,
            "text": text,
            "blocks": blocks,
            "page_dict": page_dict
        })
    
    doc.close()
    return pages_content

# 使用示例
pages = extract_text_pdf_pymupdf("产品手册.pdf")
print(f"共 {len(pages)} 页")
print(f"第1页前200字：{pages[0]['text'][:200]}")
```

### 4.2.3 Pdfplumber（适合表格）

**特点**：表格提取能力强、适合财务/数据类 PDF

```python
import pdfplumber

def extract_tables_pdfplumber(pdf_path: str) -> list:
    """
    使用 Pdfplumber 提取 PDF 文本和表格
    """
    all_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 提取文本
            text = page.extract_text()
            
            # 提取表格
            tables = page.extract_tables()
            
            # 提取布局信息（页面上所有字符的位置）
            chars = page.chars
            
            all_content.append({
                "page_num": page_num + 1,
                "text": text,
                "tables": tables,
                "chars": chars
            })
    
    return all_content

# 处理表格的特殊情况
def extract_table_with_style(page):
    """提取表格并保留样式信息"""
    tables = page.extract_tables()
    
    formatted_tables = []
    for table in tables:
        # 表格可能返回 None（表示未检测到表格）
        if table:
            # 将表格转换为带表头的字典列表
            headers = table[0]  # 第一行通常是表头
            rows = table[1:]    # 后续行是数据
            
            formatted = []
            for row in rows:
                if row:  # 过滤空行
                    formatted.append(dict(zip(headers, row)))
            
            formatted_tables.append(formatted)
    
    return formatted_tables
```

### 4.2.4 布局分析进阶

对于复杂布局的 PDF，需要进行布局分析：

```python
def extract_with_layout_analysis(pdf_path: str):
    """
    带布局分析的 PDF 解析
    区分标题、段落、表格、图片等不同元素
    """
    import fitz
    
    doc = fitz.open(pdf_path)
    document_structure = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 获取页面上的所有块
        blocks = page.get_text("dict")["blocks"]
        
        page_elements = []
        for block in blocks:
            if "lines" in block:  # 文本块
                block_type = classify_text_block(block)
                page_elements.append({
                    "type": block_type,  # "title" | "paragraph" | "table"
                    "bbox": block["bbox"],
                    "content": block
                })
            elif "image" in block:  # 图片块
                page_elements.append({
                    "type": "image",
                    "bbox": block["bbox"],
                    "width": block.get("width"),
                    "height": block.get("height")
                })
        
        document_structure.append({
            "page_num": page_num + 1,
            "elements": page_elements
        })
    
    return document_structure

def classify_text_block(block):
    """根据字体大小、位置等特征判断文本块类型"""
    lines = block.get("lines", [])
    if not lines:
        return "unknown"
    
    # 检查是否是大字体（可能是标题）
    # 检查是否在页面顶部（可能是章节标题）
    # 检查是否有特殊样式（加粗、斜体）
    
    return "paragraph"  # 默认返回段落
```

---

## 4.3 Word 文档解析

### 4.3.1 python-docx 基础

```python
from docx import Document

def extract_text_word(docx_path: str) -> dict:
    """
    使用 python-docx 提取 Word 文档内容
    """
    doc = Document(docx_path)
    content = {
        "paragraphs": [],
        "tables": []
    }
    
    # 提取段落
    for para in doc.paragraphs:
        if para.text.strip():  # 跳过空段落
            content["paragraphs"].append({
                "text": para.text,
                "style": para.style.name,  # 样式（标题、正文等）
                "level": para._element.getpPr_sprm()  # 嵌套级别
            })
    
    # 提取表格
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        content["tables"].append(table_data)
    
    return content

# 使用示例
word_content = extract_text_word("技术规范.docx")
print(f"段落数：{len(word_content['paragraphs'])}")
print(f"表格数：{len(word_content['tables'])}")
```

### 4.3.2 处理复杂 Word 文档

```python
def extract_word_advanced(docx_path: str) -> dict:
    """
    高级 Word 解析：提取更多结构信息
    """
    from docx import Document
    from docx.oxml.table import CT_Tc
    from docx.oxml.text.paragraph import CT_P
    
    doc = Document(docx_path)
    full_text = []
    elements = []
    
    for element in doc.element.body:
        if isinstance(element, CT_P):
            # 这是一个段落
            para = element._p
            text = para.text
            style = para.getparent().getparent()
            
            elements.append({
                "type": "paragraph",
                "text": text,
                "style": style
            })
            full_text.append(text)
            
        elif isinstance(element, CT_Tc):
            # 这是一个表格单元格
            elements.append({
                "type": "table_cell",
                "text": element.text
            })
    
    return {
        "full_text": "\n".join(full_text),
        "elements": elements,
        "word_count": len("".join(full_text))
    }
```

---

## 4.4 Excel 文档解析

### 4.4.1 openpyxl 基础

```python
import openpyxl

def extract_excel(excel_path: str) -> dict:
    """
    使用 openpyxl 提取 Excel 内容
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    
    content = {
        "sheets": {},
        "total_rows": 0
    }
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_data = []
        
        for row in ws.iter_rows(values_only=True):
            # 过滤空行
            if any(cell is not None for cell in row):
                rows_data.append(list(row))
                content["total_rows"] += 1
        
        content["sheets"][sheet_name] = {
            "rows": rows_data,
            "dimensions": ws.dimensions,
            "max_row": ws.max_row,
            "max_column": ws.max_column
        }
    
    return content

# 使用示例
excel_data = extract_excel("销售报表.xlsx")
print(f"工作表：{list(excel_data['sheets'].keys())}")
print(f"总行数：{excel_data['total_rows']}")
```

### 4.4.2 使用 pandas 读取

```python
import pandas as pd

def extract_excel_pandas(excel_path: str) -> dict:
    """
    使用 pandas 读取 Excel，适合数据分析场景
    """
    content = {}
    
    # 读取所有工作表
    excel_file = pd.ExcelFile(excel_path)
    
    for sheet_name in excel_file.sheet_names:
        # 读取单个工作表
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        content[sheet_name] = {
            "dataframe": df,
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "head": df.head(10).to_dict(),  # 前10行预览
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    
    return content

# 读取特定列
def extract_column(excel_path: str, sheet: str, column: str) -> list:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    return df[column].dropna().tolist()
```

---

## 4.5 PPT 文档解析

```python
from pptx import Presentation

def extract_ppt(ppt_path: str) -> dict:
    """
    使用 python-pptx 提取 PPT 内容
    """
    prs = Presentation(ppt_path)
    
    content = {
        "slides": [],
        "total_slides": len(prs.slides)
    }
    
    for slide_num, slide in enumerate(prs.slides):
        slide_content = {
            "slide_num": slide_num + 1,
            "title": None,
            "content": [],
            "images": []
        }
        
        # 提取标题
        if slide.shapes.title:
            slide_content["title"] = slide.shapes.title.text
        
        # 提取所有形状的内容
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        slide_content["content"].append(text)
            
            # 检查是否有图片
            if hasattr(shape, "image"):
                slide_content["images"].append({
                    "name": shape.name,
                    "position": (shape.left, shape.top)
                })
        
        content["slides"].append(slide_content)
    
    return content

# 使用示例
ppt_data = extract_ppt("培训课件.pptx")
for slide in ppt_data["slides"][:3]:  # 前3张幻灯片
    print(f"第{slide['slide_num']}页：{slide['title']}")
    print(f"内容：{slide['content'][:2]}...")  # 前2条内容
```

---

## 4.6 OCR 文字识别

### 4.6.1 什么时候需要 OCR？

```
需要 OCR 的场景：
├── 扫描件 PDF（本质是图片）
├── 截图、照片中的文字
├── 老旧文档（纸质扫描）
└── 图片中的表格
```

### 4.6.2 pytesseract（基础 OCR）

```python
import pytesseract
from PIL import Image

def ocr_image(image_path: str) -> str:
    """
    使用 pytesseract 进行 OCR
    需要先安装 tesseract OCR 引擎
    """
    image = Image.open(image_path)
    
    # 基础 OCR
    text = pytesseract.image_to_string(
        image,
        lang='chi_sim+eng'  # 中文+英文
    )
    
    return text

def ocr_pdf_scan(pdf_path: str) -> list:
    """
    OCR 识别扫描件 PDF
    """
    import fitz
    
    doc = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 将页面转换为图片
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 分辨率
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # OCR
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        
        results.append({
            "page_num": page_num + 1,
            "text": text
        })
    
    return results
```

### 4.6.3 RapidOCR（推荐中文场景）

```python
from rapidocr_onnxruntime import RapidOCR

def ocr_rapid(image_path: str) -> dict:
    """
    使用 RapidOCR 进行中文 OCR
    速度更快，中文识别率更高
    """
    ocr_engine = RapidOCR()
    
    result, elapsed = ocr_engine(image_path)
    
    # 解析结果
    text_lines = []
    for line in result:
        bbox, text, confidence = line
        text_lines.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox
        })
    
    return {
        "full_text": "\n".join([line["text"] for line in text_lines]),
        "lines": text_lines
    }
```

---

## 4.7 统一文档解析接口

为了简化多格式文档的处理，创建一个统一的解析接口：

```python
from pathlib import Path
from typing import Union

class UniversalDocumentLoader:
    """
    统一文档加载器
    自动识别文件格式并调用相应的解析器
    """
    
    def __init__(self):
        self.loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_word,
            ".doc": self._load_word,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".pptx": self._load_ppt,
            ".ppt": self._load_ppt,
            ".txt": self._load_txt,
            ".md": self._load_markdown,
            ".png": self._load_image_ocr,
            ".jpg": self._load_image_ocr,
            ".jpeg": self._load_image_ocr,
        }
    
    def load(self, file_path: Union[str, Path]) -> dict:
        """
        加载任意支持的文档格式
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix not in self.loaders:
            raise ValueError(f"不支持的格式: {suffix}")
        
        return self.loaders[suffix](path)
    
    def _load_pdf(self, path: Path) -> dict:
        # 使用 PyMuPDF
        import fitz
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        return {"content": text, "source": str(path)}
    
    def _load_word(self, path: Path) -> dict:
        from docx import Document
        doc = Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return {"content": text, "source": str(path)}
    
    def _load_excel(self, path: Path) -> dict:
        import pandas as pd
        df = pd.read_excel(path)
        text = df.to_string()
        return {"content": text, "source": str(path)}
    
    def _load_ppt(self, path: Path) -> dict:
        from pptx import Presentation
        prs = Presentation(path)
        text = "\n".join([
            shape.text for slide in prs.slides 
            for shape in slide.shapes if shape.has_text_frame
        ])
        return {"content": text, "source": str(path)}
    
    def _load_txt(self, path: Path) -> dict:
        return {"content": path.read_text(encoding="utf-8"), "source": str(path)}
    
    def _load_markdown(self, path: Path) -> dict:
        return {"content": path.read_text(encoding="utf-8"), "source": str(path)}
    
    def _load_image_ocr(self, path: Path) -> dict:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        result, _ = ocr(str(path))
        text = "\n".join([line[1] for line in result]) if result else ""
        return {"content": text, "source": str(path)}

# 使用示例
loader = UniversalDocumentLoader()

# 自动识别格式并解析
for file in Path("知识库").glob("**/*"):
    if file.is_file():
        try:
            doc = loader.load(file)
            print(f"✓ {file.name}: {len(doc['content'])} 字符")
        except Exception as e:
            print(f"✗ {file.name}: {e}")
```

---

## 本节总结

### 核心要点

1. **PDF 解析**：PyMuPDF（速度快）、Pdfplumber（表格强）
2. **Word 解析**：python-docx，提取段落和表格
3. **Excel 解析**：pandas，适合数据分析
4. **PPT 解析**：python-pptx，按幻灯片提取
5. **OCR**：pytesseract（通用）、RapidOCR（中文优先）

### 代码文件

本节课配套代码：`code/doc_parser.py`

---

## 思考题

1. 为什么 PDF 解析比 Word 解析更复杂？
2. 如果要解析一个包含图片和表格的扫描件 PDF，流程应该是什么样的？
3. 为什么需要对不同格式的文档使用不同的解析器，而不是统一的方案？

### 下节预告

下一节课我们将学习文档切分策略，掌握如何将长文档合理地分割成小块。

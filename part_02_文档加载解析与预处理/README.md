# 第02节 文档加载解析与预处理

## 学习目标

1. 掌握Python中主流文档格式的加载方法
2. 理解不同文档格式（PDF、Word、HTML、TXT）的解析差异
3. 能够处理文档编码和结构识别问题

## 关联知识框架知识点

- **kp_003**: 文档加载与解析 - 文档加载器负责从多种数据源（PDF、Word、TXT、Markdown、HTML、Excel等）提取纯文本内容。unstructured.io库提供了统一的文档解析解决方案，支持自动检测文件类型并解析。
- **kp_010**: 数据清洗与预处理 - 数据清洗是文档切分前的重要预处理步骤，针对不同数据类型采用不同策略：纯文本需去除空白、修复断词；表格数据需处理NaN值；图像需去噪；代码块需处理注释和缩进。
- **kp_032**: PDF/Document Processing文档处理 - 企业文档常以PDF、Word等格式存在，需要专门的解析处理。包括：布局分析、OCR、表格结构化提取、跨页表格处理。

---

## 一、文档加载解析基础

### 1.1 为什么需要文档解析？

在企业实际场景中，数据源种类繁多：
- 金融领域的财报、报表（PDF）
- 法律合同、协议（Word）
- 网页内容（HTML）
- 配置文件（TXT、JSON）

这些文档格式各异、结构复杂，直接使用原始文档进行向量化，效果会很差。文档解析的核心目标就是**将各种格式的文档转换为统一的纯文本格式**，为后续的文档分割和向量化奠定基础。

### 1.2 常见文档格式与解析挑战

| 文档类型 | 扩展名 | 解析难点 |
|----------|--------|----------|
| PDF | .pdf | 布局复杂、扫描件需要OCR、跨页表格 |
| Word | .doc/.docx | 格式信息丢失、特殊符号处理 |
| Excel | .xlsx/.xls | 多Sheet、合并单元格、公式 |
| HTML | .html/.htm | 标签噪声、JavaScript代码 |
| Markdown | .md | 语法符号、代码块 |
| 纯文本 | .txt | 编码问题、断词 |

### 1.3 unstructured.io库简介

**unstructured.io** 是目前最流行的文档解析库，提供了统一的接口来解析各种格式的文档。

核心组件：
- `partition` - 自动检测文件类型并解析
- `partition_pdf` - PDF专用解析
- `partition_docx` - Word文档解析
- `partition_html` - 网页解析
- `partition_md` - Markdown解析
- `partition_xlsx` - Excel解析
- `partition_csv` - CSV解析
- `partition_image` - 图片OCR

---

## 二、核心概念讲解

### 2.1 文档加载器的工作原理

文档加载器的核心任务是从原始文档中提取文本内容，同时尽可能保留文档的结构信息（如标题、表格、列表等）。

工作流程：
```
原始文档 → 文件类型检测 → 格式-specific解析 → Element元素列表 → 统一文本输出
```

### 2.2 Element元素类型

解析结果以`Element`对象列表形式返回，每个Element代表文档中的一个逻辑单元：

| Element类型 | 说明 | 示例 |
|-------------|------|------|
| Title | 标题 | 文档的各级标题 |
| NarrativeText | 段落文本 | 正文内容 |
| Table | 表格 | 数据表格 |
| List | 列表 | 有序/无序列表 |
| Image | 图片 | 内嵌图片 |
| PageBreak | 分页符 | 页面边界 |
| Formula | 公式 | 数学公式 |

### 2.3 解析策略

针对不同类型的文档，unstructured支持多种解析策略：

| 策略 | 适用场景 | 特点 |
|------|----------|------|
| `auto` | 通用场景 | 自动选择最佳策略 |
| `fast` | 快速处理 | 跳过复杂元素，速度快100倍 |
| `hi_res` | PDF/图片 | 高精度解析，需要模型支持 |
| `ocr_only` | 扫描件 | 仅使用OCR识别文字 |

### 2.4 通用参数

| 参数 | 说明 |
|------|------|
| `encoding` | 指定字符编码（如utf-8、gbk） |
| `include_page_breaks` | 是否输出分页符 |
| `strategy` | 解析策略选择 |
| `languages` | OCR语言列表 |
| `metadata_include/exclude` | 元数据包含/排除控制 |

---

## 三、数据清洗与预处理

### 3.1 为什么要清洗？

原始解析结果往往存在以下问题：
- 多余的空白字符（多个空格、换行）
- 断词问题（英文单词被截断）
- 特殊字符残留（HTML标签、Markdown语法）
- 编码混乱（中文乱码）

### 3.2 不同数据类型的清洗策略

| 数据类型 | 清洗策略 |
|----------|----------|
| **纯文本** | 去除多余空白、修复断词 |
| **表格数据** | 处理NaN值、规范化格式 |
| **图像文字** | 去噪、二值化 |
| **代码块** | 处理注释、保持缩进 |

### 3.3 常见清洗操作

1. **去除多余空白**：将多个空格/换行压缩为单个
2. **去除特殊字符**：移除HTML标签、Markdown语法符号
3. **编码规范化**：统一转换为UTF-8
4. **文本规范化**：大小写统一、标点标准化

---

## 四、企业级文档处理流程

```
┌─────────────────────────────────────────────────────────────┐
│                    企业文档处理流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 文档收集    →    2. 格式检测    →    3. 解析提取         │
│       │                 │                  │               │
│       │                 ▼                  ▼               │
│  PDF/Word/          自动检测          Element列表            │
│  HTML/TXT          文件类型          结构化输出             │
│                                                             │
│       │                 │                  │               │
│       ▼                 ▼                  ▼               │
│                                                             │
│  4. 数据清洗    →    5. 结构识别    →    6. 输出标准化        │
│       │                 │                  │               │
│ 去除噪声/        识别标题/表格/    统一文本格式    ──→  后续   │
│ 编码规范化        段落结构                      分割向量化    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、代码案例

### 5.1 代码路径

```
E:/ai-learning/09-agent-engineering/rag-learning/part_02_文档加载解析与预处理/02_codes/document_loader.py
```

### 5.2 运行说明

1. **依赖安装**

```bash
pip install unstructured python-magic-bin pdf2image pytesseract poppler
```

> 注意：Windows系统需要安装`python-magic-bin`（libmagic的Python封装），其他系统需安装`libmagic`。

2. **可选依赖（用于OCR和高精度PDF解析）**

```bash
pip install pytesseract tesseract-ocr
```

3. **Tesseract OCR安装（Windows）**

下载安装：https://github.com/UB-Mannheim/tesseract/wiki/Download

4. **Poppler安装（Windows，用于PDF转图像）**

下载：https://github.com/oschwartz10612/Poppler-Release/releases

5. **运行方式**

```bash
# 确保测试文档存在，或修改代码中的文件路径
python E:/ai-learning/09-agent-engineering/rag-learning/part_02_文档加载解析与预处理/02_codes/document_loader.py
```

6. **代码说明**

该代码演示了：
- 使用`partition`函数自动解析多种文档格式
- 专用解析函数的使用（PDF、Word、HTML、Markdown、Excel、CSV、图片）
- 解析结果的Element类型分析
- 批量文档解析功能
- 数据清洗与预处理示例

---

## 六、总结

| 知识点 | 关键内容 |
|--------|----------|
| 文档加载器 | 负责从多种数据源提取纯文本内容 |
| unstructured.io | 统一文档解析解决方案，支持自动检测文件类型 |
| Element元素 | 文档解析的基本单元，包含text、category、metadata |
| 解析策略 | auto/fast/hi_res/ocr_only，适用于不同场景 |
| 数据清洗 | 针对不同数据类型（文本/表格/图像/代码）采用不同策略 |
| 企业流程 | 文档收集 → 格式检测 → 解析提取 → 数据清洗 → 输出标准化 |

---

## 课后思考

1. 为什么PDF文档解析比Word文档解析更具挑战性？
2. 在什么场景下应该选择`fast`策略而非`hi_res`策略？
3. 数据清洗与文档切分这两个预处理步骤，哪个应该先执行？为什么？
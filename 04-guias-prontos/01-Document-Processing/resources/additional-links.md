# Resources - Document Processing

## 📚 Documentação Oficial

### LangChain
- **Document Loaders:** https://python.langchain.com/docs/modules/data_connection/document_loaders/
- **Text Splitters:** https://python.langchain.com/docs/modules/text_splitters/
- **Examples:** https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/document_loaders

### Loaders by Format

#### PDF
- **PyPDF:** https://pypdf.readthedocs.io/
- **PyMuPDF:** https://pymupdf.readthedocs.io/
- **PDFPlumber:** https://github.com/jsvine/pdfplumber

#### Documents
- **python-docx:** https://python-docx.readthedocs.io/
- **python-pptx:** https://python-pptx.readthedocs.io/

#### Web
- **BeautifulSoup:** https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- **Selenium:** https://selenium-python.readthedocs.io/
- **Playwright:** https://playwright-python.readthedocs.io/

#### Data
- **pandas:** https://pandas.pydata.org/docs/
- **openpyxl:** https://openpyxl.readthedocs.io/

## 🛠️ Ferramentas e Bibliotecas

### General Purpose
1. **LangChain Community**
   - URL: https://github.com/langchain-ai/langchain
   - 160+ document loaders
   - Text splitters
   - Text cleaners

2. **Unstructured**
   - URL: https://github.com/Unstructured-IO/unstructured
   - Universal document parser
   - Supports 20+ formats
   - Format detection

3. **DocumentAI**
   - Google Cloud
   - OCR and parsing
   - Structured output

### PDF Processing
4. **PyPDF2/PyPDF4**
   - Pure Python
   - Basic operations

5. **pdfminer.six**
   - Advanced PDF parsing
   - Text extraction

6. **Tesseract OCR**
   - Open-source OCR
   - 100+ languages
   - Command line + Python

### Web Scraping
7. **Scrapy**
   - Framework for web scraping
   - Async support
   - Built-in data extraction

8. **Readability**
   - Extract main content
   - Remove ads/navigation
   - Fast and simple

9. **Newspaper3k**
   - Article extraction
   - Multi-language
   - NLP features

### Specialized Parsers
10. **Tabula-py**
    - PDF tables to CSV
    - Uses Tabula Java

11. **Mammoth**
    - DOCX to HTML/Markdown
    - Preserves styling

12. **Pandoc**
    - Universal converter
    - 40+ formats
    - Command line tool

## 📊 Benchmarks e Comparações

### PDF Loaders Performance
```
                Load Time    Text Quality    Memory Usage
PyPDFLoader     2.3s        90%            150 MB
PDFPlumber      3.1s        95%            180 MB
PyMuPDF         1.8s        85%            200 MB
Unstructured    4.2s        92%            220 MB
```

### File Size Limits
```
Loader           Max File Size    Batch Support
TextLoader       Unlimited        ✅
PyPDFLoader      500 MB           ❌
Docx2txtLoader   100 MB           ❌
CSVLoader        1 GB             ✅
WebBaseLoader    N/A              ❌
```

### Accuracy by Format
```
Format          Accuracy    Notes
TXT             100%        Perfect
DOCX            98%         Tables may vary
PDF (text)      90%         Formatting loss
PDF (scanned)   75%         OCR dependent
HTML            85%         JS content missing
CSV             100%        Structured data
Markdown        95%         Headers preserved
```

## 🎓 Tutoriais e Cursos

### Video Tutorials
1. **LangChain Document Loaders (YouTube)**
   - Channel: LangChain
   - URL: https://youtube.com/playlist?list=PLq2IkYpAHPIWRj9AU8-Pb4qzgFhncKoG
   - Duration: 3 videos

2. **Document Processing in RAG**
   - Channel: AI Engineering
   - Advanced techniques
   - Real-world examples

### Blog Posts
3. **"Document Processing Best Practices"** - Towards Data Science
   - URL: https://towardsdatascience.com/document-processing
   - Common pitfalls
   - Optimization tips

4. **"OCR for Scanned Documents"** - Machine Learning Mastery
   - URL: https://machinelearningmastery.com/ocr
   - Tesseract tutorial
   - Accuracy tips

### Courses
5. **"Document AI"** - Coursera (University of Michigan)
   - OCR, parsing, extraction
   - 4 weeks

6. **"Web Scraping with Python"** - Udemy
   - BeautifulSoup, Scrapy
   - Practical projects

## 📖 Papers

### Document Understanding
1. **"LayoutLM: Pre-training of Text and Layout for Document Image Understanding"**
   - Microsoft Research
   - Document structure understanding

2. **"DocFormer: Multi-modal Transformer for Document Understanding"**
   - Transformer architecture
   - Visual + textual features

### PDF Processing
3. **"PDF to Structured Data Extraction"**
   - University of Amsterdam
   - Table extraction

4. **"OCR Accuracy Assessment"**
   - NIST
   - Evaluation metrics

## 🔧 Utilities

### Text Cleaning
1. **ftfy** - Fix text encoding
   - https://ftfy.readthedocs.io/

2. **chardet** - Character encoding detector
   - https://github.com/chardet/chardet

3. **textblob** - Text processing
   - https://textblob.readthedocs.io/

### Document Conversion
4. **Pandoc** - Universal converter
   - https://pandoc.org/
   - Command line

5. **LibreOffice** - Office documents
   - Headless mode
   - Batch conversion

### Web Content
6. **readability-lxml** - Content extraction
   - https://github.com/buriy/python-readability

7. **newspaper3k** - Article parser
   - https://newspaper.readthedocs.io/

## 🌍 Comunidades

### Forums
1. **Stack Overflow**
   - Tag: langchain, document-loaders
   - Active Q&A

2. **Reddit - r/LangChain**
   - Tips and tricks
   - User experiences

3. **Discord - LangChain**
   - Real-time help
   - Community projects

### GitHub
4. **LangChain Discussions**
   - Feature requests
   - Bug reports

5. **Awesome Document Processing**
   - Curated list
   - https://github.com/rokeedoc/document-processing

## 📈 Datasets

### Benchmark Datasets
1. **DocVQA**
   - Document question answering
   - Real documents

2. **Kleister**
   - Long documents
   - Information extraction

3. **CORD-19**
   - Research papers
   - PDF format

### Sample Documents
4. **Sample PDFs**
   - Government reports
   - Academic papers
   - https://github.com/jakevdp/PDFData

5. **Web Corpus**
   - Web text extraction
   - https://commoncrawl.org/

## 🎯 Best Practices Guides

### From Companies
1. **OpenAI - Data Preparation**
   - Tokenization
   - Format handling
   - https://platform.openai.com/docs/guides/data-preparation

2. **Anthropic - Long Context**
   - Document summarization
   - Context management
   - https://docs.anthropic.com/claude/docs/building-applications

### Community
3. **RAG Newsletter**
   - Monthly updates
   - Best practices
   - https://newsletter.ragflow.com/

4. **AI Engineering Handbook**
   - Chapter: Document Processing
   - Real-world patterns
   - https://www.promptingguide.ai/engineering

## 🔍 Search Tools

### Semantic Search for Documents
1. **Qdrant**
   - Vector search
   - Metadata filtering

2. **Weaviate**
   - Hybrid search
   - Document classification

3. **Pinecone**
   - Cloud vector DB
   - High performance

## 📊 Monitoring & Observability

### Document Processing Metrics
1. **Processing Time**
   - Per document
   - Per format

2. **Success Rate**
   - Loaded successfully
   - Quality score

3. **Error Distribution**
   - By format
   - By error type

### Tools
4. **Prometheus**
   - Metrics collection
   - Grafana dashboards

5. **LangSmith**
   - Tracing
   - Quality evaluation

## 🤝 Contributing

### Open Source
1. **LangChain**
   - Contribute loaders
   - Bug fixes
   - Documentation

2. **Unstructured**
   - New format parsers
   - Improvements

### Formats to Support
3. **LaTeX** - Academic papers
4. **ePub** - E-books
5. **PowerPoint** - Presentations
6. **Images** - OCR integration
7. **Audio** - Transcription
8. **Video** - Frame extraction

## 📅 Events

### Conferences
1. **Document AI Summit**
   - Annual
   - Industry talks

2. **ACL Conference**
   - Academic papers
   - Latest research

### Webinars
3. **LangChain Webinars**
   - Monthly
   - New features

4. **Vector Institute**
   - RAG workshops
   - Document processing

## 📦 Libraries Cheat Sheet

```python
# PDF
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader

# Office
from langchain.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PowerPointLoader

# Web
from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import SeleniumURLLoader

# Data
from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import ExcelLoader

# Text
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
```

## 🔗 Quick Links

- [LangChain Docs](https://python.langchain.com/docs)
- [Text Splitters](https://python.langchain.com/docs/modules/text_splitters)
- [API Reference](https://api.python.langchain.com)
- [GitHub Issues](https://github.com/langchain-ai/langchain/issues)
- [Community Discord](https://discord.gg/langchain)

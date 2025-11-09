# Relatório de Pesquisa: Seção 01 - Document Processing

### Data: 09/11/2025
### Status: Fase 1 - Foundation

---

## 1. RESUMO EXECUTIVO

Document Processing é a primeira etapa crítica do pipeline RAG, responsável por transformar documentos brutos em chunks indexáveis. A qualidade do preprocessing determina diretamente a qualidade do retrieval e geração. O pipeline padrão: `Load → Split → Store`.

**Insights Chave:**
- LangChain oferece 160+ document loaders
- Text splitting é essencial (chunk_size=1000, chunk_overlap=200)
- Diferentes formatos requerem abordagens específicas
- Metadata preservation é crucial para explicabilidade

---

## 2. FONTES PRIMÁRIAS

### 2.1 LangChain Documentation
**URL**: https://docs.langchain.com/oss/python/langchain/rag
- **Concept**: 3 etapas do pipeline: Load, Split, Store
- **Document Loaders**: 160+ integrações
- **Text Splitters**: RecursiveCharacterTextSplitter recomendado
- **Interface**: Document objects com page_content e metadata

### 2.2 Document Loaders Overview
- **WebBaseLoader**: Usa urllib + BeautifulSoup para HTML
- **160+ integrações** para diferentes formatos
- **Output**: Lista de Document objects
- **Customization**: bs_kwargs, SoupStrainer para filtrar elementos

---

## 3. PIPELINE DE PROCESSAMENTO

### 3.1 Etapas do Pipeline

```python
# 1. Load
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_paths=("url",), bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content",))))
docs = loader.load()

# 2. Split
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = splitter.split_documents(docs)

# 3. Store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=splits)
```

### 3.2 Document Object Structure

```python
class Document:
    page_content: str  # Texto do documento
    metadata: dict     # Metadata (fonte, título, etc.)
```

---

## 4. FORMAT HANDLING

### 4.1 Supported Formats

| Formato | Loader Principal | Bibliotecas | Complexidade | Suporte |
|---------|------------------|-------------|--------------|---------|
| **PDF** | PDFLoader | PyMuPDF, pdfplumber | Média | ✅ Estruturado e Scanned |
| **DOCX** | Docx2txtLoader | python-docx | Baixa | ✅ Texto, Tabelas, Imagens |
| **HTML** | WebBaseLoader | BeautifulSoup | Baixa | ✅ Páginas Web |
| **TXT** | TextLoader | Built-in | Baixa | ✅ Texto Puro |
| **Markdown** | UnstructuredMarkdownLoader | Unstructured | Baixa | ✅ Texto, Headers |
| **PPTX** | PowerPointLoader | python-pptx | Média | ✅ Texto, Tabelas |
| **CSV/Excel** | CSVLoader/ExcelLoader | pandas | Baixa | ✅ Dados Tabulares |
| **JSON** | JSONLoader | Built-in | Baixa | ✅ Dados Estruturados |

### 4.2 Format-Specific Considerations

#### PDF
- **Estruturado**: Texto extraível diretamente
- **Scanned**: Requer OCR (Tesseract, EasyOCR)
- **Tabelas**: Bibliotecas especializadas
- **Images**: Extração de metadados
- **Limitações**: Layout, fontes, qualidade

#### DOCX
- **Texto**: Automático
- **Tabelas**: Estrutura preservada
- **Imagens**: Extração com metadados
- **Headers/Footers**: Preservação opcional
- **Limitações**: Formatação complexa

#### HTML
- **Parsing**: BeautifulSoup customization
- **Filtros**: SoupStrainer para elementos específicos
- **Limpagem**: Remoção de scripts, styles
- **Extração**: Headers, links, metadados

#### TXT
- **Encoding**: UTF-8, ISO-8859-1
- **Delimitadores**: Parágrafos, quebras de linha
- **Limpeza**: Caracteres especiais
- **Metadados**: Timestamp, autor

---

## 5. TEXT SPLITTING

### 5.1 RecursiveCharacterTextSplitter (Recomendado)

**Características:**
- Divide recursivamente usando separadores comuns
- Quebras de linha → parágrafos → sentenças → palavras
- Garante chunks do tamanho especificado

**Parâmetros:**
- `chunk_size`: Tamanho alvo (padrão: 1000)
- `chunk_overlap`: Sobreposição (padrão: 200)
- `add_start_index`: Rastrear índice original (padrão: False)

**Exemplo:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
```

**Resultado:** Documento de 43.131 caracteres → 66 sub-documentos

### 5.2 Other Splitters (To Research)

- **CharacterTextSplitter**: Por número de caracteres
- **TokenTextSplitter**: Por tokens (TikToken)
- **SentenceTransformersSplit**: Semantic-aware
- **MarkdownHeaderTextSplitter**: Preserva headers
- **Semantic**: Por similaridade semântica

### 5.3 Best Practices

1. **Chunk Size**:
   - 1000 chars: Bom equilíbrio
   - 500-800: Mais granular, mais resultados
   - 1500+: Menos granular, melhor contexto

2. **Overlap**:
   - 200 chars: Padrão recomendado
   - 10-20% do chunk size
   - Preserva contexto entre chunks

3. **Index Tracking**:
   - `add_start_index=True`: Para citations
   - Rastreia posição no documento original
   - Essencial para provenance

---

## 6. PREPROCESSING

### 6.1 Text Normalization

**Técnicas:**
- Remoção de quebras de linha extras
- Normalização de espaços
- Remoção de caracteres especiais
- Unificação de encoding (UTF-8)
- Lowercasing (se aplicável)

**Exemplo:**
```python
import re

def normalize_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special chars
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize
    return text.strip()
```

### 6.2 Encoding Handling

**Problemas Comuns:**
- UTF-8 vs ISO-8859-1
- Caracteres especiais
- BOM (Byte Order Mark)
- Mixed encodings

**Soluções:**
```python
# Try different encodings
for encoding in ['utf-8', 'iso-8859-1', 'cp1252']:
    try:
        with open(file, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        continue
```

### 6.3 Noise Removal

**Tipos de Noise:**
- Header/footer repetitivos
- Navegação (menus, links)
- Scripts e estilos
- Texto boilerplate
- Publicidade

**Técnicas:**
- **SoupStrainer**: Filtrar por tags/classe
- **Pattern matching**: Regex para remover padrões
- **Content length**: Ignorar chunks muito pequenos
- **Duplicates**: Detectar e remover

**Exemplo:**
```python
from bs4 import BeautifulSoup, SoupStrainer

# Only parse specific elements
only_content = SoupStrainer(class_=("post-content", "article"))
soup = BeautifulSoup(html, 'html.parser', parse_only=only_content)
```

---

## 7. METADATA EXTRACTION

### 7.1 Common Metadata

| Metadata | Fonte | Uso |
|----------|-------|-----|
| **URL** | WebLoader | Citations, source |
| **Title** | HTML/PDF | Chunk identification |
| **Author** | DOCX/HTML | Attribution |
| **Date** | Multiple | Temporal context |
| **Language** | Auto-detect | Routing |
| **File Path** | Local files | Tracing |
| **Page Number** | PDF | Positioning |
| **Section** | Structure | Hierarchy |

### 7.2 Metadata Preservation

```python
# LangChain automatically preserves metadata
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs={
        "parse_only": SoupStrainer(class_=("content",)),
        "add_to_docstore": True
    }
)
docs = loader.load()

# Metadata available
for doc in docs:
    print(doc.metadata)
    # {'source': 'https://example.com', 'title': 'Example'}
```

### 7.3 Custom Metadata

```python
def add_custom_metadata(docs):
    for doc in docs:
        doc.metadata.update({
            'processed_at': datetime.now().isoformat(),
            'chunk_id': generate_id(),
            'word_count': len(doc.page_content.split()),
            'language': detect_language(doc.page_content)
        })
    return docs
```

---

## 8. DATA CLEANING

### 8.1 Validation Pipeline

**Checks:**
1. **Empty chunks**: Remover chunks vazios
2. **Length**: Mín/Máx size
3. **Encoding**: Válido UTF-8
4. **Duplicates**: Hash-based detection
5. **Quality**: Stop words ratio

**Exemplo:**
```python
def validate_chunks(docs):
    validated = []
    seen_hashes = set()

    for doc in docs:
        # Check length
        if len(doc.page_content) < 50:
            continue

        # Check for duplicates
        text_hash = hash(doc.page_content)
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        validated.append(doc)

    return validated
```

### 8.2 Quality Metrics

| Métrica | Descrição | Bom | Ruim |
|---------|-----------|-----|------|
| **Word Count** | Palavras por chunk | 100-200 | <50 ou >500 |
| **Stop Words Ratio** | % stop words | 30-40% | >60% ou <10% |
| **Readability** | Flesch score | Médio | Muito baixo/alto |
| **Uniqueness** | % texto único | >80% | <50% |

### 8.3 Cleaning Automation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Complete pipeline
def process_document(file_path):
    # Load
    loader = TextLoader(file_path)
    docs = loader.load()

    # Clean
    docs = clean_documents(docs)

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = splitter.split_documents(docs)

    # Validate
    splits = validate_chunks(splits)

    return splits
```

---

## 9. WINDOWS-SPECIFIC CONSIDERATIONS

### 9.1 Path Handling

```python
import os
from pathlib import Path

# Windows paths
path = Path(r"C:\Users\Documents\file.pdf")
absolute_path = path.resolve()

# Handle spaces and special chars
quoted_path = f'"{absolute_path}"'
```

### 9.2 Encoding

```python
# Windows-specific encoding issues
import locale

# Default encoding
print(locale.getpreferredencoding())  # Usually 'cp1252' on Windows

# Use UTF-8 explicitly
with open(file, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
```

### 9.3 WSL2 Integration

```powershell
# PowerShell to WSL2
wsl --exec python3 /mnt/c/scripts/process.py

# Or use Windows paths in WSL
/mnt/c/Users/Documents/file.pdf
```

### 9.4 Docker Desktop

```dockerfile
# Mount Windows directory
VOLUME /c/Users/Bushido/Documents:/data
```

---

## 10. TOOLS & LIBRARIES

### 10.1 LangChain Ecosystem

**Document Loaders:**
- `langchain-community`: Base loaders
- `langchain-chroma`: ChromaDB integration
- `langchain-pinecone`: Pinecone integration
- `langchain-openai`: OpenAI integration

**Installation:**
```bash
pip install langchain-community
pip install langchain-chroma
pip install langchain-pinecone
```

### 10.2 Specialized Libraries

| Biblioteca | Uso | Integração |
|------------|-----|------------|
| **PyMuPDF** | PDF processing | `PyMuPDFLoader` |
| **pdfplumber** | PDF tables | Custom loader |
| **python-docx** | DOCX processing | `Docx2txtLoader` |
| **BeautifulSoup** | HTML parsing | `WebBaseLoader` |
| **Unstructured** | Multi-format | `UnstructuredLoader` |
| **Tesseract** | OCR | Custom loader |

### 10.3 Windows Installation

```powershell
# Using Chocolatey
choco install python
choco install tesseract
choco install poppler-utils  # For PDF processing

# Or using pip
pip install PyMuPDF pdfplumber python-docx
```

---

## 11. BEST PRACTICES

### 11.1 File Processing

1. **Detect format** antes de escolher loader
2. **Preserve metadata** sempre
3. **Add timestamps** para tracking
4. **Log processing** para debugging
5. **Handle errors** graciosamente

### 11.2 Chunking

1. **Test different sizes** (500, 1000, 1500)
2. **Measure retrieval quality** para cada configuração
3. **Use overlap** para preservar contexto
4. **Track indices** para citations
5. **Validate results** (não vazios, não duplicados)

### 11.3 Data Quality

1. **Clean before splitting** (ruim se propaga)
2. **Validate encoding** (evita erros futuros)
3. **Remove duplicates** (economiza storage)
4. **Check completeness** (páginas faltando)
5. **Log statistics** (counts, sizes, etc.)

### 11.4 Performance

1. **Batch processing** para muitos documentos
2. **Parallel loading** (multi-threading)
3. **Progress tracking** para long jobs
4. **Caching** processed documents
5. **Incremental updates** para new docs

---

## 12. COMMON ISSUES & SOLUTIONS

### 12.1 Encoding Errors

**Symptom**: `UnicodeDecodeError`
**Solution**:
```python
with open(file, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
```

### 12.2 Empty Chunks

**Symptom**: Chunks with no content
**Solution**:
```python
splits = [d for d in splits if d.page_content.strip()]
```

### 12.3 Wrong Text Extraction (PDF)

**Symptom**: Garbled or missing text
**Solution**: Use OCR for scanned PDFs
```python
# Tesseract OCR integration needed
```

### 12.4 Metadata Loss

**Symptom**: Source information missing
**Solution**:
```python
# Preserve metadata during splitting
splits = splitter.split_documents(docs)
# Each split inherits metadata from parent doc
```

### 12.5 Large File Handling

**Symptom**: Memory errors
**Solution**:
```python
# Process in chunks
for chunk in read_large_file(file, chunk_size=1024*1024):
    process_chunk(chunk)
```

---

## 13. CODE EXAMPLES

### 13.1 Complete Pipeline

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_document(file_path, chunk_size=1000, chunk_overlap=200):
    """Process a document through the full pipeline."""
    try:
        # 1. Load
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        splits = splitter.split_documents(docs)

        # 3. Add metadata
        for i, split in enumerate(splits):
            split.metadata.update({
                'chunk_id': i,
                'source': file_path,
                'word_count': len(split.page_content.split())
            })

        print(f"✅ Processed {file_path}: {len(splits)} chunks")
        return splits

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return []

# Usage
splits = process_document(r"C:\data\document.txt")
```

### 13.2 Web Scraping

```python
from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer

def process_url(url):
    """Process a web page."""
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            "parse_only": SoupStrainer(
                class_=["post-content", "article-body"]
            )
        }
    )
    docs = loader.load()
    return docs
```

### 13.3 Batch Processing

```python
import os
from concurrent.futures import ThreadPoolExecutor

def process_directory(directory):
    """Process all documents in a directory."""
    all_splits = []
    files = [f for f in os.listdir(directory) if f.endswith(('.txt', '.pdf', '.docx'))]

    with ThreadPoolExecutor(max_workers=4) as executor:
        for splits in executor.map(process_document, files):
            all_splits.extend(splits)

    print(f"✅ Total: {len(all_splits)} chunks from {len(files)} files")
    return all_splits
```

---

## 14. PRÓXIMOS PASSOS

### 14.1 Pesquisa Adicional Necessária
- [ ] OCR for scanned documents
- [ ] Table extraction techniques
- [ ] Image handling
- [ ] Code block processing
- [ ] Multi-language support
- [ ] Unstructured.io capabilities

### 14.2 Code Examples to Create
- [ ] PDF with tables extraction
- [ ] DOCX with images
- [ ] HTML with custom filtering
- [ ] Batch processing script
- [ ] Quality validation pipeline

### 14.3 Benchmarks
- [ ] Processing time per format
- [ ] Accuracy comparison
- [ ] File size impact
- [ ] Memory usage
- [ ] Quality metrics

---

## 15. REFERENCES

### 15.1 Official Documentation
- LangChain Document Loaders: https://docs.langchain.com/oss/python/langchain/rag
- BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/
- Unstructured: https://unstructured.io/

### 15.2 Tools
- PyMuPDF: https://pymupdf.readthedocs.io/
- python-docx: https://python-docx.readthedocs.io/
- Tesseract: https://github.com/tesseract-ocr/tesseract

### 15.3 Papers
- "Document Layout Analysis for PDF" (to research)
- "Efficient Text Processing Pipelines" (to research)

---

**Status**: ✅ Base para Document Processing coletada
**Próximo**: Seção 02 - Chunking Strategies
**Data Conclusão**: 09/11/2025

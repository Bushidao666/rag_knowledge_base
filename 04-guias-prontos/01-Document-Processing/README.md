# 📚 Guia 01: Document Processing

### Data: 09/11/2025
### Versão: 1.0
### Tempo de Leitura: 50 minutos
### Nível: Iniciante a Avançado

---

## 📑 Índice

1. [Getting Started (20-30 min)](#1-getting-started-20-30-min)
2. [Tutorial Intermediário (1-2h)](#2-tutorial-intermediário-1-2h)
3. [Tutorial Avançado (3-4h)](#3-tutorial-avançado-3-4h)
4. [Implementation End-to-End (half-day)](#4-implementation-end-to-end-half-day)
5. [Best Practices](#5-best-practices)
6. [Code Examples](#6-code-examples)
7. [Decision Trees](#7-decision-trees)
8. [Troubleshooting](#8-troubleshooting)
9. [Recursos Adicionais](#9-recursos-adicionais)

---

## 1. Getting Started (20-30 min)

### 1.1 O que é Document Processing?

**Document Processing** é a primeira etapa crítica do pipeline RAG, responsável por transformar documentos brutos em chunks indexáveis.

**Pipeline Padrão:**
```
Documento Bruto → Load → Split → Store → Vector DB
```

**Por que é importante?**
- ✅ Quality do preprocessing determina quality do retrieval
- ✅ Dados limpos = melhores embeddings
- ✅ Chunking adequado = melhor context
- ✅ Metadata preservado = explicabilidade

### 1.2 Document Object Structure

LangChain usa `Document` objects:

```python
from langchain.schema import Document

doc = Document(
    page_content="Texto do documento...",
    metadata={
        "source": "file.pdf",
        "page": 1,
        "author": "João Silva"
    }
)
```

**Atributos:**
- `page_content`: Texto do documento
- `metadata`: Dicionário com informações adicionais

### 1.3 Suporte a Formatos

| Formato | Loader | Complexidade | Suporte | Limitações |
|---------|--------|------------|---------|-----------|----------|
| **PDF** | PyMuPDFLoader | Média | ✅ Texto + Tabelas | Layout complexo |
| **DOCX** | Docx2txtLoader | Baixa | ✅ Texto + Tabelas | Formatação perdida |
| **HTML** | WebBaseLoader | Baixa | ✅ Páginas web | Scripts/styles |
| **TXT** | TextLoader | Baixa | ✅ Texto puro | Encoding issues |
| **Markdown** | UnstructuredMarkdownLoader | Baixa | ✅ Headers | Links perdidos |
| **PPTX** | PowerPointLoader | Média | ✅ Texto + Tabelas | Slides = páginas |
| **CSV/Excel** | CSVLoader/ExcelLoader | Baixa | ✅ Dados tabulares | Fórmulas perdidas |
| **JSON** | JSONLoader | Baixa | ✅ Estrutura | Hierarquia perdida |

### 1.4 Primeiro Exemplo (5 min)

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Load
loader = TextLoader("documento.txt")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# 3. Store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

print(f"Processed {len(chunks)} chunks")
```

### 1.5 Pipeline 3 Etapas

**ETAPA 1: LOAD**
```python
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("documento.pdf")
docs = loader.load()
```

**ETAPA 2: SPLIT**
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
```

**ETAPA 3: STORE**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)
```

### 1.6 Chunking Best Practices

**Tamanho recomendado:**
- **chunk_size**: 1000 caracteres
- **chunk_overlap**: 200 caracteres (20% do chunk_size)
- **add_start_index**: True (para citations)

**Exemplo:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Preserve posição original
)
```

### 1.7 Quando Usar Cada Formato?

**PDF** → Quando tem estrutura
**DOCX** → Para documentos Word
**HTML** → Para páginas web
**TXT** → Para texto simples
**Markdown** → Para documentação
**CSV/Excel** → Para dados tabulares
**JSON** → Para dados estruturados

---

## 2. Tutorial Intermediário (1-2h)

### 2.1 Carregando PDFs

#### PDF com Texto Extraível
```python
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("manual.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")
for i, doc in enumerate(docs[:3]):
    print(f"Page {i+1}: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}\n")
```

#### PDF com Tabelas
```python
from langchain.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(
    "invoice.pdf",
    strategy="fast"
)
docs = loader.load()

# Tabelas como separado chunks
for doc in docs:
    if "table" in doc.page_content.lower():
        print("Table found!")
```

#### PDF Scanned (OCR)
```python
from langchain.document_loaders import OnlinePDFLoader

# Requer EasyOCR ou Tesseract
loader = OnlinePDFLoader("scanned_doc.pdf")
docs = loader.load()

# OCR slow but works
```

### 2.2 Carregando Web Pages

```python
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup, SoupStrainer

# Carregar apenas conteúdo específico
only_posts = SoupStrainer(class_=("post-content", "post-title"))
loader = WebBaseLoader(
    web_paths=["https://blog.com/post"],
    bs_kwargs=dict(parse_only=only_posts)
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")
```

**Customization:**
```python
loader = WebBaseLoader(
    web_paths=["https://site.com"],
    bs_kwargs={
        "parse_only": SoupStrainer(id=("content", "title")),
        "remove_tags": ["nav", "footer", "header"]
    }
)
```

### 2.3 Carregando DOCX

```python
from langchain.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("documento.docx")
docs = loader.load()

# Preservar metadados
for doc in docs:
    doc.metadata["source"] = "documento.docx"
    doc.metadata["type"] = "word_document"
```

### 2.4 Carregando CSV/Excel

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader(
    "dados.csv",
    source_column="id"
)
docs = loader.load()

# Cada linha = um Document
print(f"Loaded {len(docs)} rows")
print(docs[0].page_content)
print(docs[0].metadata)
```

### 2.5 Carregando JSON

```python
from langchain.document_loaders import JSONLoader

# Custom function para extrair texto
def process_item(item):
    return f"ID: {item['id']}, Name: {item['name']}, Value: {item['value']}"

loader = JSONLoader(
    file_path="data.json",
    jq_schema=".items[]",
    text_process_func=process_item
)

docs = loader.load()
```

### 2.6 Carregando Multiple Formats

```python
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader, UnstructuredFileLoader

loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",
    loader_cls=TextLoader
)

# Diferentes loaders para diferentes tipos
def loader_fn(path):
    if path.endswith(".pdf"):
        return UnstructuredFileLoader(path)
    elif path.endswith(".docx"):
        return Docx2txtLoader(path)
    else:
        return TextLoader(path)

loader = DirectoryLoader(
    "data/",
    loader_func=loader_fn
)
```

### 2.7 Preprocessing

#### Text Normalization
```python
import re

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special chars
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize
    return text.strip()

# Apply to all docs
for doc in docs:
    doc.page_content = clean_text(doc.page_content)
```

#### Encoding Detection
```python
def detect_encoding(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    return 'utf-8'  # fallback
```

#### Noise Removal
```python
from bs4 import BeautifulSoup

def remove_noise(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Get text
    text = soup.get_text()
    return text
```

### 2.8 Metadata Extraction

```python
from langchain.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("manual.pdf")

# Custom metadata
docs = loader.load()
for doc in docs:
    doc.metadata.update({
        "source": "manual.pdf",
        "page_number": doc.metadata.get("page"),
        "total_pages": len(docs),
        "processing_date": "2025-11-09"
    })
```

**Common Metadata:**
- `source`: Arquivo ou URL
- `title`: Título do documento
- `author`: Autor
- `date`: Data de criação
- `language`: Idioma detectado
- `page_number`: Número da página (PDFs)
- `file_type`: Tipo do arquivo

### 2.9 Batch Processing

```python
import os
from pathlib import Path

def process_directory(directory):
    """Process all files in directory"""
    all_docs = []

    for file_path in Path(directory).rglob("*.*"):
        if file_path.suffix.lower() in ['.pdf', '.txt', '.docx']:
            # Load
            loader = get_loader(file_path)
            docs = loader.load()

            # Add metadata
            for doc in docs:
                doc.metadata["file_path"] = str(file_path)
                doc.metadata["file_name"] = file_path.name

            all_docs.extend(docs)

    return all_docs

# Usage
docs = process_directory("data/")
print(f"Processed {len(docs)} documents")
```

### 2.10 Validation

```python
def validate_documents(docs):
    """Validate document quality"""
    issues = []

    for i, doc in enumerate(docs):
        # Check empty content
        if len(doc.page_content.strip()) < 10:
            issues.append(f"Document {i}: Empty or too short")

        # Check metadata
        if "source" not in doc.metadata:
            issues.append(f"Document {i}: Missing source metadata")

        # Check encoding
        try:
            doc.page_content.encode('utf-8')
        except UnicodeError:
            issues.append(f"Document {i}: Encoding issues")

    return issues

# Validate
issues = validate_documents(docs)
if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("All documents valid!")
```

---

## 3. Tutorial Avançado (3-4h)

### 3.1 Custom Document Loaders

```python
from langchain.document_loaders import BaseLoader
from langchain.schema import Document

class CustomDBLoader(BaseLoader):
    """Load from custom database"""

    def __init__(self, connection_string):
        self.connection_string = connection_string

    def lazy_load(self):
        """Lazy loading from database"""
        # Connect to DB
        conn = connect_db(self.connection_string)

        for row in conn.execute("SELECT * FROM documents"):
            yield Document(
                page_content=row["content"],
                metadata={
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"]
                }
            )

# Usage
loader = CustomDBLoader("postgresql://localhost/db")
docs = loader.lazy_load()
```

### 3.2 Advanced PDF Processing

```python
from PyPDF2 import PdfReader
from langchain.document_loaders import BaseLoader

class AdvancedPDFLoader(BaseLoader):
    """Advanced PDF processing with tables and images"""

    def __init__(self, file_path, extract_tables=True):
        self.file_path = file_path
        self.extract_tables = extract_tables

    def lazy_load(self):
        reader = PdfReader(self.file_path)

        for page_num, page in enumerate(reader.pages):
            # Extract text
            text = page.extract_text()

            # Extract tables
            tables = []
            if self.extract_tables:
                tables = self.extract_tables_from_page(page)

            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "page_number": page_num + 1,
                    "tables": tables,
                    "total_pages": len(reader.pages)
                }
            )

            yield doc

    def extract_tables_from_page(self, page):
        """Extract tables using pdfplumber"""
        import pdfplumber

        with pdfplumber.open(self.file_path) as pdf:
            tables = pdf.pages[page.page_number - 1].tables
            return [table for table in tables]

# Usage
loader = AdvancedPDFLoader("document.pdf", extract_tables=True)
docs = loader.lazy_load()
```

### 3.3 Stream Processing

```python
import asyncio
from langchain.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

async def process_large_document(file_path):
    """Process large documents asynchronously"""
    # Load asynchronously
    loader = AsyncHtmlLoader(file_path)
    docs = await alazy_load()

    # Process in batches
    batch_size = 100
    all_chunks = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(batch)
        all_chunks.extend(chunks)

    return all_chunks

# Usage
chunks = asyncio.run(process_large_document("large_file.pdf"))
print(f"Processed {len(chunks)} chunks")
```

### 3.4 OCR Integration

```python
import easyocr
from langchain.document_loaders import BaseLoader

class OCRLoader(BaseLoader):
    """OCR for scanned documents"""

    def __init__(self, file_path, languages=['en']):
        self.file_path = file_path
        self.languages = languages
        self.reader = easyocr.Reader(languages)

    def lazy_load(self):
        # OCR processing
        results = self.reader.readtext(self.file_path)

        for bbox, text, confidence in results:
            if confidence > 0.5:  # Filter low confidence
                yield Document(
                    page_content=text,
                    metadata={
                        "bbox": bbox,
                        "confidence": confidence,
                        "source": self.file_path
                    }
                )

# Usage
loader = OCRLoader("scanned_doc.png", languages=['en', 'pt'])
docs = loader.lazy_load()
```

### 3.5 Multi-Modal Processing

```python
from PIL import Image
import pytesseract
from langchain.document_loaders import BaseLoader

class ImageDocumentLoader(BaseLoader):
    """Extract text from images using OCR"""

    def __init__(self, image_path):
        self.image_path = image_path

    def lazy_load(self):
        # Extract text via OCR
        text = pytesseract.image_to_string(Image.open(self.image_path))

        # Extract metadata
        image = Image.open(self.image_path)
        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "source": self.image_path
        }

        yield Document(page_content=text, metadata=metadata)

# Usage
loader = ImageDocumentLoader("image.png")
docs = loader.lazy_load()
```

### 3.6 Streaming Documents

```python
def stream_large_file(file_path, chunk_size=8192):
    """Stream large files without loading into memory"""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            yield Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "chunk_size": chunk_size
                }
            )

# Usage
for doc in stream_large_file("large_file.txt"):
    # Process chunk
    print(f"Processing chunk: {doc.page_content[:100]}...")
```

### 3.7 Database Integration

```python
import sqlite3
from langchain.document_loaders import BaseLoader

class SQLiteLoader(BaseLoader):
    """Load from SQLite database"""

    def __init__(self, db_path, table_name, text_column):
        self.db_path = db_path
        self.table_name = table_name
        self.text_column = text_column

    def lazy_load(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {self.table_name}")

        for row in cursor.fetchall():
            # Get column index
            cols = [desc[0] for desc in cursor.description]
            text_idx = cols.index(self.text_column)

            # Create document
            doc = Document(
                page_content=row[text_idx],
                metadata={
                    "table": self.table_name,
                    "row_id": row[0],
                    "columns": cols
                }
            )

            yield doc

        conn.close()

# Usage
loader = SQLiteLoader(
    db_path="data.db",
    table_name="documents",
    text_column="content"
)
docs = loader.lazy_load()
```

### 3.8 Error Handling

```python
from langchain.document_loaders import TextLoader

def safe_load_document(file_path):
    """Load document with error handling"""
    encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']

    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            docs = loader.load()

            # Validate
            if not docs:
                print(f"Warning: {file_path} is empty")

            for doc in docs:
                if len(doc.page_content.strip()) < 10:
                    print(f"Warning: {file_path} has very short content")

            return docs

        except Exception as e:
            print(f"Error loading {file_path} with {encoding}: {e}")
            continue

    print(f"Failed to load {file_path} with any encoding")
    return []

# Usage
docs = safe_load_document("file.txt")
```

### 3.9 Progress Tracking

```python
import tqdm
from pathlib import Path

def load_with_progress(directory):
    """Load with progress bar"""
    files = list(Path(directory).rglob("**/*.pdf"))

    all_docs = []

    for file in tqdm.tqdm(files, desc="Loading documents"):
        try:
            loader = PyMuPDFLoader(str(file))
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            tqdm.write(f"Error: {file} - {e}")

    return all_docs

# Usage
docs = load_with_progress("data/")
print(f"Loaded {len(docs)} documents")
```

### 3.10 Caching Processed Documents

```python
import json
import hashlib
from pathlib import Path

def get_cache_path(file_path, cache_dir="cache"):
    """Generate cache file path"""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    return Path(cache_dir) / f"{file_hash}.json"

def load_with_cache(file_path, cache_dir="cache"):
    """Load with caching"""
    cache_path = get_cache_path(file_path, cache_dir)

    # Check cache
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        return [Document(**doc) for doc in cached]

    # Load and cache
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # Save cache
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump([
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ], f)

    return docs

# Usage
docs = load_with_cache("document.pdf")
```

---

## 4. Implementation End-to-End (half-day)

### 4.1 Estrutura do Projeto

```
document_processing/
├── src/
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   ├── web_loader.py
│   │   └── generic_loader.py
│   ├── splitters/
│   │   ├── __init__.py
│   │   ├── text_splitter.py
│   │   └── semantic_splitter.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── metadata_extractor.py
│   │   └── validator.py
│   └── pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── config/
│   └── settings.yaml
├── main.py
└── requirements.txt
```

### 4.2 src/pipeline.py

```python
#!/usr/bin/env python3
"""
Document Processing Pipeline
Complete implementation
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio

from langchain.schema import Document
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Loaders
from langchain.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    WebBaseLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)

from config.settings import Settings

settings = Settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    """Complete document processing pipeline"""

    def __init__(self, data_dir: str, persist_dir: str):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)

        # Init components
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True
        )

        self.vectorstore = None
        logger.info("Pipeline initialized")

    def detect_file_type(self, file_path: Path) -> str:
        """Detect file type and return loader class"""
        ext = file_path.suffix.lower()

        loaders = {
            '.pdf': PyMuPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
            '.html': UnstructuredFileLoader,
            '.md': UnstructuredFileLoader,
        }

        if ext in loaders:
            return loaders[ext]
        else:
            return UnstructuredFileLoader

    def load_document(self, file_path: Path) -> List[Document]:
        """Load single document"""
        loader_class = self.detect_file_type(file_path)
        loader = loader_class(str(file_path))

        try:
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} from {file_path}")

            # Add metadata
            for doc in docs:
                doc.metadata["source_file"] = str(file_path)
                doc.metadata["file_type"] = file_path.suffix
                doc.metadata["file_name"] = file_path.name

            return docs
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def load_all_documents(self) -> List[Document]:
        """Load all documents from directory"""
        all_docs = []

        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.docx', '.html', '.md']:
                docs = self.load_document(file_path)
                all_docs.extend(docs)

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    def clean_document(self, doc: Document) -> Document:
        """Clean and normalize document"""
        # Basic cleaning
        content = doc.page_content
        content = content.replace('\n\n\n', '\n\n')  # Extra newlines
        content = content.strip()

        # Create cleaned document
        cleaned_doc = Document(
            page_content=content,
            metadata=doc.metadata
        )

        return cleaned_doc

    def extract_metadata(self, doc: Document) -> Document:
        """Extract additional metadata"""
        # Add timestamp
        from datetime import datetime
        doc.metadata["processed_at"] = datetime.now().isoformat()

        # Add file stats
        if "source_file" in doc.metadata:
            file_path = Path(doc.metadata["source_file"])
            if file_path.exists():
                doc.metadata["file_size"] = file_path.stat().st_size

        return doc

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info("Splitting documents...")

        # Clean first
        cleaned_docs = [self.clean_document(doc) for doc in docs]

        # Split
        chunks = self.splitter.split_documents(cleaned_docs)

        # Add metadata to chunks
        for chunk in chunks:
            chunk = self.extract_metadata(chunk)

        logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document]):
        """Create and persist vector store"""
        logger.info("Creating vector store...")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )

        self.vectorstore.persist()
        logger.info(f"Vector store saved to {self.persist_dir}")

    def load_vectorstore(self) -> bool:
        """Load existing vector store"""
        if self.persist_dir.exists():
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            logger.info(f"Loaded vector store from {self.persist_dir}")
            return True
        return False

    def run(self) -> Dict[str, Any]:
        """Run complete pipeline"""
        logger.info("Starting document processing pipeline")

        start_time = time.time()

        # Check if vector store exists
        if self.load_vectorstore():
            return {
                "status": "loaded_from_cache",
                "chunks": self.vectorstore._collection.count(),
                "time_seconds": time.time() - start_time
            }

        # Load documents
        docs = self.load_all_documents()

        if not docs:
            logger.warning("No documents found")
            return {"status": "no_documents"}

        # Split
        chunks = self.split_documents(docs)

        # Create vector store
        self.create_vectorstore(chunks)

        end_time = time.time()

        logger.info(f"Pipeline completed in {end_time - start_time:.2f}s")

        return {
            "status": "completed",
            "documents": len(docs),
            "chunks": len(chunks),
            "time_seconds": end_time - start_time
        }

# CLI
if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--persist-dir", default="vectorstore")
    args = parser.parse_args()

    pipeline = DocumentProcessingPipeline(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir
    )

    result = pipeline.run()
    print(f"Result: {result}")
```

### 4.3 config/settings.yaml

```yaml
# Document processing settings
embedding_model: "text-embedding-ada-002"
chunk_size: 1000
chunk_overlap: 200

# Paths
data_dir: "data/raw"
persist_dir: "data/processed"

# Logging
log_level: "INFO"
log_file: "logs/processing.log"

# Processing
max_workers: 4
batch_size: 100
```

### 4.4 tests/test_processing.py

```python
import pytest
from pathlib import Path
from document_processing.pipeline import DocumentProcessingPipeline

def test_pipeline():
    """Test pipeline"""
    # Setup
    pipeline = DocumentProcessingPipeline(
        data_dir="tests/data",
        persist_dir="tests/vectorstore"
    )

    # Run
    result = pipeline.run()

    # Verify
    assert result["status"] in ["completed", "loaded_from_cache"]
    assert result["chunks"] > 0

    print(f"Test passed: {result}")

def test_loaders():
    """Test individual loaders"""
    from document_processing.loaders.pdf_loader import PDFLoader
    from document_processing.loaders.text_loader import TextLoader

    # Test PDF
    pdf_loader = PDFLoader()
    docs = pdf_loader.load("tests/data/sample.pdf")
    assert len(docs) > 0

    # Test Text
    text_loader = TextLoader()
    docs = text_loader.load("tests/data/sample.txt")
    assert len(docs) > 0

    print("Loader tests passed!")

if __name__ == "__main__":
    test_loaders()
    test_pipeline()
```

---

## 5. Best Practices

### 5.1 File Processing

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Detect file type automaticamente | Hardcode loaders |
| Preserve metadata sempre | Perder informações de source |
| Add timestamps | Processar sem tracking |
| Log processing steps | Silently fail |
| Handle errors graciosamente | Crash on bad files |
| Validate loaded content | Assume files are valid |
| Clean data before splitting | Split dirty data |
| Use batch processing | One file at a time |
| Track progress | Long running jobs sem feedback |

### 5.2 Chunking

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Use RecursiveCharacterTextSplitter | CharacterTextSplitter apenas |
| Test different chunk sizes | Usar um tamanho fixo |
| Use overlap (10-20%) | Chunks sem overlap |
| Add start index | Lose posição original |
| Preserve structure | Random splits |
| Validate chunks | Assume split worked |
| Measure quality | Blind splitting |
| A/B test sizes | Escolher por intuição |

### 5.3 Data Quality

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Clean before splitting | Dirty data in, dirty out |
| Detect encoding issues | UTF-8 assumed |
| Remove noise | Scripts/styles in chunks |
| Check for duplicates | Duplicate chunks |
| Validate completeness | Missing pages |
| Log statistics | No visibility |
| Fix issues early | Propagate errors |
| Document preprocessing | Implicit assumptions |

### 5.4 Performance

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Batch processing | One-by-one |
| Async/await I/O | Synchronous only |
| Stream large files | Load all in memory |
| Cache processed docs | Reprocess same files |
| Parallel processing | Sequential only |
| Progress tracking | Black box processing |
| Optimize I/O | Random disk access |
| Use connection pooling | New connection per file |

### 5.5 Production

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Health checks | Deploy sem monitoring |
| Error logging | Silent failures |
| Retry logic | Single attempt only |
| Circuit breakers | Cascading failures |
| Resource limits | Unlimited processing |
| Config from env | Hardcoded values |
| Version documents | Implicit changes |
| Backup processing | No recovery plan |

---

## 6. Code Examples

### Example 1: Multi-Format Loader (30 linhas)

```python
from langchain.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from pathlib import Path

class MultiFormatLoader:
    def __init__(self):
        self.loaders = {
            '.pdf': PyMuPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader
        }

    def load(self, file_path):
        ext = Path(file_path).suffix.lower()
        loader_class = self.loaders.get(ext)

        if not loader_class:
            raise ValueError(f"Unsupported format: {ext}")

        loader = loader_class(file_path)
        return loader.load()

# Usage
loader = MultiFormatLoader()
docs = loader.load("document.pdf")
print(f"Loaded {len(docs)} documents")
```

### Example 2: Document Cleaner (40 linhas)

```python
import re
from langchain.schema import Document

def clean_document(doc: Document) -> Document:
    """Clean and normalize document"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', doc.page_content)
    text = text.strip()

    # Remove special chars (keep only word chars, spaces, basic punctuation)
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}]', '', text)

    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Create cleaned document
    cleaned = Document(
        page_content=text,
        metadata=doc.metadata
    )

    return cleaned

# Batch clean
def clean_batch(docs):
    return [clean_document(doc) for doc in docs]

# Usage
docs = clean_batch(raw_docs)
print(f"Cleaned {len(docs)} documents")
```

### Example 3: Metadata Extractor (50 linhas)

```python
import os
from datetime import datetime
from pathlib import Path
from langchain.schema import Document

def extract_metadata(file_path: str, doc: Document) -> Document:
    """Extract comprehensive metadata"""
    path = Path(file_path)

    # File stats
    metadata = {
        "file_path": str(path),
        "file_name": path.name,
        "file_ext": path.suffix,
        "file_size": path.stat().st_size if path.exists() else 0,
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
    }

    # Processing info
    metadata["processed_at"] = datetime.now().isoformat()
    metadata["processor"] = "DocumentProcessingPipeline v1.0"

    # Content stats
    metadata["char_count"] = len(doc.page_content)
    metadata["word_count"] = len(doc.page_content.split())
    metadata["line_count"] = doc.page_content.count('\n') + 1

    # Quality checks
    metadata["has_content"] = len(doc.page_content.strip()) > 0
    metadata["too_short"] = len(doc.page_content) < 10
    metadata["potential_noise"] = doc.page_content.count('\n\n\n') > 5

    # Create document with metadata
    enhanced_doc = Document(
        page_content=doc.page_content,
        metadata=metadata
    )

    return enhanced_doc

# Usage
for doc in docs:
    doc = extract_metadata("file.pdf", doc)
    print(f"Metadata: {doc.metadata}")
```

### Example 4: Batch Processor (60 linhas)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

class BatchProcessor:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def process_file(self, file_path: str) -> List[Document]:
        """Process single file"""
        from document_processing.pipeline import DocumentProcessingPipeline

        pipeline = DocumentProcessingPipeline(
            data_dir=os.path.dirname(file_path),
            persist_dir="temp"
        )
        # Implementation here
        return pipeline.load_single_file(file_path)

    def process_batch(self, file_paths: List[str]) -> List[Document]:
        """Process multiple files in parallel"""
        all_docs = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.process_file, path): path
                for path in file_paths
            }

            # Collect results
            for future in futures:
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                except Exception as e:
                    file_path = futures[future]
                    print(f"Error processing {file_path}: {e}")

        return all_docs

    async def process_directory_async(self, directory: str) -> List[Document]:
        """Async directory processing"""
        import aiofiles
        import aiofiles.os

        file_paths = []
        async for file in aiofiles.os.scandir(directory):
            if file.is_file():
                file_paths.append(file.path)

        tasks = [self.process_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs = []
        for result in results:
            if isinstance(result, list):
                all_docs.extend(result)
            elif isinstance(result, Exception):
                print(f"Error: {result}")

        return all_docs

# Usage
processor = BatchProcessor(max_workers=4)
docs = processor.process_batch(["file1.pdf", "file2.txt", "file3.docx"])
print(f"Processed {len(docs)} documents")
```

### Example 5: Quality Validator (45 linhas)

```python
from typing import List, Dict, Any
from langchain.schema import Document

class DocumentValidator:
    """Validate document quality"""

    def __init__(self):
        self.issues = []
        self.stats = {}

    def validate(self, docs: List[Document]) -> Dict[str, Any]:
        """Run all validations"""
        self.issues = []
        self.stats = {
            "total_docs": len(docs),
            "total_chars": 0,
            "empty_docs": 0,
            "short_docs": 0,
            "metadata_missing": 0,
            "encoding_issues": 0
        }

        for i, doc in enumerate(docs):
            # Count stats
            self.stats["total_chars"] += len(doc.page_content)

            # Check empty
            if len(doc.page_content.strip()) == 0:
                self.stats["empty_docs"] += 1
                self.issues.append(f"Document {i}: Empty content")

            # Check short
            if len(doc.page_content) < 10:
                self.stats["short_docs"] += 1
                self.issues.append(f"Document {i}: Too short (< 10 chars)")

            # Check metadata
            if not doc.metadata or "source" not in doc.metadata:
                self.stats["metadata_missing"] += 1
                self.issues.append(f"Document {i}: Missing metadata")

            # Check encoding
            try:
                doc.page_content.encode('utf-8')
            except UnicodeError:
                self.stats["encoding_issues"] += 1
                self.issues.append(f"Document {i}: Encoding issues")

        # Calculate averages
        if docs:
            self.stats["avg_chars"] = self.stats["total_chars"] / len(docs)
            self.stats["empty_rate"] = self.stats["empty_docs"] / len(docs)
            self.stats["short_rate"] = self.stats["short_docs"] / len(docs)

        return self.get_report()

    def get_report(self) -> Dict[str, Any]:
        """Get validation report"""
        report = {
            "stats": self.stats,
            "issues": self.issues,
            "pass": len(self.issues) == 0,
            "quality_score": self.calculate_score()
        }
        return report

    def calculate_score(self) -> float:
        """Calculate quality score (0-100)"""
        if not self.stats["total_docs"]:
            return 0.0

        # Deduct points for issues
        score = 100.0
        score -= self.stats["empty_rate"] * 30  # 30 points for empty docs
        score -= self.stats["short_rate"] * 20  # 20 points for short docs
        score -= (self.stats["metadata_missing"] / self.stats["total_docs"]) * 20  # 20 points for missing metadata
        score -= (self.stats["encoding_issues"] / self.stats["total_docs"]) * 30  # 30 points for encoding

        return max(0.0, min(100.0, score))

# Usage
validator = DocumentValidator()
report = validator.validate(docs)
print(f"Quality score: {report['quality_score']:.1f}/100")
if report["issues"]:
    print("Issues found:")
    for issue in report["issues"][:10]:  # Show first 10
        print(f"  - {issue}")
```

---

## 7. Decision Trees

### 7.1 Decision Tree: Qual Loader Usar?

```
START
  └─> Formato do arquivo?
      ├─ PDF ──> Scanning?
      │   ├─ YES ──> OCR (PyMuPDFLoader + OCR)
      │   └─ NO ──> PyMuPDFLoader
      │
      ├─ DOCX/DOC ──> Docx2txtLoader
      │
      ├─ TXT ──> TextLoader
      │
      ├─ HTML/MD ──> UnstructuredFileLoader
      │
      ├─ CSV/Excel ──> CSVLoader/ExcelLoader
      │
      ├─ JSON ──> JSONLoader
      │
      └─ Outro ──> UnstructuredFileLoader
```

### 7.2 Decision Tree: Chunking Strategy

```
START
  └─> Tipo de documento?
      ├─ Structured (PDF tables, DOCX) ──> Section-based + chunk_size=800
      │
      ├─ Semi-structured (Markdown, HTML) ──> Header-aware + chunk_size=1000
      │
      ├─ Unstructured (TXT) ──> RecursiveCharacterTextSplitter + chunk_size=1000
      │
      └─ Technical (Code, Config) ──> Language-specific + chunk_size=500
          └─ Overlap=10% (vs 20% normal)
```

### 7.3 Decision Tree: Processing Approach

```
START
  └─> Volume de dados?
      ├─ < 100 files ──> Single-threaded
      │
      ├─ 100-1000 files ──> ThreadPoolExecutor (4 workers)
      │
      └─ > 1000 files ──> Async + Batch processing
          └─ Streaming + Progress tracking
```

### 7.4 Decision Tree: Quality Checks

```
START
  └─> Results quality
      ├─ Empty content ──> Check loader + file format
      ├─ Low retrieval ──> Adjust chunking + overlap
      ├─ Missing metadata ──> Enhance extraction
      └─ Encoding errors ──> Detect + fix encoding
          └─ Multiple encodings support
```

---

## 8. Troubleshooting

### 8.1 Problema: Empty Documents

**Sintomas:**
- Documentos carregados mas content está vazio
- Splitting resulta em chunks vazios
- Vector store sem conteúdo

**Causas Comuns:**
- File encoding diferente do esperado
- Loader incorreto para formato
- Arquivo corrompido
- Permissões de acesso

**Soluções:**

1. **Verificar encoding:**
```python
# Detect encoding
def detect_encoding(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                if content.strip():
                    return encoding, content
        except UnicodeDecodeError:
            continue
    return None, ""

encoding, content = detect_encoding(file_path)
if encoding:
    loader = TextLoader(file_path, encoding=encoding)
    docs = loader.load()
```

2. **Verificar loader:**
```python
# Testar diferentes loaders
for loader_class in [PyMuPDFLoader, UnstructuredFileLoader]:
    try:
        loader = loader_class(file_path)
        docs = loader.load()
        if docs and docs[0].page_content.strip():
            print(f"Success with {loader_class}")
            break
    except Exception as e:
        print(f"Failed with {loader_class}: {e}")
```

3. **Log content preview:**
```python
# Debug loading
loader = PyMuPDFLoader(file_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages")
print(f"First page preview: {docs[0].page_content[:200]}")
print(f"Metadata: {docs[0].metadata}")
```

### 8.2 Problema: Poor Chunking Quality

**Sintomas:**
- Chunks sem sentido
- Contexto perdido
- Retrieval irrelevante

**Causas Comuns:**
- Chunk size inadequado
- Overlap muito pequeno
- Separadores incorretos
- Estrutura não preservada

**Soluções:**

1. **Ajustar chunking:**
```python
# Testar diferentes configurações
for chunk_size in [500, 800, 1000, 1200, 1500]:
    for overlap in [100, 150, 200, 250]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        chunks = splitter.split_documents(docs)
        quality = evaluate_quality(chunks)
        print(f"size={size}, overlap={overlap}, quality={quality}")
```

2. **Usar semantic splitting:**
```python
from langchain.text_splitters import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="gradient"
)
chunks = splitter.split_text(text)
```

3. **Preservar estrutura:**
```python
from langchain.text_splitters import MarkdownHeaderTextSplitter

# Para Markdown
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "header 1"),
        ("##", "header 2"),
        ("###", "header 3")
    ]
)
chunks = splitter.split_text(text)
```

### 8.3 Problema: Slow Processing

**Sintomas:**
- Loading muito lento
- Splitting demora muito
- Vector store creation lenta

**Causas Comuns:**
- Muitos arquivos grandes
- I/O síncrono
- Sem paralelização
- Processamento desnecessário

**Soluções:**

1. **Batch processing:**
```python
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

with ThreadPoolExecutor(max_workers=4) as executor:
    file_paths = ["file1.pdf", "file2.pdf", "file3.pdf"]
    futures = [executor.submit(process_file, path) for path in file_paths]
    all_docs = [f.result() for f in futures]
```

2. **Async processing:**
```python
import asyncio

async def load_file(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

async def main():
    file_paths = ["file1.pdf", "file2.pdf", "file3.pdf"]
    tasks = [load_file(path) for path in file_paths]
    all_docs = await asyncio.gather(*tasks)
    return all_docs

docs = asyncio.run(main())
```

3. **Cache processed files:**
```python
import hashlib
import json
from pathlib import Path

def get_cache_key(file_path):
    return hashlib.md5(file_path.encode()).hexdigest()

def load_with_cache(file_path, cache_dir="cache"):
    cache_file = Path(cache_dir) / f"{get_cache_key(file_path)}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return [Document(**doc) for doc in json.load(f)]

    # Process and cache
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump([
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ], f)

    return docs
```

### 8.4 Problema: Metadata Lost

**Sintomas:**
- Citations não funcionam
- Source tracking quebrado
- Context sem provenance

**Causas Comuns:**
- Metadata não preservada no split
- Loader não extrai metadata
- Processamento limpo metadata

**Soluções:**

1. **Preservar metadata:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Preserve position
)

# Metadata é preservado automaticamente
chunks = splitter.split_documents(docs)
for chunk in chunks:
    print(chunk.metadata)  # Has source info
```

2. **Enrich metadata:**
```python
for doc in docs:
    doc.metadata.update({
        "source": file_path,
        "processed_at": datetime.now().isoformat(),
        "chunk_id": get_chunk_id(doc)
    })
```

3. **Custom metadata extraction:**
```python
def extract_metadata(file_path, doc):
    # File stats
    path = Path(file_path)
    doc.metadata.update({
        "file_size": path.stat().st_size,
        "file_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    })
    return doc

docs = [extract_metadata(file_path, doc) for doc in docs]
```

### 8.5 Problema: Encoding Errors

**Sintomas:**
- UnicodeDecodeError
- Caracteres estranhos no output
- Chunks com text truncado

**Soluções:**

1. **Encoding detection:**
```python
def safe_read(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return encoding, content
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Cannot read {file_path} with any encoding")

encoding, content = safe_read(file_path)
loader = TextLoader(file_path, encoding=encoding)
docs = loader.load()
```

2. **Robust processing:**
```python
def clean_text(text):
    # Handle encoding issues
    try:
        return text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        return text.encode('utf-8', errors='ignore').decode('utf-8')

# Apply to all docs
docs = [Document(
    page_content=clean_text(doc.page_content),
    metadata=doc.metadata
) for doc in docs]
```

---

## 9. Recursos Adicionais

### 9.1 Document Loaders Reference
- LangChain: https://python.langchain.com/docs/integrations/document_loaders/
- 160+ loaders disponíveis
- Community loaders: https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain/document_loaders

### 9.2 Text Splitters Reference
- RecursiveCharacterTextSplitter (recomendado)
- CharacterTextSplitter (básico)
- TokenTextSplitter (token-aware)
- SemanticSplitter (BERT-based)
- MarkdownHeaderTextSplitter (preserva estrutura)

### 9.3 Formatos Suportados

**Texto:**
- PDF, TXT, MD, HTML
- DOCX, PPTX
- JSON, CSV, Excel
- Jupyter Notebooks
- Code files (Python, JS, etc.)

**Dados:**
- Databases (SQL, NoSQL)
- APIs (REST, GraphQL)
- Web content
- Email
- Chat logs

**Multimodal:**
- Images (com OCR)
- Audio (transcription)
- Video (frame extraction)

### 9.4 Bibliotecas Úteis

**PDF Processing:**
- PyMuPDF (recomendado)
- pdfplumber (tabelas)
- PDFMiner (básico)
- pdf2image (converter para imagens)

**Office Documents:**
- python-docx (Word)
- python-pptx (PowerPoint)
- openpyxl (Excel)

**Web Content:**
- BeautifulSoup (HTML parsing)
- Selenium (JavaScript-rendered)
- Playwright (modern web)

**General:**
- Unstructured.io (multi-format)
- Tesseract (OCR)
- pypandoc (format conversion)

### 9.5 Tutorial Videos
- LangChain Document Loaders
- Text Splitting Strategies
- Vector Store Creation
- Metadata Best Practices

### 9.6 Community Resources
- LangChain Discord #document-loaders
- GitHub Discussions
- Stack Overflow tags: langchain, document-loaders
- Reddit: r/LangChain

### 9.7 Performance Benchmarks

| Loader | Speed | Memory | Quality | Best For |
|--------|--------|--------|----------|----------|
| PyMuPDF | Fast | Medium | High | PDFs estruturados |
| TextLoader | Very Fast | Low | High | Plain text |
| Unstructured | Medium | Medium | High | Multi-format |
| WebBaseLoader | Slow | Medium | Medium | HTML content |
| Docx2txt | Fast | Low | Medium | Word docs |

### 9.8 Example Datasets
- Common Crawl (web content)
- ArXiv (academic papers)
- Project Gutenberg (books)
- Wikipedia dumps
- News datasets

---

**Última atualização**: 09/11/2025
**Versão**: 1.0
**Autor**: RAG Knowledge Base Project

---

**Próximo**: [Guia 02 - Chunking Strategies](./02-Chunking-Strategies/README.md)

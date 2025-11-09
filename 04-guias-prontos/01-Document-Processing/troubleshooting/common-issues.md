# Troubleshooting - Document Processing

## Problemas Comuns e Soluções

### 1. Import Errors

**Sintomas:**
- ModuleNotFoundError: No module named 'langchain'
- ImportError: No module named 'langchain_community'

**Soluções:**

```bash
# Instalar dependências
pip install --upgrade langchain langchain-community

# Para PDF
pip install pymupdf  # ou pdfplumber

# Para DOCX
pip install python-docx

# Para Excel/CSV
pip install pandas openpyxl
```

### 2. PDF Loading Issues

**Sintomas:**
- PDF não carrega (páginas em branco)
- Texto illegível
- Erro: "Unable to get text"

**Soluções:**

#### A. Usar Loader Alternativo
```python
from langchain.document_loaders import PDFPlumberLoader

# Se PyPDF não funciona, use PDFPlumber
loader = PDFPlumberLoader("documento.pdf")
docs = loader.load()
```

#### B. Forçar OCR (para PDFs escaneados)
```python
import pytesseract
from langchain.document_loaders import PyMuPDFLoader

# OCR support
loader = PyMuPDFLoader("scanned.pdf")
# Adicionar OCR se necessário
```

#### C. Configurar PyMuPDF
```python
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader(
    "documento.pdf",
    extract_images=True  # Extrair imagens
)
docs = loader.load()
```

### 3. Text Encoding Errors

**Sintomas:**
- UnicodeDecodeError
- Caracteres estranhos no texto
- Encoding mismatch

**Soluções:**

```python
from langchain.document_loaders import TextLoader

# Especificar encoding
loader = TextLoader(
    "documento.txt",
    encoding="utf-8"  # ou "latin-1", "cp1252"
)
docs = loader.load()

# Para múltiplos encodings
try:
    loader = TextLoader("documento.txt", encoding="utf-8")
except UnicodeDecodeError:
    loader = TextLoader("documento.txt", encoding="latin-1")
```

### 4. Chunk Overlap Issues

**Sintomas:**
- Contexto perdido entre chunks
- Respostas fragmentadas
- Information gaps

**Soluções:**

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Aumentar overlap para preservar contexto
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,  # Aumentar de 200 para 300
    separators=["\n\n", "\n", ".", " "]
)
```

### 5. Memory Issues

**Sintomas:**
- Out of Memory
- Process killed
- Slow processing

**Soluções:**

#### A. Batch Processing
```python
def process_in_batches(documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        # Processar batch
        chunks = splitter.split_documents(batch)
        yield chunks
```

#### B. Limitar Documentos
```python
# Processar apenas primeiros N documentos
documents = documents[:100]  # Limitar a 100
```

### 6. Web Page Loading Issues

**Sintomas:**
- Página não carrega
- Content vazio
- Timeout errors

**Soluções:**

```python
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

# Configurar loader
loader = WebBaseLoader(
    web_paths=["https://exemplo.com"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=["post-content", "article"]
        )
    )
)
docs = loader.load()

# Para sites JavaScript
# WebBaseLoader não suporta JS
# Use PlaywrightLoader ou similar
```

### 7. DOCX Issues

**Sintomas:**
- Tabelas não preservadas
- Formatação perdida
- Imagens ignoradas

**Soluções:**

```python
from langchain.document_loaders import Docx2txtLoader

# Docx2txt apenas texto
loader = Docx2txtLoader("documento.docx")

# Para tabelas, use python-docx diretamente
from docx import Document

doc = Document("documento.docx")
full_text = []
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            full_text.append(cell.text)
```

### 8. CSV Loading Issues

**Sintomas:**
- Colunas mal formatadas
- Separadores incorretos
- Encoding issues

**Soluções:**

```python
from langchain.document_loaders import CSVLoader

# Configurar separador
loader = CSVLoader(
    "dados.csv",
    delimiter=";",  # Se CSV usa ;
    encoding="utf-8"
)

# Especificar colunas
loader = CSVLoader(
    "dados.csv",
    source_column="produto"  # Coluna como source
)
```

### 9. Markdown Issues

**Sintomas:**
- Headers perdidos
- Code blocks mal formatados
- Links quebrados

**Soluções:**

```python
from langchain.document_loaders import UnstructuredMarkdownLoader

# Preservar estrutura
loader = UnstructuredMarkdownLoader(
    "readme.md",
    mode="elements"  # Preservar elementos
)

# Para melhor parsing
loader = UnstructuredMarkdownLoader(
    "readme.md",
    strategy="fast"
)
```

### 10. Large File Processing

**Sintomas:**
- Processamento lento
- Timeout
- System freeze

**Soluções:**

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Processar arquivo grande
def process_large_file(file_path, max_chunk_size=1000):
    loader = TextLoader(file_path)
    documents = []

    for doc in loader.lazy_load():  # Lazy loading
        documents.append(doc)

        if len(documents) >= 100:
            # Processar em batches
            yield from process_batch(documents)
            documents = []

    if documents:
        yield from process_batch(documents)
```

### 11. Metadata Not Preserved

**Sintomas:**
- Source information lost
- Can't trace back to original
- No citations possible

**Soluções:**

```python
# Verificar se metadata está presente
docs = loader.load()
print(docs[0].metadata)

# Adicionar metadata customizada
for doc in docs:
    doc.metadata["source"] = "meu_documento.txt"
    doc.metadata["processed_at"] = datetime.now().isoformat()

# Preservar metadata durante split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Adiciona posição
)
chunks = splitter.split_documents(docs)
```

### 12. Vector Store Size Issues

**Sintomas:**
- Vector store muito grande
- Slow queries
- High memory usage

**Soluções:**

```python
# Ajustar chunk_size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Menor = mais chunks
    chunk_overlap=100
)

# Comprimir embeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Usar modelo menor
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'  # Menor
)
```

## Debug Checklist

- [ ] Verificar se loader está correto para o formato
- [ ] Testar com arquivo simples primeiro
- [ ] Verificar encoding do arquivo
- [ ] Validar chunk_size (não muito grande)
- [ ] Testar se metadata está preservada
- [ ] Verificar se documentos têm conteúdo
- [ ] Validar separator characters
- [ ] Testar com batch pequeno

## Validation Script

```python
def validate_documents(documents):
    """Validar documentos carregados"""
    for i, doc in enumerate(documents):
        # Check content
        if not doc.page_content or len(doc.page_content.strip()) < 10:
            print(f"⚠️  Document {i}: Empty or too short")
            continue

        # Check metadata
        if not doc.metadata:
            print(f"⚠️  Document {i}: No metadata")

        # Check length
        if len(doc.page_content) > 100000:
            print(f"⚠️  Document {i}: Very long ({len(doc.page_content)} chars)")

    print(f"✓ Validated {len(documents)} documents")
```

## Prevention Tips

1. **Teste com sample files** antes de processar em massa
2. **Verifique encoding** antes de load
3. **Log progress** para arquivos grandes
4. **Use lazy loading** quando possível
5. **Validate output** após cada etapa
6. **Handle errors gracefully** com try-catch
7. **Monitor memory usage** durante processing
8. **Backup original files** antes de processar

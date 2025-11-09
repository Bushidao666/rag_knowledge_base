# Quick Start: Document Processing

**Tempo estimado:** 15-30 minutos
**Nível:** Iniciante
**Pré-requisitos:** Python 3.8+

## Objetivo
Aprender o pipeline básico de processamento de documentos: Load → Split → Store

## O que é Document Processing?
Primeira etapa do RAG que transforma documentos brutos em chunks indexáveis:
```
Load → Split → Store
```

## Pipeline Padrão

### 1. Load - Carregar Documentos
```python
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

# Carregar diferentes formatos
text_loader = TextLoader("documento.txt")
pdf_loader = PyPDFLoader("manual.pdf")
web_loader = WebBaseLoader("https://exemplo.com")

docs = text_loader.load()
```

### 2. Split - Dividir em Chunks
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Dividir em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
```

### 3. Store - Indexar
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Criar vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
```

## Exemplo Completo

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Load
loader = TextLoader("meu_documento.txt")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# 3. Store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

print(f"Criado vector store com {len(chunks)} chunks")
```

## Formatos Suportados

| Formato | Loader | Exemplo |
|---------|--------|---------|
| **TXT** | TextLoader | `TextLoader("arquivo.txt")` |
| **PDF** | PyPDFLoader | `PyPDFLoader("manual.pdf")` |
| **HTML** | WebBaseLoader | `WebBaseLoader("https://site.com")` |
| **DOCX** | Docx2txtLoader | `Docx2txtLoader("doc.docx")` |
| **CSV** | CSVLoader | `CSVLoader("dados.csv")` |
| **Markdown** | UnstructuredMarkdownLoader | `UnstructuredMarkdownLoader("readme.md")` |

## Parâmetros Importantes

**Chunking:**
- `chunk_size=1000` - Tamanho ideal (1K tokens)
- `chunk_overlap=200` - Sobreposição preserva contexto

**Por que chunking?**
- LLM têm context window limitado
- Melhora precisão do retrieval
- Reduz custo de embedding

## Exemplo com PDF

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("manual.pdf")
pages = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)

# Store (local)
embeddings = HuggingFaceEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(chunks, embeddings)
```

## Document Object

Cada documento carregado é um objeto com:
```python
class Document:
    page_content: str  # Texto
    metadata: dict     # {source, page, etc.}
```

## Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'langchain'"
```bash
pip install --upgrade langchain
```

### Erro: "Failed to load PDF"
```python
# Usar loader alternativo
from langchain.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader("manual.pdf")
```

### Resposta sem sentido
**Problema:** Documentos muito grandes
**Solução:** Ajustar chunk_size
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Menor
    chunk_overlap=100
)
```

## Próximos Passos

- 📖 **Tutorial Intermediário:** [Tutoriais](../tutorials/)
- 💻 **Exemplos Práticos:** [Code Examples](../code-examples/)
- 🔧 **Troubleshooting:** [Problemas Comuns](../troubleshooting/common-issues.md)

## Recursos

- 📄 **LangChain Docs:** https://docs.langchain.com/oss/python/langchain/rag
- 📊 **Comparação Loaders:** [Guia completo](../README.md)
- 🎯 **Chunking Strategies:** [Guia 02 - Chunking](../02-Chunking-Strategies/README.md)

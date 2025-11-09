# Quick Start: Embedding Models

**Tempo estimado:** 15-30 minutos
**Nível:** Iniciante
**Pré-requisitos:** Python, documentos para indexar

## Objetivo
Aprender a selecionar e usar modelos de embedding para RAG

## O que são Embeddings?
Representações vetoriais de texto que capturam significado semântico:
```
Texto → Embedding Model → Vetor (e.g., 1536 dimensões)
```

Similaridade entre vetores = similaridade semântica

## Modelos Populares

### 1. OpenAI Embeddings (Commercial)
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"  # ou "text-embedding-3-small"
)
```

### 2. BGE (Open Source)
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)
```

### 3. E5 (Open Source)
```python
embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/E5-large-v2"
)
```

### 4. MiniLM (Open Source - Rápido)
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Comparação Models

| Model | Dimensões | Qualidade | Velocidade | Custo | Multilingue |
|-------|-----------|-----------|------------|-------|-------------|
| **text-embedding-3-large** | 3072 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $$$ | ✅ |
| **text-embedding-3-small** | 1536 | ⭐⭐⭐ | ⭐⭐⭐⭐ | $ | ✅ |
| **BGE-large** | 1024 | ⭐⭐⭐⭐⭐ | ⭐⭐ | $ | ✅ |
| **E5-large** | 1024 | ⭐⭐⭐⭐ | ⭐⭐⭐ | $ | ✅ |
| **MiniLM** | 384 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $ | ❌ |

## Exemplo Básico

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Embed a single text
text = "RAG é uma técnica que combina busca com geração"
vector = embeddings.embed_query(text)

print(f"Vector dimensions: {len(vector)}")
print(f"First 5 values: {vector[:5]}")

# 3. Embed multiple texts
texts = ["Texto 1", "Texto 2", "Texto 3"]
vectors = embeddings.embed_documents(texts)
```

## Embedding + Vector Store

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter

# 1. Load and split
loader = TextLoader("documento.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Create embeddings + vector store
embeddings = HuggingFaceEmbeddings('BAAI/bge-large-en-v1.5')
vectorstore = Chroma.from_documents(chunks, embeddings)

# 3. Query
query = "O que é RAG?"
docs = vectorstore.similarity_search(query, k=3)
print(docs[0].page_content)
```

## Seleção de Modelos

### Use OpenAI se:
- ✅ Precisa máxima qualidade
- ✅ Não se importa com custo
- ✅ Multilingue
- ✅ Production-ready
- ✅ Suporte oficial

### Use BGE se:
- ✅ Quality alta + gratuito
- ✅ Open source
- ✅ Research/academic
- ✅ Fine-tuning possível

### Use E5 se:
- ✅ Balance qualidade/velocidade
- ✅ Open source
- ✅ Instruction-tuned
- ✅ Good general use

### Use MiniLM se:
- ✅ Velocidade máxima
- ✅ Recursos limitados
- ✅ Prototipagem
- ✅ Quality média ok

## Código Production-Ready

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import os

class EmbeddingModel:
    def __init__(self, provider="openai", model_name=None):
        self.provider = provider
        self.model_name = model_name or self._get_default_model()
        self.model = self._load_model()

    def _get_default_model(self):
        defaults = {
            "openai": "text-embedding-3-small",
            "huggingface": "BAAI/bge-large-en-v1.5"
        }
        return defaults[self.provider]

    def _load_model(self):
        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not set")
            return OpenAIEmbeddings(model=self.model_name)
        else:
            return HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_query(self, text):
        return self.model.embed_query(text)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

# Usage
embeddings = EmbeddingModel(provider="openai")
# ou
embeddings = EmbeddingModel(provider="huggingface")
```

## Troubleshooting

### API Key Error
**Problema:** `AuthenticationError`
**Solução:**
```python
import os
os.environ["OPENAI_API_KEY"] = "sua-key-aqui"
```

### Model Not Found
**Problema:** `Model not found`
**Solução:**
```python
# Verificar nome correto
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
```

### Slow Embeddings
**Problema:** Muito lento
**Soluções:**
1. Usar modelo menor (MiniLM)
2. Batch processing
3. Async embedding

### High Memory Usage
**Problema:** Out of memory
**Soluções:**
1. Smaller model
2. Process in batches
3. Reduce batch size

## Próximos Passos

- 💻 **Exemplos Práticos:** [Code Examples](../code-examples/)
- 🔧 **Troubleshooting:** [Problemas Comuns](../troubleshooting/common-issues.md)
- 🗄️ **Vector DBs:** [Guia 04 - Vector Databases](../04-Vector-Databases/README.md)

# Quick Start: Vector Databases

**Tempo estimado:** 15-30 minutos
**Nível:** Iniciante
**Pré-requisitos:** Embeddings já criados

## Objetivo
Aprender a usar bancos de dados vetoriais para armazenamento e busca

## O que é Vector Database?
Sistema para armazenar e buscar vetores (embeddings) por similaridade:
```
Query → Embedding → Similarity Search → Top-K Results
```

## Vector DBs Populares

### 1. Chroma (Local/Open Source)
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
```

### 2. Pinecone (Cloud/Managed)
```python
import pinecone
from langchain.vectorstores import Pinecone

# Setup
pinecone.init(api_key="your-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")

# Create
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="my-index")

# Search
docs = vectorstore.similarity_search("query")
```

### 3. FAISS (Library/Local)
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
```

### 4. Weaviate (Open Source/Cloud)
```python
from langchain.vectorstores import Weaviate
import weaviate

client = weaviate.Client("https://your-cluster.weaviate.network")
vectorstore = Weaviate.from_documents(chunks, embeddings, client=client)
```

## Comparação Vector DBs

| Database | Deployment | Scale | Cost | Features | Complexity |
|----------|------------|-------|------|----------|------------|
| **Chroma** | Local/Cloud | Small-Med | Free | Basic | ⭐ |
| **Pinecone** | Cloud | Large-Enterprise | $$$ | Full | ⭐⭐ |
| **FAISS** | Local | Small-Med | Free | Fast | ⭐⭐ |
| **Weaviate** | Both | Medium-Large | Free-$ | Rich | ⭐⭐⭐ |
| **Qdrant** | Both | Medium-Large | Free-$ | Fast | ⭐⭐ |
| **Milvus** | Both | Large | Free | Scalable | ⭐⭐⭐ |

## Exemplo Completo

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Prepare documents
loader = TextLoader("documento.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Create embeddings + vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 3. Similarity search
query = "O que é RAG?"
docs = vectorstore.similarity_search(query, k=3)

print(f"Found {len(docs)} relevant documents:")
for doc in docs:
    print(f"- {doc.page_content[:100]}...")
```

## Operações Básicas

### Add Documents
```python
# Add new documents
vectorstore.add_documents(new_chunks)

# Add with metadata
vectorstore.add_texts(
    texts=["Text 1", "Text 2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)
```

### Similarity Search
```python
# Search with score
docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)

# Search with filter
docs = vectorstore.similarity_search(
    query,
    k=3,
    filter={"source": "documento.pdf"}
)
```

### Delete Documents
```python
# Delete by metadata
vectorstore.delete(ids=["doc1", "doc2"])

# Delete all
vectorstore.delete_collection()
```

## Persistência

```python
# Chroma - persist locally
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./vectorstore"  # Save to disk
)

# Load later
vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings
)
```

## Selection Guide

### Development/Prototyping
**Use Chroma ou FAISS**
- Local, fácil setup
- Free
- Good para experiments

### Production (Small-Medium Scale)
**Use Weaviate ou Qdrant**
- Cloud ou self-hosted
- Good features
- Balance cost/complexity

### Production (Large Scale)
**Use Pinecone ou Milvus**
- Enterprise features
- High performance
- Managed service

### Research/Academic
**Use FAISS**
- Library, not full DB
- Very fast
- Custom indexing

## Production Checklist

- [ ] Choose right database
- [ ] Setup persistence
- [ ] Implement backup
- [ ] Add monitoring
- [ ] Configure security
- [ ] Plan for scaling
- [ ] Test with real data
- [ ] Document setup

## Troubleshooting

### Index Not Found
**Problema:** Vector store não existe
**Solução:** Build index primeiro

### Connection Error
**Problema:** Não conecta ao DB
**Solução:** Verificar credentials e URL

### Slow Search
**Problemas possíveis:**
- Índice não otimizado
- Muitoos vectors
- Network latency

**Soluções:**
- Add index parameters
- Comprimir vectors
- Usar local DB

### Memory Issues
**Problema:** High RAM usage
**Soluções:**
- Smaller embeddings
- Compressão (PQ, SQ8)
- Cloud DB

## Próximos Passos

- 💻 **Exemplos Práticos:** [Code Examples](../code-examples/)
- 🔧 **Troubleshooting:** [Problemas Comuns](../troubleshooting/common-issues.md)
- 🔍 **Similarity Search:** [Guia 05 - Retrieval Optimization](../05-Retrieval-Optimization/README.md)

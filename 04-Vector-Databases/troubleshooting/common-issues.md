# Troubleshooting - Vector Databases

## Problemas Comuns e Soluções

### 1. Index Not Found

**Sintomas:**
- IndexNotFoundError
- Vector store não existe
- Collection not found

**Soluções:**

```python
# Verificar se índice existe
from langchain.vectorstores import Chroma
import os

if not os.path.exists("./vectorstore"):
    print("Index not found, creating new one")

# Criar índice
vectorstore = Chroma.from_documents(chunks, embeddings)

# Verificar collection
try:
    count = vectorstore._collection.count()
    print(f"Index has {count} documents")
except Exception as e:
    print(f"Error: {e}")
```

### 2. Connection Errors

**Sintomas:**
- Connection timeout
- Network error
- Authentication failure

**Soluções:**

```python
# Pinecone
import pinecone
pinecone.init(
    api_key="your-key",
    environment="us-west1-gcp"
)
# Test connection
index = pinecone.Index("test-index")
index.describe_index_stats()

# Weaviate
import weaviate
try:
    client = weaviate.Client("https://cluster.weaviate.network")
    print(client.is_ready())
except Exception as e:
    print(f"Connection error: {e}")
```

### 3. Slow Search Performance

**Sintomas:**
- High query latency
- Slow similarity search
- Timeout errors

**Soluções:**

#### A. Otimizar parâmetros
```python
# Chroma
docs = vectorstore.similarity_search(
    query,
    k=5,  # Reduzir k
    filter=None  # Remove filter
)

# Pinecone
docs = vectorstore.similarity_search_by_vector(
    vector,
    k=5,
    include_metadata=False  # Faster
)
```

#### B. Usar índice otimizado
```python
# FAISS com índice otimizado
import faiss
import numpy as np

dimension = 384
index = faiss.IndexIVFPQ(
    faiss.IndexFlatL2(dimension),
    dimension,
    1000,  # nlist
    64     # m (code size)
)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10  # Search in 10 cells
```

#### C. Compressão de vetores
```python
# Quantization para reduzir tamanho
# Pinecone - automatic
# FAISS
index = faiss.IndexScalarQuantizer(
    faiss.IndexFlatL2(dimension),
    faiss.METRIC_L2
)
index.train(embeddings)
index.add(embeddings)
```

### 4. High Memory Usage

**Sintomas:**
- Out of Memory
- High RAM consumption
- System slowdown

**Soluções:**

#### A. Usar cloud database
```python
# Pinecone (managed)
vectorstore = Pinecone.from_documents(
    chunks,
    embeddings,
    index_name="my-index"
)
# Memory managed by Pinecone
```

#### B. Delete old data
```python
# Chroma
vectorstore.delete(ids=["doc1", "doc2"])

# Delete entire collection
vectorstore.delete_collection()
vectorstore = Chroma.from_documents(new_chunks, embeddings)
```

#### C. Batch processing
```python
# Processar em batches
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    vectorstore.add_documents(batch)
```

### 5. Data Inconsistency

**Sintomas:**
- Documents missing
- Duplicated entries
- Wrong metadata

**Soluções:**

```python
# Verificar count
count = vectorstore._collection.count()
print(f"Total documents: {count}")

# Listar IDs
try:
    ids = vectorstore.get()["ids"]
    print(f"Document IDs: {ids[:10]}...")
except:
    print("Cannot list IDs")

# Deduplicar
def deduplicate_by_id(vectorstore):
    ids = vectorstore.get()["ids"]
    seen = set()
    for id in ids:
        if id in seen:
            vectorstore.delete(ids=[id])
        seen.add(id)
```

### 6. Filtering Not Working

**Sintomas:**
- Filter not applied
- All results returned
- Wrong filter syntax

**Soluções:**

```python
# Correto filter syntax
docs = vectorstore.similarity_search(
    query,
    k=3,
    filter={"source": "document.pdf"}  # Metadado correto
)

# Verificar metadados
try:
    docs_with_metadata = vectorstore.get(
        include=["metadatas"]
    )
    print(docs_with_metadata["metadatas"][0])
except Exception as e:
    print(f"Error: {e}")
```

### 7. Persistance Issues

**Sintomas:**
- Data lost after restart
- No persistence
- Empty on reload

**Soluções:**

```python
# Chroma - persist to disk
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./vectorstore"  # Important!
)

# Reload
vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings
)

# Pinecone - automatic cloud persistence
```

### 8. Scale Limitations

**Sintomas:**
- Performance degrades
- Can't handle large datasets
- Timeouts

**Soluções:**

#### A. Choose scalable DB
```python
# For large scale, use:
# Pinecone (enterprise)
# Milvus (open source, scalable)
# Weaviate (sharding)
```

#### B. Sharding
```python
# Weaviate - sharding
import weaviate
client = weaviate.Client(
    "https://cluster.weaviate.network",
    additional_config=weaviate.AdditionalConfig(
        sharding={
            "desiredCount": 3  # 3 shards
        }
    )
)
```

### 9. Error Handling

**Sintomas:**
- Unhandled exceptions
- Crashes
- No error recovery

**Soluções:**

```python
def safe_search(vectorstore, query):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return docs
    except Exception as e:
        print(f"Search error: {e}")
        # Fallback
        return []
```

### 10. Backup and Recovery

**Sintomas:**
- No backup strategy
- Data loss risk
- Can't restore

**Soluções:**

```python
# Chroma - backup
import shutil
import os

# Backup entire directory
if os.path.exists("./vectorstore"):
    shutil.copytree("./vectorstore", "./vectorstore_backup")

# Pinecone - export data
docs = vectorstore.get(include=["documents", "metadatas"])
# Save to file for backup
```

## Monitoring

```python
# Track search performance
import time

def monitor_search():
    start = time.time()
    docs = vectorstore.similarity_search("test", k=3)
    duration = time.time() - start
    print(f"Search took {duration:.3f}s")

    # Log metrics
    return {
        "duration": duration,
        "num_results": len(docs)
    }
```

## Checklist de Produção

- [ ] Choose appropriate vector DB
- [ ] Setup persistence
- [ ] Implement backup
- [ ] Add monitoring
- [ ] Configure alerts
- [ ] Test with production data
- [ ] Document setup
- [ ] Plan for scaling

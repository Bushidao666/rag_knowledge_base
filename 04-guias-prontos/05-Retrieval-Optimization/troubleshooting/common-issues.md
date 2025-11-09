# Troubleshooting - Retrieval Optimization

## Problemas Comuns

### 1. Poor Retrieval Quality

**Problema:** Resultados irrelevantes

**Soluções:**
```python
# Aumentar k
docs = vectorstore.similarity_search(query, k=8)

# Usar hybrid search
ensemble = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)
docs = ensemble.get_relevant_documents(query)

# Ajustar chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Menor
    chunk_overlap=150
)
```

### 2. Slow Search

**Problema:** Latência alta

**Soluções:**
```python
# Reduzir k
docs = vectorstore.similarity_search(query, k=3)

# Cache frequent queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query_hash):
    return vectorstore.similarity_search(query, k=3)
```

### 3. Inconsistent Results

**Problema:** Variação nos resultados

**Soluções:**
```python
# Fixar score threshold
docs = vectorstore.similarity_search(
    query,
    k=3,
    score_threshold=0.8
)
```

## Debug Checklist

- [ ] Verify embeddings quality
- [ ] Check chunking parameters
- [ ] Test different k values
- [ ] Try hybrid search
- [ ] Add reranking
- [ ] Monitor latency

# Quick Start: Performance Optimization

**Tempo estimado:** 15-30 minutos
**Nível:** Avançado
**Pré-requisitos:** RAG em produção

## Objetivo
Otimizar performance de sistemas RAG para alta escala

## Estratégias de Otimização

### 1. Caching
Cache embeddings e queries frequentes:
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def get_cached_embedding(text):
    return embeddings.embed_query(text)

# Cache queries
@lru_cache(maxsize=1000)
def cached_query(question):
    return rag.query(question)
```

### 2. Async Processing
Operações assíncronas:
```python
import asyncio

async def async_batch_search(queries):
    tasks = [vectorstore.asimilarity_search(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

# Usage
queries = ["Query 1", "Query 2", "Query 3"]
results = asyncio.run(async_batch_search(queries))
```

### 3. Batch Processing
Processar múltiplas queries:
```python
def batch_embed(texts, batch_size=100):
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
    return all_vectors
```

### 4. Vector Compression
Comprimir vetores para economizar memória:
```python
# FAISS - Quantization
import faiss

dimension = 384
quantizer = faiss.IndexScalarQuantizer(faiss.METRIC_L2)
index = quantizer.train(embeddings)
index.add(embeddings)

# 32-bit to 8-bit = 75% reduction
```

### 5. GPU Acceleration
Usar GPU para embeddings:
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="model",
    model_kwargs={"device": "cuda"}
)
```

## Performance Metrics

### Latency
- Target: <3s for query
- Monitor: p50, p95, p99

### Throughput
- Target: 100+ QPS
- Measure: Queries per second

### Memory
- Target: <1GB for 1M vectors
- Optimize: Compression, batching

## Exemplo Completo

```python
import asyncio
from functools import lru_cache

class OptimizedRAG:
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.cache = {}

    @lru_cache(maxsize=10000)
    def get_embedding(self, text):
        return self.embeddings.embed_query(text)

    async def async_query(self, question):
        # Check cache
        if question in self.cache:
            return self.cache[question]

        # Get embedding
        query_vector = self.get_embedding(question)

        # Search
        docs = await self.vectorstore.asimilarity_search(
            question,
            k=3
        )

        # Cache result
        result = {"docs": docs}
        self.cache[question] = result
        return result

# Usage
rag = OptimizedRAG(vectorstore, embeddings)
result = asyncio.run(rag.async_query("What is RAG?"))
```

## Comparison

| Optimization | Speed Gain | Memory Savings | Complexity |
|--------------|------------|----------------|------------|
| **Caching** | 10-100x | Low | Low |
| **Async** | 2-5x | None | Medium |
| **Batch** | 3-10x | Low | Low |
| **Compression** | 1.2x | 50-75% | Medium |
| **GPU** | 5-50x | None | Medium |

## When to Use

- **Caching:** High traffic, repeated queries
- **Async:** Multiple concurrent users
- **Batch:** Processing documents
- **Compression:** Memory constraints
- **GPU:** Embedding bottlenecks

## Monitoring

```python
import time
from prometheus_client import Counter, Histogram

# Metrics
query_counter = Counter('rag_queries_total', 'Total queries')
query_latency = Histogram('rag_query_duration_seconds', 'Query latency')

def monitored_query(question):
    query_counter.inc()
    start = time.time()
    result = rag.query(question)
    query_latency.observe(time.time() - start)
    return result
```

## Troubleshooting

### Out of Memory
- ✅ Use compression
- ✅ Reduce batch size
- ✅ Clear cache
- ✅ Use cloud DB

### Slow Embeddings
- ✅ Use GPU
- ✅ Smaller model
- ✅ Batch processing
- ✅ Cache results

### High Latency
- ✅ Cache frequent queries
- ✅ Reduce k
- ✅ Use async
- ✅ Optimize prompt

## Próximos Passos

- 💻 **Code Examples:** [Exemplos Avançados](../code-examples/)
- 🚀 **Production:** [Guia 11 - Production Deployment](../11-Production-Deployment/README.md)

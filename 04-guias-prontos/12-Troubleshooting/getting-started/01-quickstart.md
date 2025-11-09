# Quick Start: Troubleshooting

**Tempo estimado:** 15-30 minutos
**Nível:** Todos os níveis
**Pré-requisitos:** Sistema RAG implementado

## Objetivo
Diagnosticar e resolver problemas em sistemas RAG

## Problemas Comuns

### 1. Low Retrieval Quality

**Sintomas:** Respostas irrelevantes, baixa precisão

**Diagnóstico:**
```python
# Verificar chunks
print(f"Total chunks: {len(chunks)}")
print(f"Sample chunk: {chunks[0].page_content[:100]}")

# Verificar embeddings
test_vector = embeddings.embed_query("test")
print(f"Embedding dim: {len(test_vector)}")

# Testar similaridade
docs = vectorstore.similarity_search("test query", k=3)
print(f"Retrieved {len(docs)} docs")
for doc in docs:
    print(f"  - {doc.page_content[:80]}...")
```

**Soluções:**
- Ajustar chunk_size/overlap
- Testar diferentes embeddings
- Verificar qualidade dos dados
- Usar hybrid search

### 2. High Latency

**Sintomas:** Queries lentas, timeout

**Diagnóstico:**
```python
import time

# Profile each step
start = time.time()
docs = vectorstore.similarity_search(query, k=3)
retrieve_time = time.time() - start

start = time.time()
answer = llm(context)
generate_time = time.time() - start

print(f"Retrieve: {retrieve_time:.3f}s")
print(f"Generate: {generate_time:.3f}s")
```

**Soluções:**
- Cache frequent queries
- Reduzir k
- Usar async
- Batch processing
- Compressão de vetores

### 3. Out of Memory

**Sintomas:** Process killed, alta RAM

**Diagnóstico:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

**Soluções:**
- Processar em batches menores
- Usar cloud vector DB
- Compressão de vetores
- Limpar cache

### 4. Hallucinations

**Sintomas:** Respostas factualmente incorretas

**Diagnóstico:**
```python
# Verificar se contexto tem a resposta
context = "\n".join([doc.page_content for doc in docs])
print(f"Context length: {len(context)}")
print(f"Context: {context[:500]}...")

# Verificar prompt
print(f"Prompt: {prompt}")
```

**Soluções:**
- Melhorar retrieval (mais contexto relevante)
- Ajustar prompt (exigir citations)
- Usar fact-checking
- Reranking

### 5. Inconsistent Results

**Sintomas:** Variação nos resultados

**Diagnóstico:**
```python
# Test same query multiple times
for i in range(5):
    result = rag.query(question)
    print(f"Run {i+1}: {result['answer'][:100]}")
```

**Soluções:**
- Fixar temperature=0
- Usar cache determinístico
- Verificar preprocessing
- Versionar modelo

## Debugging Tools

### LangSmith Tracing
```python
from langsmith import Client

client = Client()
with client.trace("rag-query") as run:
    run.inputs = {"question": question}
    result = rag.query(question)
    run.outputs = {"answer": result["answer"]}
```

### Vector Store Inspector
```python
def inspect_vectorstore(vectorstore):
    print(f"Total vectors: {vectorstore._collection.count()}")

    # Sample vectors
    sample = vectorstore.get(limit=5)
    print(f"Sample metadata: {sample['metadatas']}")
```

### Embedding Validator
```python
def validate_embeddings(texts, embeddings):
    vectors = embeddings.embed_documents(texts)

    # Check consistency
    v1 = embeddings.embed_query(texts[0])
    v2 = embeddings.embed_query(texts[0])
    assert v1 == v2, "Non-deterministic embeddings"

    print(f"✅ Embeddings valid")
```

## Monitoring & Alerting

### Metrics
```python
from prometheus_client import Counter, Histogram

# Define metrics
query_counter = Counter('rag_queries_total', 'Total queries')
query_latency = Histogram('rag_query_duration_seconds', 'Query latency')
error_counter = Counter('rag_errors_total', 'Total errors')

# In query
try:
    result = rag.query(question)
    query_counter.inc()
except Exception as e:
    error_counter.inc()
    raise
```

### Alerts
```yaml
# Prometheus alert
- alert: RAGHighLatency
  expr: histogram_quantile(0.95, rag_query_duration_seconds) > 5
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "RAG latency is high"
```

## Error Recovery

### Circuit Breaker
```python
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None

    def call(self, func, *args, **kwargs):
        if self.failure_count >= self.threshold:
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker open")
            else:
                self.failure_count = 0

        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            raise
```

## Health Checks

```python
@app.get("/health")
async def health_check():
    checks = {
        "vector_db": check_vector_db(),
        "llm": check_llm(),
        "embeddings": check_embeddings()
    }

    all_healthy = all(checks.values())

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks
    }

def check_vector_db():
    try:
        vectorstore._collection.count()
        return True
    except:
        return False
```

## Common Solutions Table

| Problem | Quick Fix | Long-term Fix |
|---------|-----------|---------------|
| Low quality | Increase k | Better chunking |
| High latency | Cache results | Optimize architecture |
| OOM | Reduce batch size | Use cloud DB |
| Hallucinations | Add citations | Better retrieval |
| Inconsistent | Set temp=0 | Fix randomness |

## Checklist

- [ ] Check logs
- [ ] Verify metrics
- [ ] Test with sample data
- [ ] Inspect vector store
- [ ] Validate embeddings
- [ ] Profile performance
- [ ] Check error rates
- [ ] Verify configurations
- [ ] Test recovery
- [ ] Document issues

## Resources

- 📊 **LangSmith:** For tracing
- 📈 **Prometheus:** For metrics
- 🔍 **Logs:** For debugging
- 🧪 **Tests:** For validation

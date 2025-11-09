# Relatório de Pesquisa: Seção 07 - Performance Optimization

### Data: 09/11/2025
### Status: Fase 4 - Advanced Topics

---

## 1. RESUMO EXECUTIVO

Performance Optimization é crítico para sistemas RAG em produção, especialmente com grandes volumes de dados e tráfego. Otimizações podem melhorar latência 10x-100x, throughput e reduzir custos significativamente.

**Insights Chave:**
- **Vector Compression**: PQ, scalar quantization reduzem memória 4x-32x
- **GPU Acceleration**: 10x-50x speedup para embedding generation
- **Caching**: Redis, in-memory para queries frequentes
- **Approximate NN**: HNSW, IVF para busca sub-linear
- **Batch Processing**: Throughput 10x maior
- **Connection Pooling**: Reduce overhead de redes

---

## 2. VECTOR COMPRESSION

### 2.1 Product Quantization (PQ)

**PQ** divide cada vetor em sub-vetores e quantiza cada sub-vetor independently.

**Como Funciona:**
```
Original vector: [1024 dimensions]
↓ Split into m parts
Sub-vector 1: [256 dims] → Quantize to 1 code
Sub-vector 2: [256 dims] → Quantize to 1 code
...
Sub-vector m: [256 dims] → Quantize to 1 code

Total: m codes (vs 1024 floats)
Compression: 1024 / m bytes
```

**Implementação (FAISS):**
```python
import faiss
import numpy as np

# Create training data
vectors = np.random.rand(100000, 1024).astype('float32')

# Create PQ index
m = 64  # Number of sub-vectors
nbits = 8  # Bits per sub-vector
pq = faiss.IndexPQ(1024, m, nbits)

# Train and add
pq.train(vectors)
pq.add(vectors)

# Search
query = np.random.rand(1, 1024).astype('float32')
D, I = pq.search(query, k=10)

print(f"Compression ratio: {1024 / (m * nbits / 8):.1f}x")
```

**Parâmetros:**
- `m`: Número de sub-vetores (typical: 64, 96, 128)
- `nbits`: Bits por sub-vetor (typical: 8, 16)
- Trade-off: **Compression vs Recall**

**Vantagens:**
- ✅ 4x-32x compression
- ✅ Fast search
- ✅ Good recall com m adequado

**Desvantagens:**
- ❌ Reconstruction error
- ❌ Lower precision que Flat
- ❌ Training required

### 2.2 Scalar Quantization (SQ)

**SQ** quantiza cada dimensão independently para N levels.

**Tipos:**
- **SQ8**: 8 bits per dimension (1 byte)
- **SQ4**: 4 bits per dimension
- **SQ2**: 2 bits per dimension

**Implementação (FAISS):**
```python
import faiss

# Create SQ index
sq = faiss.IndexScalarQuantizer(1024, faiss.ScalarQuantizer.QT_8bit)

# Train and add
sq.train(vectors)
sq.add(vectors)

# Search
D, I = sq.search(query, k=10)
```

**Vantagens:**
- ✅ Simple implementation
- ✅ Fast training
- ✅ Good compression

**Desvantagens:**
- ❌ Coarser quantization que PQ
- ❌ Lower accuracy

### 2.3 Binary Quantization (BQ)

**BQ** representa cada dimensión como 0 ou 1 (1 bit).

**Implementação:**
```python
# Convert to binary
binary_vectors = (vectors > np.median(vectors, axis=0, keepdims=True)).astype('uint8')

# Use Hamming distance
from sklearn.metrics.pairwise import cosine_similarity

def hamming_distance(a, b):
    return np.sum(np.bitwise_xor(a, b), axis=1)

distances = hamming_distance(binary_query, binary_vectors)
```

**Vantagens:**
- ✅ Ultra compression (1/32 of original)
- ✅ XOR-based search (very fast)
- ✅ Memory efficient

**Desvantagens:**
- ❌ Significant information loss
- ❌ Lower recall

### 2.4 Comparison

| Method | Compression | Speed | Recall | Use Case |
|--------|-------------|-------|--------|----------|
| **Flat** | 1x | Medium | Perfect | Small datasets |
| **PQ** | 4x-32x | Fast | Good | Large datasets |
| **SQ8** | 8x | Fast | Medium | General use |
| **BQ** | 32x | Very Fast | Poor | Extreme compression |

### 2.5 Guidelines

**Use PQ when:**
- ✅ Large datasets (1M+ vectors)
- ✅ Memory constrained
- ✅ Accept small recall loss
- ✅ Training data available

**Use SQ8 when:**
- ✅ Medium compression needed
- ✅ Simple implementation
- ✅ Fast training

**Use BQ when:**
- ✅ Extreme compression critical
- ✅ Speed is paramount
- ✅ Can accept quality loss

---

## 3. GPU ACCELERATION

### 3.1 Embedding Generation

**GPU** pode acelerar embedding generation 10x-100x.

**PyTorch CUDA:**
```python
import torch
from sentence_transformers import SentenceTransformer

# Enable CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model on GPU
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
model = model.to(device)

# Batch processing on GPU
sentences = ["text1", "text2", ...]  # 1000 sentences
batch_size = 64

embeddings = []
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]

    # Move to GPU
    batch_tensors = model.tokenize(batch).to(device)

    # Generate embeddings
    batch_embeddings = model.encode(batch_tensors)
    embeddings.extend(batch_embeddings.cpu().numpy())

print(f"Generated {len(embeddings)} embeddings")
```

**Optimizations:**
- **Batch size**: Balance GPU memory vs throughput
- **Mixed precision**: torch.cuda.amp para 2x speed
- **DataLoader**: Use num_workers para multi-processing
- **Model parallel**: For very large models

### 3.2 Vector Search on GPU

**FAISS GPU:**
```python
# Create GPU index
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# Use GPU for search
D, I = gpu_index.search(query_vectors, k=100)

# Cleanup
del gpu_index
del res
```

**Performance (Typical):**
- **CPU (IVF)**: 1,000 QPS
- **GPU (Flat)**: 10,000 QPS
- **GPU (IVF)**: 50,000 QPS

### 3.3 CUDA Best Practices

```python
# 1. Warmup GPU
torch.cuda.empty_cache()
for _ in range(10):
    _ = model.encode(["warmup"])

# 2. Use pinned memory for data transfer
pin_memory = True
dataloader = DataLoader(dataset, batch_size=64, pin_memory=pin_memory)

# 3. Avoid CPU-GPU transfers in loop
# Bad
for batch in batches:
    embeddings = model.encode(batch)  # Transfers every iteration
    process(embeddings)

# Good
all_embeddings = []
for batch in batches:
    embeddings = model.encode(batch)
    all_embeddings.append(embeddings)
all_embeddings = torch.cat(all_embeddings)
process(all_embeddings)
```

---

## 4. CACHING STRATEGIES

### 4.1 Query Caching

**Cache** frequent queries para evitar recomputation.

**In-Memory Cache (LRU):**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embedding(query):
    """Cache embeddings by hash of query."""
    return model.encode(query)

def get_embedding_cached(query):
    """Get embedding with caching."""
    # Create hash of query
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return cached_embedding(query)
```

**Redis Cache:**
```python
import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_embedding_redis(query, ttl=3600):
    """Get embedding with Redis cache."""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"embedding:{query_hash}"

    # Try cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Generate and cache
    embedding = model.encode(query)
    redis_client.setex(
        cache_key,
        ttl,
        json.dumps(embedding.tolist())
    )
    return embedding
```

### 4.2 Result Caching

**Cache** retrieval results por query.

```python
def get_cached_retrieval(query, ttl=1800):
    """Cache retrieval results."""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"retrieval:{query_hash}"

    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Perform retrieval
    results = retriever.get_relevant_documents(query)

    # Cache results
    redis_client.setex(
        cache_key,
        ttl,
        json.dumps([{
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results])
    )

    return results
```

### 4.3 Embedding Cache

**Cache** embeddings de documents.

```python
from langchain_community.document_loaders import TextLoader
import hashlib

class CachedEmbeddingStore:
    def __init__(self, redis_client, model):
        self.redis_client = redis_client
        self.model = model

    def get_document_embedding(self, doc_id, text, ttl=86400):
        """Get or compute document embedding."""
        cache_key = f"doc_embedding:{doc_id}"

        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        # Compute embedding
        embedding = self.model.encode(text)

        # Cache
        self.redis_client.setex(
            cache_key,
            ttl,
            json.dumps(embedding.tolist())
        )

        return embedding
```

### 4.4 Cache Invalidation

**Estrategies:**

1. **TTL-based**: Automatic expiration
2. **Manual invalidation**: When documents updated
3. **Version-based**: Cache by document version
4. **Size-based**: LRU eviction

```python
# Invalidate when document changes
def update_document(doc_id, new_text):
    # Update document
    save_document(doc_id, new_text)

    # Invalidate caches
    redis_client.delete(f"doc_embedding:{doc_id}")

    # Invalidate retrieval results that might include this doc
    pattern = f"retrieval:*"
    keys = redis_client.keys(pattern)
    for key in keys:
        redis_client.delete(key)
```

---

## 5. APPROXIMATE NEAREST NEIGHBOR

### 5.1 HNSW (Hierarchical Navigable Small World)

**HNSW** cria multi-level graph para search rápido.

**Vantagens:**
- ✅ Fast search (logarithmic)
- ✅ High recall
- ✅ Dynamic updates
- ✅ No training required

**Implementação (FAISS):**
```python
import faiss

# Create HNSW index
dim = 1024
index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per level

# Add vectors
index.add(vectors)

# Set efConstruction (build time vs quality)
index.hnsw.efConstruction = 200

# Search parameters
index.hnsw.efSearch = 64  # Higher = better recall, slower

# Search
D, I = index.search(query_vectors, k=10)
```

**Parâmetros:**
- `efConstruction`: Build quality (10-200)
- `efSearch`: Search quality (16-128)
- `M`: Number of neighbors (16-64)

### 5.2 IVF (Inverted File Index)

**IVF** particiona space em cells usando k-means.

**Implementação:**
```python
# Create IVF index
nlist = 100  # Number of cells
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

# Train (use subset for large datasets)
index.train(vectors[:10000])  # Train on 10k vectors
index.add(vectors)

# Search parameters
nprobe = 10  # Number of cells to search

# Search
D, I = index.search(query_vectors, k=10, params={'nprobe': nprobe})
```

**Parâmetros:**
- `nlist`: Number of cells (sqrt(N) typical)
- `nprobe`: Cells to search (1-nlist)
- Trade-off: `nprobe` vs speed vs recall

### 5.3 IVF-PQ (IVF + Product Quantization)

**Combines IVF** para coarse quantization + **PQ** para fine quantization.

```python
# Create IVF-PQ index
nlist = 100
m = 64  # PQ sub-vectors
nbits = 8
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

# Train
index.train(vectors[:10000])
index.add(vectors)

# Search
D, I = index.search(query_vectors, k=10)
```

**Vantages:**
- ✅ Very fast
- ✅ Memory efficient
- ✅ Good for large datasets

### 5.4 Comparison

| Index | Build Time | Search Speed | Memory | Recall | Best For |
|-------|-----------|--------------|--------|--------|----------|
| **Flat** | None | Slow | High | Perfect | Small datasets |
| **HNSW** | Fast | Fast | Medium | High | General use |
| **IVF** | Medium | Fast | Low | High | Large datasets |
| **IVF-PQ** | Medium | Very Fast | Very Low | Medium | Very large datasets |

---

## 6. BATCH PROCESSING

### 6.1 Batch Embedding Generation

```python
def batch_encode(sentences, batch_size=100, show_progress=True):
    """Batch encode sentences for efficiency."""
    embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

        if show_progress:
            print(f"Processed {min(i + batch_size, len(sentences))}/{len(sentences)}")

    return np.array(embeddings)

# Usage
sentences = load_documents()
embeddings = batch_encode(sentences, batch_size=128)
```

**Benefits:**
- ✅ 5x-10x faster than individual
- ✅ Better GPU utilization
- ✅ Reduced overhead

### 6.2 Batch Retrieval

```python
def batch_retrieve(queries, batch_size=50):
    """Batch retrieve for multiple queries."""
    all_results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]

        # Vectorize queries
        query_embeddings = model.encode(batch)

        # Batch search
        D, I = index.search(query_embeddings, k=10)

        all_results.append((D, I))

    return all_results
```

### 6.3 Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_batch_encode(sentences, batch_size=100, max_workers=4):
    """Async batch encoding."""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            task = loop.run_in_executor(
                executor,
                model.encode,
                batch
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    return np.vstack(results)

# Usage
embeddings = asyncio.run(async_batch_encode(sentences))
```

---

## 7. CONNECTION POOLING

### 7.1 Database Connection Pooling

**Reduce overhead** de database connections.

**PostgreSQL (pgvector):**
```python
from psycopg2 import pool
import os

# Create connection pool
connection_pool = pool.SimpleConnectionPool(
    1,  # min connections
    20,  # max connections
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

# Get connection from pool
conn = connection_pool.getconn()
cursor = conn.cursor()

# Use connection
cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
result = cursor.fetchone()

# Return connection to pool
connection_pool.putconn(conn)
```

**Redis Connection Pool:**
```python
import redis

# Create connection pool
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=20
)

# Use pool
redis_client = redis.Redis(connection_pool=redis_pool)
```

### 7.2 HTTP Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create session with retry and connection pooling
session = requests.Session()

# Retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

# Connection pooling
adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=retry_strategy
)

session.mount("http://", adapter)
session.mount("https://", adapter)

# Use session
response = session.get("https://api.example.com/data")
```

---

## 8. INDEX OPTIMIZATION

### 8.1 Optimize Vector DB Index

**HNSW Parameters:**
```python
# For production, tune these based on your data
index = faiss.IndexHNSWFlat(dim, 32)

# Higher efConstruction = better quality, slower build
index.hnsw.efConstruction = 200

# Higher efSearch = better recall, slower search
index.hnsw.efSearch = 64

# Add vectors
index.add(vectors)
```

**IVF Parameters:**
```python
# For large datasets, nlist ≈ sqrt(N)
nlist = int(np.sqrt(len(vectors)))
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

# Train on subset (not all vectors)
train_size = min(100 * nlist, len(vectors))
index.train(vectors[:train_size])

# Add all vectors
index.add(vectors)

# For search, nprobe = 10-100
index.nprobe = 10
```

### 8.2 Pre-filtering

**Filter** por metadata antes de similarity search.

```python
# Only search in specific category
filtered_docs = [
    doc for doc in all_docs
    if doc.metadata.get("category") == target_category
]

# Then search
results = vectorstore.similarity_search(
    query,
    k=10,
    filter={"category": target_category}
)
```

**Benefits:**
- ✅ Reduces search space
- ✅ Faster search
- ✅ Better precision

---

## 9. MONITORING & PROFILING

### 9.1 Performance Monitoring

```python
import time
import psutil
import logging

class RAGProfiler:
    def __init__(self):
        self.metrics = {
            "queries": 0,
            "total_time": 0,
            "avg_latency": 0,
            "throughput": 0
        }

    def log_query(self, query, start_time, end_time, num_results):
        """Log query performance."""
        latency = end_time - start_time

        self.metrics["queries"] += 1
        self.metrics["total_time"] += latency
        self.metrics["avg_latency"] = (
            self.metrics["total_time"] / self.metrics["queries"]
        )
        self.metrics["throughput"] = (
            1 / self.metrics["avg_latency"]
        )

        # Log
        logging.info(
            f"Query: {query[:50]}... | "
            f"Latency: {latency*1000:.1f}ms | "
            f"Results: {num_results}"
        )

    def get_metrics(self):
        """Get current metrics."""
        return {
            **self.metrics,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
```

### 9.2 Profiling

```python
import cProfile
import pstats

def profile_rag_function(func, *args, **kwargs):
    """Profile RAG function."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return result

# Usage
profile_rag_function(rag.query, "What is AI?")
```

---

## 10. WINDOWS-SPECIFIC OPTIMIZATIONS

### 10.1 PowerShell Optimizations

```powershell
# Set process priority
$process = Get-Process python
$process.PriorityClass = "High"

# Enable performance mode
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable Windows Defender real-time for Python directory
Add-MpPreference -ExclusionPath "C:\path\to\python\project"
```

### 10.2 WSL2 Optimization

```bash
# In WSL2, set max_map_count
sudo sysctl -w vm.max_map_count=262144

# Increase file limits
ulimit -n 65536

# Use tmpfs for temporary data
sudo mount -t tmpfs -o size=10G tmpfs /tmp/fast_storage
```

---

## 11. PERFORMANCE TUNING CHECKLIST

### 11.1 Database Level
- [ ] Use appropriate index (HNSW, IVF, IVF-PQ)
- [ ] Tune index parameters (efConstruction, nprobe, etc.)
- [ ] Enable connection pooling
- [ ] Use persistent connections
- [ ] Pre-filter when possible

### 11.2 Application Level
- [ ] Batch process operations
- [ ] Use caching (query, result, embedding)
- [ ] Enable GPU acceleration
- [ ] Use async/await for I/O
- [ ] Monitor and profile regularly

### 11.3 System Level
- [ ] Monitor CPU, memory, disk I/O
- [ ] Optimize network (connection pooling)
- [ ] Use SSD for storage
- [ ] Enable hardware acceleration
- [ ] Tune OS parameters (ulimit, etc.)

### 11.4 RAG-Specific
- [ ] Optimize chunk size
- [ ] Use hybrid retrieval
- [ ] Enable reranking only when needed
- [ ] Compress vectors if memory constrained
- [ ] Monitor retrieval quality

---

## 12. COMMON PITFALLS

### 12.1 Not Using Batch Processing
❌ **Problem**: Processing 1000 documents one by one
✅ **Solution**: Batch process with size 64-128

### 12.2 No Caching
❌ **Problem**: Recomputing embeddings for same queries
✅ **Solution**: Use Redis or in-memory cache

### 12.3 Wrong Index Type
❌ **Problem**: Using Flat index for 10M vectors
✅ **Solution**: Use HNSW or IVF for large datasets

### 12.4 No GPU Usage
❌ **Problem**: Slow embedding generation on CPU
✅ **Solution**: Use GPU with CUDA

### 12.5 Not Monitoring
❌ **Problem**: Don't know performance bottlenecks
✅ **Solution**: Profile and monitor regularly

---

## 13. PERFORMANCE BENCHMARKS

### 13.1 Typical Performance (Reference)

**Embedding Generation (BGE-large):**
- CPU (1 core): 10 docs/second
- CPU (8 cores): 50 docs/second
- GPU (RTX 4090): 500 docs/second

**Vector Search (1M vectors, k=10):**
- Flat: 100 QPS
- HNSW: 1,000 QPS
- IVF: 2,000 QPS
- IVF-PQ: 10,000 QPS

**End-to-End RAG Query:**
- Naive: 500ms
- Optimized: 50ms
- Ultra-optimized: 20ms

### 13.2 Scaling Guidelines

| Data Size | Index Type | Hardware | Expected QPS |
|-----------|------------|----------|--------------|
| < 1M | Flat | 8 CPU cores | 100 |
| 1M-10M | HNSW | 8 CPU cores | 1,000 |
| 10M-100M | IVF | 8 CPU cores | 2,000 |
| 100M+ | IVF-PQ | 8 CPU + GPU | 10,000 |

---

## 14. RECOMMENDATIONS

### 14.1 For Small Datasets (<1M vectors)
- Flat index or HNSW
- No compression needed
- Focus on query optimization
- Use caching

### 14.2 For Medium Datasets (1M-10M)
- HNSW index
- Connection pooling
- Batch processing
- Consider GPU for embedding

### 14.3 For Large Datasets (10M+)
- IVF or IVF-PQ
- Vector compression (PQ)
- GPU acceleration
- Async processing

### 14.4 For Real-time Applications
- Heavy caching
- Pre-computed embeddings
- Minimal filtering
- Fast index (HNSW, IVF-PQ)

---

**Status**: ✅ Base para Performance Optimization coletada
**Próximo**: Seção 08 - Advanced Patterns
**Data Conclusão**: 09/11/2025

# Relatório de Pesquisa: Seção 12 - Troubleshooting

### Data: 09/11/2025
### Status: Fase 4 - Advanced Topics

---

## 1. RESUMO EXECUTIVO

Troubleshooting é essential para manter sistemas RAG confiáveis e performing. Identificação rápida de problemas e solutions efetivas evitam downtime e impact negativo na user experience.

**Insights Chave:**
- **Common Issues**: Retrieval quality, performance, cost, availability
- **Debugging Tools**: Logging, profiling, tracing, monitoring
- **Error Handling**: Graceful degradation, fallbacks, retries
- **Performance Issues**: Latency, throughput, memory leaks
- **Root Cause Analysis**: Systematic approach
- **Prevention**: Best practices, monitoring, testing

---

## 2. COMMON ISSUES

### 2.1 Low Retrieval Quality

**Symptoms:**
- Irrelevant results returned
- Missing important information
- Poor answer quality
- User complaints

**Possible Causes:**

**1. Poor Chunking**
```python
# Problem: Chunks too small
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Too small
    chunk_overlap=0     # No overlap
)

# Solution: Increase chunk size and add overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Better
    chunk_overlap=200   # Preserves context
)
```

**2. Wrong Embedding Model**
```python
# Problem: Using model for wrong language/domain
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"  # English-only
    # For multilingual data, should use:
    # model_name="BAAI/bge-large-zh-v1.5"
)

# Solution: Use appropriate model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"  # Multilingual
)
```

**3. Missing Metadata**
```python
# Problem: No context in chunks
chunk.text = "The answer is 42."

# Solution: Add metadata
chunk.text = "The answer is 42."
chunk.metadata = {
    "section": "Chapter 5",
    "page": 42,
    "source": "book.pdf"
}
```

**4. Vector DB Configuration**
```python
# Problem: Wrong similarity metric
# Cosine for angle-based, L2 for distance-based

# For text embeddings (angle), use cosine
index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_COSINE)

# If using wrong metric, retraining might help
```

**Debugging Steps:**
```python
def debug_retrieval(query, top_k=5):
    # 1. Check retrieved documents
    docs = retriever.get_relevant_documents(query, k=top_k)

    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. Score: {doc.metadata.get('score', 'N/A')}")
        print(f"   Text: {doc.page_content[:200]}...")
        print(f"   Metadata: {doc.metadata}")

    # 2. Check query embedding
    query_embedding = embeddings.embed_query(query)
    print(f"\nQuery embedding shape: {query_embedding.shape}")
    print(f"Query embedding sample: {query_embedding[:10]}")

    # 3. Check similar documents in vector space
    # Find documents most similar to query
    all_docs = vectorstore.get()
    similarities = []
    for doc in all_docs:
        sim = cosine_similarity([query_embedding], [doc.embedding])[0][0]
        similarities.append((doc, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 most similar documents:")
    for doc, sim in similarities[:5]:
        print(f"  Score: {sim:.3f}")
        print(f"  Text: {doc.text[:100]}...")
```

### 2.2 High Latency

**Symptoms:**
- Slow response times (>2s)
- Timeout errors
- User complaints about speed
- Rate limiting

**Possible Causes:**

**1. Large Context**
```python
# Problem: Including too many documents
docs = retriever.get_relevant_documents(query, k=20)  # Too many

# Solution: Reduce k
docs = retriever.get_relevant_documents(query, k=5)  # Better
```

**2. Expensive LLM Model**
```python
# Problem: Using expensive model
llm = ChatOpenAI(model="gpt-4")  # Expensive

# Solution: Use cheaper model
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Faster, cheaper
```

**3. No Caching**
```python
# Problem: No caching
def query(question):
    docs = retriever.get_relevant_documents(question)
    response = llm.generate(question, context)
    return response

# Solution: Add caching
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_embedding(text):
    return embeddings.embed_query(text)

# Or use Redis
def query_with_cache(question):
    cache_key = f"query:{hashlib.md5(question.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    if cached:
        return cached

    docs = retriever.get_relevant_documents(question)
    response = llm.generate(question, context)
    redis_client.setex(cache_key, 3600, response)
    return response
```

**4. Synchronous Operations**
```python
# Problem: Synchronous processing
def process_document(doc):
    text = extract_text(doc)
    embedding = embeddings.embed_query(text)  # Sync
    vectorstore.add(embedding, text)

# Solution: Async/batch processing
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_embed_batch(texts):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    tasks = [
        loop.run_in_executor(executor, embeddings.embed_query, text)
        for text in texts
    ]
    embeddings = await asyncio.gather(*tasks)
    return embeddings
```

**Debugging Steps:**
```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

@timing_decorator
def query_with_timing(question):
    # Retrieval
    t0 = time.time()
    docs = retriever.get_relevant_documents(question)
    retrieval_time = time.time() - t0

    # Embedding
    t0 = time.time()
    query_embedding = embeddings.embed_query(question)
    embedding_time = time.time() - t0

    # Generation
    t0 = time.time()
    context = "\n".join([doc.text for doc in docs])
    response = llm.generate(context, question)
    generation_time = time.time() - t0

    total_time = retrieval_time + embedding_time + generation_time

    print(f"\nTiming breakdown:")
    print(f"  Retrieval: {retrieval_time:.3f}s")
    print(f"  Embedding: {embedding_time:.3f}s")
    print(f"  Generation: {generation_time:.3f}s")
    print(f"  Total: {total_time:.3f}s")

    return response
```

### 2.3 Out of Memory (OOM)

**Symptoms:**
- Application crashes
- Container restarts
- Memory usage 100%
- System slowdowns

**Possible Causes:**

**1. Large Embeddings Batch**
```python
# Problem: Processing too many at once
embeddings = model.encode(all_documents)  # All at once

# Solution: Batch processing
def batch_encode(texts, batch_size=100):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        yield batch_embeddings
```

**2. Memory Leaks**
```python
# Problem: Not cleaning up
docs = []
while True:
    new_docs = load_more()
    docs.extend(new_docs)  # Memory grows forever

# Solution: Circular buffer or cleanup
from collections import deque

docs = deque(maxlen=1000)  # Keep only last 1000
while True:
    new_docs = load_more()
    docs.extend(new_docs)  # Old items removed automatically
```

**3. Vector DB Memory**
```python
# Problem: Loading everything into memory
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(all_documents)  # All in memory

# Solution: Use persistent vector store
vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings
)
```

**Debugging Steps:**
```python
import psutil
import tracemalloc

def memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")

    # Get top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("\nTop 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
```

### 2.4 API Rate Limits

**Symptoms:**
- 429 errors
- Throttling messages
- Incomplete responses

**Possible Causes:**

**1. Too Many Requests**
```python
# Problem: Not respecting rate limits
for query in many_queries:
    results = openai.complete(query)  # May hit rate limit

# Solution: Add delays
import time
import random

def rate_limited_completion(query):
    for attempt in range(3):
        try:
            result = openai.complete(query)
            return result
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

**2. Token Limits**
```python
# Problem: Exceeding context window
context = "\n".join([doc.text for doc in all_docs])  # Too long

# Solution: Truncate or summarize
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,  # Stay under token limit
    chunk_overlap=0
)

chunks = splitter.split_text(context)
```

---

## 3. DEBUGGING TOOLS

### 3.1 Logging

**Configuration:**
```python
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Use in application
@app.get("/query")
def query(question: str):
    logger.info(f"Query received: {question}")

    try:
        start = time.time()
        result = rag_query(question)
        elapsed = time.time() - start

        logger.info(
            f"Query successful: {question[:50]}... "
            f"(took {elapsed:.2f}s, "
            f"returned {len(result)} chars)"
        )

        return result

    except Exception as e:
        logger.error(
            f"Query failed: {question[:50]}... "
            f"Error: {str(e)}",
            exc_info=True
        )
        raise
```

**Structured Logging:**
```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Use
logger.info(
    "Query processed",
    query=question,
    user_id=user.id,
    duration_ms=elapsed_ms,
    num_results=len(results)
)
```

### 3.2 Profiling

**Time Profiling:**
```python
import cProfile
import pstats
import io

def profile_function(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # Print to console
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        print(s.getvalue())

        return result
    return wrapper

@profile_function
def rag_query(question):
    # Your code here
    pass
```

**Memory Profiling:**
```python
from memory_profiler import profile

@profile
def process_documents():
    # Memory profiling line by line
    for doc in documents:
        text = extract_text(doc)
        embedding = model.encode(text)
        vectorstore.add(embedding, text)
```

### 3.3 Tracing

**LangSmith:**
```python
from langsmith import trace

# Trace individual functions
@trace("embed-query")
def embed_query(text):
    return embeddings.embed_query(text)

@trace("retrieve-documents")
def retrieve_documents(query):
    return vectorstore.similarity_search(query)

# Trace complete chain
with trace("rag-query"):
    embedding = embed_query(question)
    docs = retrieve_documents(question)
    context = "\n".join([doc.text for doc in docs])
    response = llm.generate(context, question)
```

**OpenTelemetry:**
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@app.get("/query")
def query(question: str):
    with tracer.start_as_current_span("rag-query") as span:
        span.set_attribute("question", question)

        # Retrieval
        with tracer.start_as_current_span("retrieval"):
            docs = retriever.search(question)

        # Generation
        with tracer.start_as_current_span("generation"):
            context = "\n".join([doc.text for doc in docs])
            response = llm.generate(context, question)

        span.set_attribute("response_length", len(response))
        return response
```

### 3.4 Debugging Tools

**Py-Spy (Sampling Profiler):**
```bash
# Install
pip install py-spy

# Profile running process
py-spy record -o profile.svg -- python app.py

# Profile specific process
py-spy record -o profile.svg --pid 12345
```

**Line Profiler:**
```bash
# Install
pip install line-profiler

# Add decorator
@profile
def expensive_function():
    # Your code
    pass

# Run profiler
kernprof -l -v script.py
```

---

## 4. ERROR HANDLING

### 4.1 Graceful Degradation

```python
from enum import Enum
from typing import Optional

class SystemStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"

def get_system_status() -> SystemStatus:
    """Check system health."""
    try:
        # Check database
        db_ok = check_database()

        # Check vector DB
        vector_db_ok = check_vector_db()

        # Check LLM
        llm_ok = check_llm()

        if all([db_ok, vector_db_ok, llm_ok]):
            return SystemStatus.HEALTHY
        elif any([db_ok, vector_db_ok, llm_ok]):
            return SystemStatus.DEGRADED
        else:
            return SystemStatus.DOWN

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemStatus.DOWN

@app.get("/query")
def query(question: str):
    status = get_system_status()

    if status == SystemStatus.HEALTHY:
        # Full functionality
        return full_rag_query(question)

    elif status == SystemStatus.DEGRADED:
        # Limited functionality
        logger.warning("System degraded, using fallback")
        return fallback_query(question)

    else:
        # Minimal functionality
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )
```

### 4.2 Fallback Strategies

```python
def query_with_fallback(question):
    """Try primary, fallback to secondary, then default."""
    # Primary: Vector search + LLM
    try:
        result = primary_rag_query(question)
        logger.info("Primary query successful")
        return {
            "answer": result,
            "method": "rag",
            "confidence": "high"
        }
    except Exception as e:
        logger.warning(f"Primary failed: {e}")

    # Secondary: BM25 search
    try:
        result = bm25_query(question)
        logger.info("Secondary query successful")
        return {
            "answer": result,
            "method": "bm25",
            "confidence": "medium"
        }
    except Exception as e:
        logger.warning(f"Secondary failed: {e}")

    # Default: Rule-based or cached
    result = get_cached_answer(question) or default_answer(question)
    return {
        "answer": result,
        "method": "cache/default",
        "confidence": "low"
    }
```

### 4.3 Circuit Breaker

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"       # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                # Success in half-open state
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

            raise e

# Usage
circuit_breaker = CircuitBreaker()

@circuit_breaker.call
def external_service_call():
    # Your call here
    pass
```

### 4.4 Retry Logic

```python
import time
import random
from functools import wraps

def retry(max_attempts=3, base_delay=1, max_delay=60, exponential=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RetryableError as e:
                    if attempt == max_attempts - 1:
                        raise

                    # Calculate delay
                    if exponential:
                        delay = base_delay * (2 ** attempt)
                    else:
                        delay = base_delay

                    # Add jitter
                    delay += random.uniform(0, 0.1 * delay)

                    delay = min(delay, max_delay)

                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

            return None
        return wrapper
    return decorator

class RetryableError(Exception):
    """Custom exception for retryable errors."""
    pass

@retry(max_attempts=3, base_delay=1, max_delay=10)
def api_call():
    # Your API call here
    pass
```

---

## 5. PERFORMANCE TROUBLESHOOTING

### 5.1 Slow Queries

**Investigation Steps:**

```python
def analyze_query_performance(question, num_runs=10):
    """Analyze query performance over multiple runs."""
    times = []

    for _ in range(num_runs):
        start = time.time()
        result = rag_query(question)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nQuery Performance ({num_runs} runs):")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min: {min_time:.3f}s")
    print(f"  Max: {max_time:.3f}s")
    print(f"  Std: {(sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5:.3f}s")

    # Percentiles
    sorted_times = sorted(times)
    p50 = sorted_times[int(len(sorted_times) * 0.5)]
    p90 = sorted_times[int(len(sorted_times) * 0.9)]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]

    print(f"\n  P50: {p50:.3f}s")
    print(f"  P90: {p90:.3f}s")
    print(f"  P95: {p95:.3f}s")
    print(f"  P99: {p99:.3f}s")
```

**Common Causes:**

1. **Inefficient Retrieval:**
```python
# Find slow retrievers
def compare_retrievers(question, k=5):
    retrievers = {
        'flat': FlatRetriever(),
        'hnsw': HNSWRetriever(),
        'ivf': IVFRetriever()
    }

    results = {}
    for name, retriever in retrievers.items():
        start = time.time()
        docs = retriever.search(question, k=k)
        elapsed = time.time() - start
        results[name] = {
            'time': elapsed,
            'num_results': len(docs)
        }

    return results
```

2. **Large Context:**
```python
# Analyze context size
def analyze_context_size(questions):
    sizes = []
    for question in questions:
        docs = retriever.search(question, k=5)
        context = "\n".join([doc.text for doc in docs])
        sizes.append(len(context))

    print(f"\nContext Size Analysis:")
    print(f"  Average: {sum(sizes)/len(sizes):.0f} chars")
    print(f"  Max: {max(sizes):.0f} chars")
    print(f"  Min: {min(sizes):.0f} chars")

    # Find outliers
    avg = sum(sizes) / len(sizes)
    outliers = [s for s in sizes if s > avg * 2]
    if outliers:
        print(f"\n  {len(outliers)} outliers (>2x average)")
```

### 5.2 Memory Issues

```python
import tracemalloc
import gc

def memory_snapshots():
    # Take baseline
    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    # Run operation
    rag_query("test question")

    # Take second snapshot
    current = tracemalloc.take_snapshot()

    # Calculate difference
    top_stats = current.compare_to(baseline, 'lineno')

    print("Top memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
```

### 5.3 CPU Profiling

```python
import cProfile
import pstats

def cpu_profile():
    """Profile CPU usage."""
    pr = cProfile.Profile()
    pr.enable()

    # Run workload
    for _ in range(100):
        rag_query("What is AI?")

    pr.disable()

    # Analyze
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print(s.getvalue())
```

---

## 6. MONITORING & ALERTING

### 6.1 Key Metrics

```python
# Metrics to monitor
METRICS = {
    "query_rate": "Rate of queries per second",
    "query_latency": "Query response time (P50, P95, P99)",
    "error_rate": "Percentage of failed queries",
    "throughput": "Queries per minute",
    "cpu_usage": "CPU utilization",
    "memory_usage": "Memory consumption",
    "gpu_usage": "GPU utilization (if using)",
    "queue_length": "Length of processing queue",
    "cache_hit_rate": "Cache hit percentage",
    "vector_db_size": "Number of vectors stored"
}
```

### 6.2 Alert Rules

```python
# Prometheus alert rules
ALERT_RULES = """
groups:
- name: rag_alerts
  rules:
  - alert: HighQueryLatency
    expr: histogram_quantile(0.95, rag_request_duration_seconds) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High query latency detected"
      description: "95th percentile latency is {{ $value }}s"

  - alert: HighErrorRate
    expr: rate(rag_requests_total{status="error"}[5m]) / rate(rag_requests_total[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate"
      description: "Error rate is {{ $value | humanizePercentage }}"

  - alert: HighMemoryUsage
    expr: (process_resident_memory_bytes / 1024 / 1024 / 1024) > 4
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}GB"
"""
```

### 6.3 Health Checks

```python
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI()

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ready")
def readiness_check() -> Dict[str, Any]:
    """Readiness probe."""
    checks = {
        "database": check_database(),
        "vector_db": check_vector_db(),
        "openai": check_openai()
    }

    all_healthy = all(checks.values())

    return {
        "status": "ready" if all_healthy else "not ready",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }

def check_database() -> bool:
    try:
        # Try a simple query
        result = db.execute("SELECT 1").fetchone()
        return result is not None
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False
```

---

## 7. LOG ANALYSIS

### 7.1 Structured Logs

```python
import json
from datetime import datetime

def log_query(question: str, user_id: str, result: str, latency: float):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": "query",
        "question": question[:100],  # Truncate for privacy
        "user_id": user_id,
        "result_length": len(result),
        "latency_ms": latency * 1000,
        "status": "success"
    }

    # Write to log file
    with open('queries.log', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

### 7.2 Log Analysis Queries

```python
# Analyze slow queries
import pandas as pd
import json

def analyze_slow_queries(log_file='queries.log', threshold_ms=2000):
    queries = []
    with open(log_file) as f:
        for line in f:
            queries.append(json.loads(line))

    df = pd.DataFrame(queries)
    slow_queries = df[df['latency_ms'] > threshold_ms]

    print(f"\nSlow queries (>{threshold_ms}ms): {len(slow_queries)}")
    print(slow_queries[['timestamp', 'latency_ms']].sort_values('latency_ms', ascending=False))

    # Most common slow queries
    slow_counts = slow_queries['question'].value_counts()
    print(f"\nMost common slow queries:")
    print(slow_counts.head(10))
```

---

## 8. DEBUGGING CHECKLISTS

### 8.1 Retrieval Quality Issues

**Checklist:**
- [ ] Chunk size appropriate (500-1500)
- [ ] Chunk overlap set (10-20% of chunk size)
- [ ] Proper embedding model selected
- [ ] Vector DB index configured correctly
- [ ] Metadata includes source/context
- [ ] Query preprocessing consistent
- [ ] Similarity metric appropriate (cosine/L2)
- [ ] k value appropriate (3-10)

### 8.2 Performance Issues

**Checklist:**
- [ ] Batch processing enabled
- [ ] Caching implemented
- [ ] Async/await used where applicable
- [ ] Resource limits set
- [ ] Auto-scaling configured
- [ ] Query optimization applied
- [ ] Index optimized (HNSW, IVF, etc.)
- [ ] Hardware appropriate (CPU/GPU)

### 8.3 Availability Issues

**Checklist:**
- [ ] Health checks configured
- [ ] Readiness probes set
- [ ] Liveness probes configured
- [ ] Auto-restart policies set
- [ ] Load balancer configured
- [ ] Circuit breakers implemented
- [ ] Retry logic added
- [ ] Graceful degradation implemented

### 8.4 Cost Issues

**Checklist:**
- [ ] Usage monitoring enabled
- [ ] Budget alerts set
- [ ] Expensive models reviewed
- [ ] Caching rate checked
- [ ] Batch processing optimized
- [ ] Request size limited
- [ ] Token usage tracked
- [ ] Off-hours scaling enabled

---

## 9. ROOT CAUSE ANALYSIS

### 9.1 5 Whys Method

**Example:**

**Problem**: Users report slow responses (>5s)

1. **Why is the response slow?**
   → LLM generation takes 4s

2. **Why does LLM generation take 4s?**
   → Large context (5000 tokens)

3. **Why is the context 5000 tokens?**
   → Retrieving 20 documents, each ~250 tokens

4. **Why retrieving 20 documents?**
   → Default k=20 in code

5. **Why k=20?**
   → Never optimized, copied from example

**Solution**: Reduce k to 5

### 9.2 Fishbone Diagram

**Categories:**
- **People**: Skill level, training, errors
- **Process**: Procedures, workflows, approvals
- **Technology**: Tools, systems, infrastructure
- **Materials**: Data quality, sources, formats
- **Environment**: Load, network, timing
- **Measurement**: Metrics, accuracy, frequency

### 9.3 Timeline Analysis

```python
def create_timeline(event_log):
    """Create timeline of events."""
    events = [
        ("request_received", event['request_time']),
        ("retrieval_start", event['retrieval_start_time']),
        ("retrieval_end", event['retrieval_end_time']),
        ("generation_start", event['generation_start_time']),
        ("generation_end", event['generation_end_time']),
        ("response_sent", event['response_time'])
    ]

    print("\nTimeline:")
    for event_name, timestamp in events:
        print(f"  {timestamp}: {event_name}")

    # Calculate durations
    retrieval_duration = event['retrieval_end_time'] - event['retrieval_start_time']
    generation_duration = event['generation_end_time'] - event['generation_start_time']

    print(f"\nDurations:")
    print(f"  Retrieval: {retrieval_duration:.3f}s")
    print(f"  Generation: {generation_duration:.3f}s")
    print(f"  Total: {retrieval_duration + generation_duration:.3f}s")
```

---

## 10. PREVENTION

### 10.1 Best Practices

1. **Test Early and Often**
   - Unit tests
   - Integration tests
   - Load tests
   - Chaos engineering

2. **Monitor Everything**
   - Application metrics
   - Business metrics
   - User experience metrics
   - System metrics

3. **Document Everything**
   - Architecture decisions
   - API documentation
   - Runbooks
   - Post-mortems

4. **Automate Testing**
   - CI/CD pipelines
   - Automated deployments
   - Automated rollbacks
   - Automated backups

5. **Plan for Failure**
   - Circuit breakers
   - Retry logic
   - Graceful degradation
   - Fallback strategies

### 10.2 Observability Stack

```python
# Complete observability setup
OBSERVABILITY_CONFIG = {
    "metrics": {
        "prometheus": {
            "enabled": True,
            "port": 9090
        },
        "grafana": {
            "enabled": True,
            "port": 3000
        }
    },
    "logging": {
        "structlog": {
            "enabled": True,
            "level": "INFO"
        },
        "file": {
            "enabled": True,
            "path": "/var/log/rag/app.log"
        }
    },
    "tracing": {
        "langsmith": {
            "enabled": True,
            "project": "rag-production"
        },
        "opentelemetry": {
            "enabled": True,
            "exporter": "jaeger"
        }
    },
    "alerting": {
        "prometheus_alertmanager": {
            "enabled": True,
            "smtp_config": "..."
        }
    }
}
```

---

## 11. INCIDENT RESPONSE

### 11.1 Runbook Template

```markdown
# Incident Response Runbook

## Detection
- How was the incident detected?
- What triggered the alert?

## Assessment
- Severity (P0-P4)
- Impact scope
- Affected users

## Investigation
- Timeline
- Key findings
- Root cause (if known)

## Resolution
- Actions taken
- Time to resolution
- Temporary fixes
- Permanent fixes

## Post-Mortem
- What went well
- What went wrong
- Action items
- Lessons learned
```

### 11.2 Escalation Matrix

```python
ESCALATION_MATRIX = {
    "P0_Critical": {
        "response_time": "15 minutes",
        "escalation": ["oncall", "manager", "director"],
        "communication": ["slack", "email", "pager"]
    },
    "P1_High": {
        "response_time": "1 hour",
        "escalation": ["oncall", "team_lead"],
        "communication": ["slack", "email"]
    },
    "P2_Medium": {
        "response_time": "4 hours",
        "escalation": ["team_lead"],
        "communication": ["ticket"]
    },
    "P3_Low": {
        "response_time": "1 day",
        "escalation": ["ticket"],
        "communication": ["ticket"]
    }
}
```

---

**Status**: ✅ Base para Troubleshooting coletada
**Próximo**: Consolidar Fase 4
**Data Conclusão**: 09/11/2025

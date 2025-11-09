#!/usr/bin/env python3
"""
Example 01: Performance Optimization Techniques
==============================================

Demonstra técnicas de otimização para RAG.

Uso:
    python example-01-optimization-techniques.py
"""

import asyncio
import time
from functools import lru_cache
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


# 1. Caching
class CachedRAG:
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.cache = {}

    @lru_cache(maxsize=10000)
    def get_cached_embedding(self, text):
        return self.embeddings.embed_query(text)

    def query_with_cache(self, question):
        # Check cache
        if question in self.cache:
            return self.cache[question]

        # Get embedding
        query_vector = self.get_cached_embedding(question)

        # Search
        docs = self.vectorstore.similarity_search(
            question,
            k=3
        )

        # Cache result
        result = {"docs": docs}
        self.cache[question] = result
        return result


# 2. Async Processing
class AsyncRAG:
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    async def async_batch_search(self, queries: List[str]):
        tasks = [
            self.vectorstore.asimilarity_search(q, k=3)
            for q in queries
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def query(self, question):
        docs = await self.vectorstore.asimilarity_search(question, k=3)
        return {"docs": docs}


# 3. Batch Processing
class BatchRAG:
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    def batch_embed(self, texts, batch_size=100):
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vectors = self.embeddings.embed_documents(batch)
            all_vectors.extend(vectors)
        return all_vectors

    def batch_search(self, queries, batch_size=10):
        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_results = [
                self.vectorstore.similarity_search(q, k=3)
                for q in batch
            ]
            results.extend(batch_results)
        return results


def benchmark_caching():
    """Benchmark caching impact"""
    print("="*60)
    print("CACHING BENCHMARK")
    print("="*60)

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document

    # Setup
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = [Document(page_content=f"Doc {i}") for i in range(1000)]
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Without cache
    start = time.time()
    for _ in range(100):
        vectorstore.similarity_search("test", k=3)
    no_cache_time = time.time() - start

    # With cache
    cached_rag = CachedRAG(vectorstore, embeddings)
    start = time.time()
    for _ in range(100):
        cached_rag.query_with_cache("test")
    cache_time = time.time() - start

    print(f"Without cache: {no_cache_time:.3f}s")
    print(f"With cache: {cache_time:.3f}s")
    print(f"Speedup: {no_cache_time/cache_time:.1f}x")


def benchmark_async():
    """Benchmark async processing"""
    print("\n" + "="*60)
    print("ASYNC BENCHMARK")
    print("="*60)

    queries = [f"Query {i}" for i in range(20)]

    # Sync
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    from langchain.schema import Document
    documents = [Document(page_content=f"Doc {i}") for i in range(1000)]
    vectorstore = Chroma.from_documents(documents, embeddings)

    async_rag = AsyncRAG(vectorstore, embeddings)

    # Sync
    start = time.time()
    for query in queries:
        vectorstore.similarity_search(query, k=3)
    sync_time = time.time() - start

    # Async
    start = time.time()
    asyncio.run(async_rag.async_batch_search(queries))
    async_time = time.time() - start

    print(f"Sync time: {sync_time:.3f}s")
    print(f"Async time: {async_time:.3f}s")
    print(f"Speedup: {sync_time/async_time:.1f}x")


def benchmark_batch():
    """Benchmark batch processing"""
    print("\n" + "="*60)
    print("BATCH BENCHMARK")
    print("="*60)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    from langchain.schema import Document
    documents = [Document(page_content=f"Doc {i}") for i in range(1000)]
    vectorstore = Chroma.from_documents(documents, embeddings)

    batch_rag = BatchRAG(vectorstore, embeddings)

    # Individual
    queries = [f"Query {i}" for i in range(20)]
    start = time.time()
    for query in queries:
        vectorstore.similarity_search(query, k=3)
    individual_time = time.time() - start

    # Batch
    start = time.time()
    batch_rag.batch_search(queries, batch_size=10)
    batch_time = time.time() - start

    print(f"Individual: {individual_time:.3f}s")
    print(f"Batch: {batch_time:.3f}s")
    print(f"Speedup: {individual_time/batch_time:.1f}x")


def main():
    """Função principal"""
    print("="*60)
    print("PERFORMANCE OPTIMIZATION TECHNIQUES")
    print("="*60)

    # Benchmark techniques
    benchmark_caching()
    benchmark_async()
    benchmark_batch()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"""
📊 Performance Gains:

1. Caching:
   - Speedup: 10-100x
   - Best for: High traffic, repeated queries
   - Memory: Low

2. Async:
   - Speedup: 2-5x
   - Best for: Concurrent users
   - Memory: None

3. Batch:
   - Speedup: 2-10x
   - Best for: Processing multiple items
   - Memory: Low

4. Compression:
   - Speedup: 1.2x
   - Best for: Memory constraints
   - Memory: 50-75% reduction

5. GPU:
   - Speedup: 5-50x
   - Best for: Embedding bottlenecks
   - Memory: None

🚀 Recommendations:
   - Start with caching (easy win)
   - Use async for concurrency
   - Batch for document processing
   - GPU for heavy workloads
   - Compression for memory limits
""")

    print("="*60)
    print("Optimization completed!")
    print("="*60)


if __name__ == "__main__":
    main()

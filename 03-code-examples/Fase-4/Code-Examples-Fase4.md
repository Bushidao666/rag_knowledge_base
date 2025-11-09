# Code Examples: Fase 4 (Seções 07-12)

### Data: 09/11/2025
### Status: Executáveis no Windows
### Foco: Advanced Topics - Performance, Patterns, Deployment

---

## EXAMPLE 1: Performance Optimization (Seção 07)

### Prerequisites

```bash
pip install faiss-cpu torch sentence-transformers redis
```

### Vector Compression & GPU Acceleration

```python
"""
Performance Optimization: Compression, Caching, GPU Acceleration
Demonstra PQ compression, batch processing, e caching strategies
"""

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import time
import redis
import hashlib
from functools import lru_cache

# ===== VECTOR COMPRESSION =====

class VectorCompressor:
    """Vector compression using FAISS PQ, SQ, Binary"""

    def __init__(self, dimension: int, compression_type: str = "pq", m: int = 64):
        self.dimension = dimension
        self.compression_type = compression_type
        self.m = m  # Number of sub-vectors for PQ
        self.index = None
        self.is_trained = False

    def create_pq_index(self, vectors: np.ndarray):
        """Create Product Quantization index."""
        print(f"Creating PQ index: {self.m} sub-vectors")

        # Create PQ index
        nbits = 8  # 8 bits per sub-vector
        self.index = faiss.IndexPQ(
            self.dimension,
            self.m,
            nbits
        )

        # Train on subset
        n_train = min(100000, len(vectors))
        print(f"Training PQ on {n_train} vectors...")
        self.index.train(vectors[:n_train])

        # Add all vectors
        self.index.add(vectors)
        self.is_trained = True

        # Calculate compression ratio
        original_size = len(vectors) * self.dimension * 4  # float32 = 4 bytes
        compressed_size = len(vectors) * self.m  # m bytes
        ratio = original_size / compressed_size

        print(f"Compression ratio: {ratio:.1f}x")
        return ratio

    def create_sq_index(self, vectors: np.ndarray, bits: int = 8):
        """Create Scalar Quantization index."""
        print(f"Creating SQ index: {bits} bits per dimension")

        if bits == 8:
            qt = faiss.ScalarQuantizer.QT_8bit
        elif bits == 4:
            qt = faiss.ScalarQuantizer.QT_4bit
        else:
            qt = faiss.ScalarQuantizer.QT_2bit

        self.index = faiss.IndexScalarQuantizer(
            self.dimension,
            qt
        )

        # Train and add
        n_train = min(100000, len(vectors))
        self.index.train(vectors[:n_train])
        self.index.add(vectors)
        self.is_trained = True

        # Compression ratio
        original_size = len(vectors) * self.dimension * 4
        compressed_size = len(vectors) * self.dimension * (bits / 8)
        ratio = original_size / compressed_size

        print(f"Compression ratio: {ratio:.1f}x")
        return ratio

    def search(self, query_vectors: np.ndarray, k: int = 10):
        """Search with compressed index."""
        if not self.is_trained:
            raise ValueError("Index not trained")

        D, I = self.index.search(query_vectors, k)
        return D, I

# ===== BATCH PROCESSING =====

class BatchProcessor:
    """Efficient batch processing for embeddings"""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)

    def batch_encode(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """Batch encode texts for efficiency."""
        print(f"Batch encoding {len(texts)} texts (batch size: {batch_size})")

        all_embeddings = []
        start_time = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Move batch to device
            batch_tensors = self.model.tokenize(batch)
            batch_tensors = batch_tensors.to(self.device)

            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch_tensors)

            all_embeddings.extend(batch_embeddings.cpu().numpy())

        elapsed = time.time() - start_time
        throughput = len(texts) / elapsed

        print(f"Encoded {len(texts)} texts in {elapsed:.2f}s ({throughput:.1f} texts/s)")

        return np.array(all_embeddings)

    def parallel_batch_encode(self, texts: list, num_workers: int = 4) -> np.ndarray:
        """Parallel batch encoding with multiple workers."""
        from concurrent.futures import ThreadPoolExecutor

        print(f"Parallel encoding with {num_workers} workers")

        # Split texts into chunks
        chunk_size = (len(texts) + num_workers - 1) // num_workers
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            embeddings_list = list(executor.map(self.batch_encode, chunks))

        all_embeddings = np.vstack(embeddings_list)

        elapsed = time.time() - start_time
        throughput = len(texts) / elapsed

        print(f"Parallel encoded {len(texts)} texts in {elapsed:.2f}s ({throughput:.1f} texts/s)")

        return all_embeddings

# ===== CACHING STRATEGIES =====

class CachingRAG:
    """RAG with multiple caching layers"""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

        # In-memory LRU cache
        self.lru_cache = {}
        self.cache_size = 1000

        # Embedding model
        self.embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return f"query:{hashlib.md5(query.encode()).hexdigest()}"

    def get_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        return f"embedding:{hashlib.md5(text.encode()).hexdigest()}"

    @lru_cache(maxsize=1000)
    def get_embedding_lru(self, text: str) -> np.ndarray:
        """Get embedding with LRU cache."""
        return self.embeddings.encode(text)

    def get_embedding_redis(self, text: str) -> np.ndarray:
        """Get embedding with Redis cache."""
        cache_key = self.get_embedding_cache_key(text)

        # Try cache
        cached = self.redis_client.get(cache_key)
        if cached:
            return np.frombuffer(
                bytes.fromhex(cached),
                dtype=np.float32
            )

        # Compute and cache
        embedding = self.embeddings.encode(text)
        embedding_bytes = embedding.tobytes().hex()
        self.redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            embedding_bytes
        )

        return embedding

    def query_with_cache(self, question: str) -> tuple:
        """Query with full caching."""
        start = time.time()

        # Check query cache
        query_key = self.get_cache_key(question)
        cached_result = self.redis_client.get(query_key)

        if cached_result:
            print("✅ Query cache hit!")
            elapsed = time.time() - start
            return cached_result, elapsed

        # Retrieve similar (simulated)
        # In real implementation, would use vector store
        time.sleep(0.1)  # Simulate retrieval

        # Generate answer (simulated)
        time.sleep(0.2)  # Simulate generation

        result = f"Answer to: {question}"

        # Cache result
        self.redis_client.setex(
            query_key,
            1800,  # 30 min TTL
            result
        )

        elapsed = time.time() - start
        print(f"⏱️  Query took {elapsed:.3f}s")

        return result, elapsed

# ===== PERFORMANCE COMPARISON =====

def compare_performance():
    """Compare different optimization techniques."""
    # Sample data
    texts = [f"Document {i}: Sample text for testing performance." for i in range(1000)]
    query = "What is the main topic?"

    print("="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    # 1. Without optimization
    print("\n1. Naive approach:")
    start = time.time()
    for text in texts[:100]:  # First 100
        embedding = SentenceTransformer("BAAI/bge-large-en-v1.5").encode(text)
    naive_time = time.time() - start
    print(f"   Time: {naive_time:.2f}s")

    # 2. With batch processing
    print("\n2. Batch processing:")
    batch_processor = BatchProcessor()
    start = time.time()
    embeddings = batch_processor.batch_encode(texts[:100], batch_size=32)
    batch_time = time.time() - start
    speedup = naive_time / batch_time
    print(f"   Time: {batch_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x")

    # 3. With caching
    print("\n3. With caching:")
    rag = CachingRAG()
    start = time.time()

    # First call (cache miss)
    result1, time1 = rag.query_with_cache(query)
    print(f"   First call: {time1:.3f}s")

    # Second call (cache hit)
    result2, time2 = rag.query_with_cache(query)
    print(f"   Second call: {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.1f}x")

    # 4. Vector compression
    print("\n4. Vector compression:")
    vectors = np.random.rand(10000, 1024).astype('float32')

    # Create indexes
    pq_compressor = VectorCompressor(1024, "pq", m=64)
    start = time.time()
    pq_ratio = pq_compressor.create_pq_index(vectors)
    compress_time = time.time() - start

    # Search
    query_vec = np.random.rand(1, 1024).astype('float32')
    start = time.time()
    D, I = pq_compressor.search(query_vec, k=10)
    search_time = time.time() - start

    print(f"   Compression time: {compress_time:.2f}s")
    print(f"   Search time: {search_time*1000:.2f}ms")
    print(f"   Compression ratio: {pq_ratio:.1f}x")

# Usage example
if __name__ == "__main__":
    try:
        compare_performance()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Some features require:")
        print("  - Redis server running (for caching)")
        print("  - GPU (for GPU acceleration)")
```

---

## EXAMPLE 2: Advanced Patterns (Seção 08)

### Multimodal RAG Implementation

```python
"""
Advanced RAG Patterns: Multimodal, Agentic, Graph
Demonstra CLIP, LLaVA, knowledge graphs, agentic RAG
"""

import torch
import clip
from PIL import Image
import json
import asyncio
from typing import List, Dict, Any

# ===== MULTIMODAL RAG (CLIP) =====

class MultimodalRAG:
    """RAG com suporte a texto e imagens usando CLIP"""

    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text queries."""
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode single image."""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()

    def search(self, query: str, images: List[str], k: int = 5) -> List[Dict]:
        """Search images based on text query."""
        # Encode query
        query_features = self.encode_text([query])

        # Encode all images
        image_features = []
        for img_path in images:
            features = self.encode_image(img_path)
            image_features.append(features[0])

        image_features = np.stack(image_features)

        # Calculate similarities
        similarities = np.dot(query_features.cpu().numpy(), image_features.T)[0]

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "image": images[idx],
                "similarity": float(similarities[idx]),
                "rank": len(results) + 1
            })

        return results

# ===== GRAPH RAG =====

class GraphRAG:
    """Knowledge Graph-based RAG"""

    def __init__(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )

    def build_graph(self, documents: List[Dict]):
        """Build knowledge graph from documents."""
        with self.driver.session() as session:
            # Clear existing
            session.run("MATCH (n) DETACH DELETE n")

            # Create nodes and relationships
            for doc in documents:
                # Create entity nodes
                for entity in doc.get("entities", []):
                    session.run(
                        """
                        MERGE (e:Entity {name: $name, type: $type})
                        """,
                        name=entity["name"],
                        type=entity["type"]
                    )

                # Create relationships
                for rel in doc.get("relationships", []):
                    session.run(
                        """
                        MATCH (a:Entity {name: $subject})
                        MATCH (b:Entity {name: $object})
                        MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
                        """,
                        subject=rel["subject"],
                        object=rel["object"],
                        rel_type=rel["type"]
                    )

    def query_graph(self, question: str, max_hops: int = 2) -> List[Dict]:
        """Query knowledge graph with multi-hop reasoning."""
        from extract import extract_entities

        # Extract entities from question
        entities = extract_entities(question)

        results = []
        with self.driver.session() as session:
            for entity in entities:
                # Find related entities
                result = session.run(
                    f"""
                    MATCH (start:Entity)-[r*1..{max_hops}]-(related:Entity)
                    WHERE start.name CONTAINS $entity
                    RETURN start, related, length(path) as hops
                    ORDER BY hops
                    LIMIT 10
                    """,
                    entity=entity["text"]
                ).data()

                results.extend(result)

        return results

# ===== AGENTIC RAG =====

class AgenticRAG:
    """Agentic RAG com multi-step reasoning"""

    def __init__(self, rag_system, llm):
        self.rag_system = rag_system
        self.llm = llm

    async def query_with_reasoning(self, question: str, max_steps: int = 3) -> Dict:
        """Query com step-by-step reasoning."""
        history = []
        current_question = question

        for step in range(max_steps):
            print(f"\nStep {step + 1}:")

            # 1. Plan
            plan = await self.plan_step(current_question, history)
            print(f"  Plan: {plan}")

            # 2. Act
            if "search" in plan:
                result = await self.rag_system.search(current_question)
                print(f"  Search: Retrieved {len(result)} documents")
            elif "analyze" in plan:
                result = await self.analyze_result(history[-1] if history else None)
                print(f"  Analyze: {result[:100]}...")
            else:
                result = await self.generate_answer(current_question, history)
                print(f"  Generate: {result[:100]}...")

            history.append({
                "step": step + 1,
                "plan": plan,
                "result": result
            })

            # 3. Reflect
            is_complete = await self.reflect(result, question)
            if is_complete:
                break

            # Update question for next iteration
            current_question = await self.refine_question(question, history)

        # Final synthesis
        final_answer = await self.synthesize(history, question)
        return final_answer

    async def plan_step(self, question: str, history: List) -> str:
        """Plan next action."""
        prompt = f"""
        Current question: {question}
        History: {history}

        Plan next action:
        - search: Find relevant documents
        - analyze: Analyze current information
        - generate: Generate final answer
        - refine: Ask follow-up question

        Respond with ONE word: search, analyze, generate, or refine
        """
        return await self.llm.generate(prompt)

    async def search(self, query: str) -> List[Dict]:
        """Search for relevant documents."""
        return self.rag_system.search(query, k=5)

    async def analyze(self, data: Any) -> str:
        """Analyze current data."""
        prompt = f"""
        Analyze the following information:
        {data}

        Provide key insights:
        """
        return await self.llm.generate(prompt)

    async def generate_answer(self, question: str, history: List) -> str:
        """Generate final answer."""
        context = "\n".join([
            f"Step {h['step']}: {h['result'][:200]}"
            for h in history
        ])

        prompt = f"""
        Question: {question}
        Context: {context}

        Provide a comprehensive answer:
        """
        return await self.llm.generate(prompt)

    async def reflect(self, result: str, original_question: str) -> bool:
        """Check if we have enough information."""
        prompt = f"""
        Question: {original_question}
        Result: {result}

        Is this answer complete? (yes/no)
        """
        response = await self.llm.generate(prompt)
        return "yes" in response.lower()

# ===== FUSION RAG =====

class FusionRAG:
    """FAG com multiple query variations"""

    def __init__(self, rag_system, llm):
        self.rag_system = rag_system
        self.llm = llm

    async def generate_variations(self, question: str) -> List[str]:
        """Generate query variations."""
        prompt = f"""
        Generate 3 different variations of this question:

        Original: {question}

        Provide variations that ask the same question differently:
        """
        variations = await self.llm.generate(prompt)
        # Parse variations (simplified)
        return [question, variations[:100], variations[100:200]]

    def fuse_results(self, results_list: List[List[Dict]]) -> List[Dict]:
        """Fuse results from multiple queries using RRF."""
        from collections import Counter

        # Reciprocal Rank Fusion
        scores = Counter()
        for results in results_list:
            for rank, result in enumerate(results, 1):
                # Use document ID as key
                doc_id = result.get("id", rank)
                scores[doc_id] += 1 / (rank + 60)

        # Sort by fused score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get top results
        top_results = []
        for doc_id, score in fused[:5]:
            # Find original result
            for results in results_list:
                for result in results:
                    if result.get("id") == doc_id or result in results:
                        top_results.append(result)
                        break
                if len(top_results) >= 5:
                    break
            if len(top_results) >= 5:
                break

        return top_results

    async def query(self, question: str) -> Dict:
        """Query with fusion."""
        # 1. Generate variations
        variations = await self.generate_variations(question)

        # 2. Retrieve for each variation
        all_results = []
        for variation in variations:
            results = await self.rag_system.search(variation)
            all_results.append(results)

        # 3. Fuse results
        fused_results = self.fuse_results(all_results)

        # 4. Generate final answer
        context = "\n".join([
            f"Document {i+1}: {result.get('text', '')}"
            for i, result in enumerate(fused_results)
        ])

        final_answer = await self.rag_system.generate(question, context)

        return {
            "answer": final_answer,
            "fused_results": fused_results,
            "variations_used": variations
        }

# Usage examples
if __name__ == "__main__":
    # Multimodal RAG
    print("="*60)
    print("MULTIMODAL RAG EXAMPLE")
    print("="*60)

    multimodal_rag = MultimodalRAG()

    # Search images with text query
    images = ["img1.jpg", "img2.jpg", "img3.jpg"]
    results = multimodal_rag.search(
        "A red car in a parking lot",
        images,
        k=2
    )

    print("\nTop results:")
    for result in results:
        print(f"  {result['rank']}. {result['image']} (similarity: {result['similarity']:.3f})")

    # Note: Requires actual image files and Neo4j for full functionality
```

---

## EXAMPLE 3: Architecture Patterns (Seção 09)

### Modular RAG Implementation

```python
"""
Architecture Patterns: Modular RAG, Chunk-Join, Parent-Document
Implement different RAG patterns de forma modular
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

# ===== INTERFACES =====

class Retriever(ABC):
    """Retriever interface"""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict]:
        pass

class Generator(ABC):
    """Generator interface"""

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        pass

class Reranker(ABC):
    """Reranker interface"""

    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], k: int = 5) -> List[Dict]:
        pass

# ===== IMPLEMENTATIONS =====

class NaiveRetriever(Retriever):
    """Simple retriever implementation"""

    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model

    def search(self, query: str, k: int = 5) -> List[Dict]:
        # Simple similarity search
        results = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                "id": f"doc_{i}",
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0)
            }
            for i, doc in enumerate(results)
        ]

class LLMGenerator(Generator):
    """LLM-based generator"""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
        Question: {question}

        Context: {context}

        Answer:
        """
        return self.llm.generate(prompt)

class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict], k: int = 5) -> List[Dict]:
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Get scores from cross-encoder
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:k]

# ===== CHUNK-JOIN RAG =====

class ChunkJoinRAG:
    """Chunk-Join RAG pattern"""

    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        self.document_store = {}  # Store full documents

    def index(self, documents: List[Dict]):
        """Index documents with parent tracking."""
        for doc in documents:
            # Store full document
            self.document_store[doc["id"]] = doc

            # Create chunks
            chunks = self.create_chunks(doc)

            # Store chunks with parent reference
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                self.retriever.vectorstore.add({
                    "id": chunk_id,
                    "text": chunk["text"],
                    "metadata": {
                        "parent_id": doc["id"],
                        "chunk_index": i
                    }
                })

    def create_chunks(self, document: Dict, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """Create chunks with overlap."""
        text = document["text"]
        chunks = []

        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            chunks.append({
                "id": f"chunk_{i//(chunk_size - overlap)}",
                "text": chunk_text,
                "start": i,
                "end": i + len(chunk_text)
            })

        return chunks

    def query(self, question: str, k: int = 5) -> str:
        """Query with chunk-join."""
        # 1. Retrieve initial chunks
        initial_chunks = self.retriever.search(question, k=10)

        # 2. Group by parent
        parent_groups = {}
        for chunk in initial_chunks:
            parent_id = chunk["metadata"]["parent_id"]
            if parent_id not in parent_groups:
                parent_groups[parent_id] = []
            parent_groups[parent_id].append(chunk)

        # 3. Join chunks from same parent
        joined_chunks = []
        for parent_id, chunks in parent_groups.items():
            # Sort by chunk index
            chunks.sort(key=lambda x: x["metadata"]["chunk_index"])

            # Join consecutive chunks (2 at a time)
            for i in range(0, len(chunks), 2):
                if i + 1 < len(chunks):
                    joined = chunks[i]["text"] + "\n" + chunks[i+1]["text"]
                else:
                    joined = chunks[i]["text"]

                joined_chunks.append(joined)

        # 4. Re-score joined chunks
        scored = self.rescore_chunks(question, joined_chunks)

        # 5. Use top chunks
        top_chunks = scored[:3]
        context = "\n\n".join(top_chunks)

        # 6. Generate
        return self.generator.generate(question, context)

    def rescore_chunks(self, question: str, chunks: List[str]) -> List[str]:
        """Re-score joined chunks."""
        # Re-embed joined chunks
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        query_embedding = model.encode([question])[0]
        chunk_embeddings = model.encode(chunks)

        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

        # Sort by similarity
        indexed_chunks = list(zip(chunks, similarities))
        indexed_chunks.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in indexed_chunks]

# ===== PARENT-DOCUMENT RAG =====

class ParentDocumentRAG:
    """Parent-document RAG pattern"""

    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        self.document_store = {}

    def query(self, question: str, k: int = 3) -> str:
        """Query usando parent documents."""
        # 1. Retrieve chunks
        chunks = self.retriever.search(question, k=20)

        # 2. Get unique parent IDs
        parent_ids = list(set(
            chunk["metadata"]["parent_id"]
            for chunk in chunks
        ))

        # 3. Score parent documents
        parent_scores = {}
        for parent_id in parent_ids:
            # Count relevant chunks
            score = sum(
                1 for chunk in chunks
                if chunk["metadata"]["parent_id"] == parent_id
            )
            parent_scores[parent_id] = score

        # 4. Get top parent documents
        top_parent_ids = sorted(
            parent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # 5. Retrieve full documents
        context_parts = []
        for parent_id, score in top_parent_ids:
            if parent_id in self.document_store:
                doc = self.document_store[parent_id]
                context_parts.append(doc["text"])

        context = "\n\n".join(context_parts)

        # 6. Generate
        return self.generator.generate(question, context)

# ===== ROUTING RAG =====

class Router:
    """Router para different retrievers based on query type"""

    def __init__(self, retrievers: Dict[str, Retriever]):
        self.retrievers = retrievers

    def route(self, question: str) -> str:
        """Route question to appropriate retriever."""
        # Simple rule-based routing
        if any(word in question.lower() for word in ["code", "function", "class"]):
            return "code"
        elif any(word in question.lower() for word in ["when", "where", "who"]):
            return "factual"
        elif any(word in question.lower() for word in ["how", "why"]):
            return "explanatory"
        else:
            return "general"

class RoutingRAG:
    """RAG com routing para different retrievers"""

    def __init__(self, retrievers: Dict[str, Retriever], generator: Generator):
        self.retrievers = retrievers
        self.generator = generator
        self.router = Router(retrievers)

    def query(self, question: str) -> str:
        """Query com routing."""
        # 1. Route to appropriate retriever
        route = self.router.route(question)
        retriever = self.retrievers[route]

        # 2. Retrieve with specific retriever
        results = retriever.search(question, k=5)

        # 3. Generate
        context = "\n".join([r["text"] for r in results])

        return self.generator.generate(question, context)

# ===== MODULAR RAG PIPELINE =====

@dataclass
class RAGConfig:
    retriever_type: str
    generator_type: str
    use_reranker: bool
    chunk_size: int
    chunk_overlap: int

class ModularRAG:
    """Modular RAG com configurable pipeline"""

    def __init__(self, config: RAGConfig):
        self.config = config

        # Create components
        if config.retriever_type == "naive":
            self.retriever = NaiveRetriever(vectorstore, embeddings)
        elif config.retriever_type == "chunk_join":
            self.retriever = ChunkJoinRetriever(NaiveRetriever(vectorstore, embeddings), generator)
        # Add more retrievers...

        if config.generator_type == "llm":
            self.generator = LLMGenerator(llm)

        # Optional reranker
        if config.use_reranker:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

    def query(self, question: str) -> str:
        """Execute modular pipeline."""
        # 1. Retrieve
        results = self.retriever.search(question, k=10)

        # 2. Optionally rerank
        if self.reranker:
            results = self.reranker.rerank(question, results, k=5)

        # 3. Generate
        context = "\n".join([r["text"] for r in results])

        return self.generator.generate(question, context)

# Usage examples
if __name__ == "__main__":
    # Modular RAG
    print("="*60)
    print("MODULAR RAG EXAMPLE")
    print("="*60)

    # Configure
    config = RAGConfig(
        retriever_type="naive",
        generator_type="llm",
        use_reranker=True,
        chunk_size=1000,
        chunk_overlap=200
    )

    # Create modular RAG
    rag = ModularRAG(config)
    # answer = rag.query("What is the main topic?")
    # print(f"Answer: {answer}")
```

---

## EXAMPLE 4: Frameworks Comparison (Seção 10)

### Framework Benchmark

```python
"""
Frameworks Comparison: LangChain, LlamaIndex, Haystack
Compare different RAG frameworks
"""

import time
from typing import List, Dict
import numpy as np

# ===== LANGCHAIN IMPLEMENTATION =====

def langchain_rag(query: str) -> Dict:
    """LangChain RAG implementation."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Setup
    start = time.time()

    # Load document
    loader = TextLoader("document.txt")
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Embed
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(splits)

    # Retrieve
    results = vectorstore.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in results])

    # Generate
    prompt = ChatPromptTemplate.from_template("""
    Question: {question}
    Context: {context}
    Answer:
    """)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm
    answer = chain.invoke({"question": query, "context": context})

    elapsed = time.time() - start

    return {
        "framework": "LangChain",
        "answer": answer.content,
        "time": elapsed,
        "retrieved": len(results)
    }

# ===== LLAMAINDEX IMPLEMENTATION =====

def llamaindex_rag(query: str) -> Dict:
    """LlamaIndex RAG implementation."""
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.extractors import BaseExtractor
    from llama_index.text_splitter import SentenceSplitter
    from llama_index.embeddings import OpenAIEmbedding
    from llama_index.core.response import ResponseMode

    start = time.time()

    # Load document
    doc = Document(text="Your document text here")

    # Create index
    index = VectorStoreIndex.from_documents([doc])

    # Create query engine
    query_engine = index.as_query_engine()

    # Query
    response = query_engine.query(query)

    elapsed = time.time() - start

    return {
        "framework": "LlamaIndex",
        "answer": str(response),
        "time": elapsed,
        "retrieved": 5  # LlamaIndex doesn't expose easily
    }

# ===== HAYSTACK IMPLEMENTATION =====

def haystack_rag(query: str) -> Dict:
    """Haystack RAG implementation."""
    from haystack import Pipeline
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.generators import OpenAIGenerator

    start = time.time()

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component("doc_embedder", SentenceTransformersDocumentEmbedder())
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever())
    pipeline.add_component("generator", OpenAIGenerator())

    # Connect
    pipeline.connect("doc_embedder", "retriever")
    pipeline.connect("text_embedder", "retriever")
    pipeline.connect("retriever", "generator")

    # Run
    result = pipeline.run({
        "doc_embedder": {"documents": ["Your document text here"]},
        "text_embedder": {"text": query}
    })

    elapsed = time.time() - start

    return {
        "framework": "Haystack",
        "answer": result["generator"]["answers"][0].answer,
        "time": elapsed,
        "retrieved": 5
    }

# ===== FRAMEWORK COMPARISON =====

def compare_frameworks(queries: List[str]) -> Dict:
    """Compare multiple frameworks on same queries."""
    results = {
        "LangChain": [],
        "LlamaIndex": [],
        "Haystack": []
    }

    print("="*70)
    print("FRAMEWORK COMPARISON")
    print("="*70)

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}/{len(queries)}: {query}")
        print("-" * 70)

        try:
            # LangChain
            start = time.time()
            lc_result = langchain_rag(query)
            results["LangChain"].append(lc_result)
            print(f"  LangChain: {lc_result['time']:.2f}s")
        except Exception as e:
            print(f"  LangChain failed: {e}")
            results["LangChain"].append({"error": str(e)})

        try:
            # LlamaIndex
            li_result = llamaindex_rag(query)
            results["LlamaIndex"].append(li_result)
            print(f"  LlamaIndex: {li_result['time']:.2f}s")
        except Exception as e:
            print(f"  LlamaIndex failed: {e}")
            results["LlamaIndex"].append({"error": str(e)})

        try:
            # Haystack
            hs_result = haystack_rag(query)
            results["Haystack"].append(hs_result)
            print(f"  Haystack: {hs_result['time']:.2f}s")
        except Exception as e:
            print(f"  Haystack failed: {e}")
            results["Haystack"].append({"error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for framework, runs in results.items():
        times = [
            r["time"] for r in runs
            if "time" in r
        ]
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\n{framework}:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")
            print(f"  Runs: {len(times)}/{len(queries)}")

    return results

# ===== SELECTION GUIDE =====

def select_framework(requirements: Dict) -> str:
    """Select framework based on requirements."""
    print("="*70)
    print("FRAMEWORK SELECTION GUIDE")
    print("="*70)

    print("\nRequirements:")
    for key, value in requirements.items():
        print(f"  {key}: {value}")

    # Decision logic
    if requirements.get("use_case") == "prototyping":
        recommendation = "LangChain"  # Easy to start
    elif requirements.get("data_heavy"):
        recommendation = "LlamaIndex"  # Index-centric
    elif requirements.get("production_api"):
        recommendation = "Haystack"  # REST API built-in
    elif requirements.get("multimodal"):
        recommendation = "LangChain"  # More integrations
    else:
        recommendation = "LangChain"  # Default

    print(f"\n🎯 Recommendation: {recommendation}")
    print("\nReasoning:")

    if recommendation == "LangChain":
        print("  - Large ecosystem and community")
        print("  - Comprehensive documentation")
        print("  - Multiple integrations")
        print("  - Good for general use")
    elif recommendation == "LlamaIndex":
        print("  - Index-centric design")
        print("  - Good for data-heavy applications")
        print("  - Multiple index types")
        print("  - Query optimization focused")
    else:  # Haystack
        print("  - Production-ready with REST API")
        print("  - NLP-focused features")
        print("  - Scalable architecture")
        print("  - Good for production deployments")

    return recommendation

# Usage example
if __name__ == "__main__":
    # Sample queries
    test_queries = [
        "What is the main topic?",
        "How does it work?",
        "When was this created?"
    ]

    # Run comparison
    results = compare_frameworks(test_queries)

    # Get selection
    requirements = {
        "use_case": "general",
        "team_size": "small",
        "timeline": "fast",
        "production_ready": False,
        "data_heavy": False,
        "multimodal": False
    }
    framework = select_framework(requirements)
```

---

## EXAMPLE 5: Production Deployment (Seção 11)

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app
  labels:
    app: rag-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-app
  template:
    metadata:
      labels:
        app: rag-app
    spec:
      containers:
      - name: rag-app
        image: myregistry/rag-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: pinecone-api-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: rag-config
      - name: data
        persistentVolumeClaim:
          claimName: rag-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: rag-app-service
spec:
  selector:
    app: rag-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: rag-app-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-app-service
            port:
              number: 80
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
data:
  app.config: |
    DEBUG=false
    LOG_LEVEL=info
    CORS_ORIGINS=["https://example.com"]
    MAX_CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    RETRIEVAL_K=5
    ENABLE_CACHE=true
    CACHE_TTL=3600
---
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
type: Opaque
stringData:
  openai-api-key: "your-openai-key"
  pinecone-api-key: "your-pinecone-key"
  database-url: "postgresql://user:pass@postgres:5432/rag"
```

### Monitoring Setup

```python
# monitoring-setup.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'rag_request_duration_seconds',
    'RAG request latency'
)

ACTIVE_QUERIES = Gauge(
    'rag_active_queries',
    'Number of active queries'
)

VECTOR_DB_SIZE = Gauge(
    'rag_vector_db_size',
    'Number of vectors in database'
)

CACHE_HIT_RATE = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate percentage'
)

def setup_monitoring(port: int = 8001):
    """Start monitoring server."""
    start_http_server(port)
    print(f"Monitoring server started on port {port}")
    print("Metrics available at /metrics")

# Prometheus configuration
PROMETHEUS_CONFIG = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-app'
    static_configs:
      - targets: ['rag-app:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'vector-db'
    static_configs:
      - targets: ['chroma:8000']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""

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

# Usage in application
from fastapi import FastAPI

app = FastAPI()

@app.get("/query")
def query(question: str):
    start = time.time()
    ACTIVE_QUERIES.inc()

    try:
        # Process query
        result = rag_query(question)

        REQUEST_COUNT.labels(
            method='GET',
            endpoint='/query',
            status='success'
        ).inc()

        REQUEST_LATENCY.observe(time.time() - start)

        return result

    except Exception as e:
        REQUEST_COUNT.labels(
            method='GET',
            endpoint='/query',
            status='error'
        ).inc()
        raise

    finally:
        ACTIVE_QUERIES.dec()

# Dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## EXAMPLE 6: Troubleshooting (Seção 12)

### Troubleshooting Tools

```python
"""
Troubleshooting: Debugging, Profiling, Health Checks
Ferramentas para debug e troubleshooting
"""

import time
import logging
import structlog
import cProfile
import pstats
import psutil
import traceback
from functools import wraps
from typing import Dict, Any
import json

# ===== STRUCTURED LOGGING =====

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

# ===== PROFILING =====

def profile_function(func):
    """Decorator para profiling de funções."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # Print profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        print(f"\nProfile for {func.__name__}:")
        print(s.getvalue())

        return result
    return wrapper

# ===== MEMORY PROFILING =====

def memory_profile(func):
    """Decorator para memory profiling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc
        tracemalloc.start()

        result = func(*args, **kwargs)

        # Get memory statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nMemory profile for {func.__name__}:")
        print(f"  Current: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")

        return result
    return wrapper

# ===== HEALTH CHECKS =====

class HealthChecker:
    """Health checker para RAG system."""

    def __init__(self, db_connection, vector_db_connection, llm_connection):
        self.db = db_connection
        self.vector_db = vector_db_connection
        self.llm = llm_connection

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        return {
            "database": self.check_database(),
            "vector_db": self.check_vector_db(),
            "llm": self.check_llm(),
            "system": self.check_system()
        }

    def check_database(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            start = time.time()
            result = self.db.execute("SELECT 1").fetchone()
            latency = time.time() - start

            if result:
                return {
                    "status": "healthy",
                    "latency_ms": latency * 1000
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "No result from database"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_vector_db(self) -> Dict[str, Any]:
        """Check vector database health."""
        try:
            start = time.time()
            # Simple query to test connection
            count = self.vector_db.count()
            latency = time.time() - start

            return {
                "status": "healthy",
                "latency_ms": latency * 1000,
                "vector_count": count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_llm(self) -> Dict[str, Any]:
        """Check LLM health."""
        try:
            start = time.time()
            # Simple test query
            response = self.llm.complete("Hi")
            latency = time.time() - start

            return {
                "status": "healthy",
                "latency_ms": latency * 1000
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_system(self) -> Dict[str, Any]:
        """Check system resources."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }

# ===== ERROR TRACKING =====

class ErrorTracker:
    """Track errors with context."""

    def __init__(self, log_file: str = "errors.log"):
        self.log_file = log_file
        self.error_count = 0
        self.errors = []

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error com context."""
        self.error_count += 1

        error_info = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }

        self.errors.append(error_info)

        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(error_info) + '\n')

        print(f"❌ Error #{self.error_count}: {error}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        if not self.errors:
            return {"total_errors": 0}

        error_types = {}
        for error in self.errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": self.error_count,
            "error_types": error_types,
            "recent_errors": self.errors[-5:]  # Last 5 errors
        }

# ===== QUERY ANALYZER =====

class QueryAnalyzer:
    """Analyze query performance e quality."""

    def __init__(self):
        self.query_times = []
        self.query_results = []

    def analyze_query(self, question: str, result: str, start_time: float):
        """Analyze individual query."""
        elapsed = time.time() - start_time
        self.query_times.append(elapsed)

        # Check response length
        response_length = len(result)

        # Check response quality
        quality_score = self.assess_quality(question, result)

        analysis = {
            "question": question[:100],  # Truncate for privacy
            "response_length": response_length,
            "latency_ms": elapsed * 1000,
            "quality_score": quality_score,
            "timestamp": time.time()
        }

        self.query_results.append(analysis)
        return analysis

    def assess_quality(self, question: str, result: str) -> float:
        """Simple quality assessment."""
        # Check if result seems reasonable
        if len(result) < 50:
            return 0.3  # Too short

        if len(result) > 5000:
            return 0.7  # Very long but ok

        if "don't know" in result.lower() or "cannot" in result.lower():
            return 0.4  # Uncertainty

        return 0.8  # Good response

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.query_times:
            return {"message": "No queries yet"}

        times = self.query_times
        return {
            "total_queries": len(times),
            "avg_latency_ms": (sum(times) / len(times)) * 1000,
            "p50_ms": np.percentile(times, 50) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
            "p99_ms": np.percentile(times, 99) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000
        }

# Usage example
if __name__ == "__main__":
    # Setup
    error_tracker = ErrorTracker()
    analyzer = QueryAnalyzer()
    checker = HealthChecker(None, None, None)

    @profile_function
    @memory_profile
    def example_rag_query(question: str) -> str:
        """Example RAG query com monitoring."""
        try:
            logger.info(
                "Query received",
                question=question[:50],
                user_id="user123"
            )

            start = time.time()

            # Simulate RAG query
            time.sleep(0.1)  # Retrieval
            result = "This is a simulated response"
            time.sleep(0.2)  # Generation

            elapsed = time.time() - start

            # Analyze
            analysis = analyzer.analyze_query(question, result, start)

            logger.info(
                "Query completed",
                latency_ms=elapsed * 1000,
                response_length=len(result)
            )

            return result

        except Exception as e:
            error_tracker.log_error(e, {"question": question})
            raise

    # Test
    result = example_rag_query("What is AI?")
    print(f"\nResult: {result}")

    print(f"\nPerformance summary: {analyzer.get_performance_summary()}")
    print(f"\nError summary: {error_tracker.get_error_summary()}")
    print(f"\nHealth check: {checker.check_all()}")
```

---

## USAGE INSTRUCTIONS

### Prerequisites Installation

```bash
# Performance (Example 1)
pip install faiss-cpu torch sentence-transformers redis

# Advanced Patterns (Example 2)
pip install torch torchvision
pip install clip
pip install neo4j
pip install openai

# Frameworks (Example 3)
pip install langchain langchain-community langchain-openai
pip install llama-index
pip install haystack-ai

# Production (Example 4)
pip install fastapi uvicorn
pip install prometheus-client
pip install structlog psutil
kubectl (for deployment)

# Troubleshooting (Example 5)
pip install structlog
pip install prometheus-client
pip install psutil
```

### Running Examples

```bash
# Example 1: Performance
python example1_performance.py

# Example 2: Advanced Patterns
python example2_advanced_patterns.py

# Example 3: Frameworks
python example3_frameworks.py

# Example 4: Production
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Example 5: Troubleshooting
python example5_troubleshooting.py
```

### Windows-Specific Notes

1. **Docker**: Use Docker Desktop for Windows
2. **WSL2**: Run Linux tools in WSL2
3. **PowerShell**: Use for deployment scripts
4. **Path handling**: Use raw strings `r"C:\path"`

### Next Steps

1. Experiment com optimization techniques
2. Try different patterns
3. Test frameworks na sua use case
4. Deploy to production
5. Set up monitoring
6. Practice troubleshooting

---

**Status**: ✅ Code examples Fase 4 created
**Próximo**: Resumo Executivo Fase 4
**Data Conclusão**: 09/11/2025

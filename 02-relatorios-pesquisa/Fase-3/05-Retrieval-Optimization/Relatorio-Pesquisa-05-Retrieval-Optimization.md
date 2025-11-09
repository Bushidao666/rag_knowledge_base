# Relatório de Pesquisa: Seção 05 - Retrieval Optimization

### Data: 09/11/2025
### Status: Fase 3 - Optimization

---

## 1. RESUMO EXECUTIVO

Retrieval Optimization é o processo de maximizar a relevância e qualidade dos documentos recuperados em sistemas RAG. A combinação adequada de técnicas de retrieval determina diretamente a qualidade das respostas geradas.

**Insights Chave:**
- **Dense Retrieval**: Semantic similarity, embeddings, contextual understanding
- **Sparse Retrieval**: BM25, keyword matching, exact terms
- **Hybrid Search**: Combina ambos, typically melhor performance
- **Reranking**: Cross-encoders, ColBERT, RankGPT para precision
- **Query Expansion**: Synonym, semantic expansion para coverage

---

## 2. FONTES PRIMÁRIAS

### 2.1 Documentações Oficiais
- **LangChain RAG**: https://docs.langchain.com/oss/python/langchain/rag
- **LangChain Overview**: https://docs.langchain.com/oss/python/langchain/overview
- **LlamaIndex**: https://developers.llamaindex.ai/python/framework/
- **RAGAS**: https://docs.ragas.io/
- **TruLens**: https://docs.trulens.org/

---

## 3. DENSE RETRIEVAL

### 3.1 Como Funciona

**Dense Retrieval** usa embeddings para representar documentos e queries em vetores densos, calculando similaridade semântica.

**Arquitetura:**
```
[Query] → [Embeddings] → [Vector Search] → [Top-K Similar Docs]
```

**Exemplo LangChain:**
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Create vector store
vectorstore = InMemoryVectorStore(embeddings)

# Dense retrieval
results = vectorstore.similarity_search(
    query="What is machine learning?",
    k=5
)
```

### 3.2 Vantagens

- ✅ **Semantic understanding**: Captura meaning, não só keywords
- ✅ **Cross-lingual**: Modelos multilíngues
- ✅ **Contextual**: Entende contexto e synonyms
- ✅ **Robust**: Funciona com paraphrase, synonyms
- ✅ **State-of-the-art**: Melhor quality em benchmarks

### 3.3 Desvantagens

- ❌ **Computational cost**: Requer embedding generation
- ❌ **Vector database needed**: Infra adicional
- ❌ **Training dependency**: Quality depende do embedding model
- ❌ **Opaque**: Hard to debug why retrieved

### 3.4 Parâmetros Importantes

| Parâmetro | Descrição | Typical Value |
|-----------|-----------|---------------|
| **k** | Number of documents to retrieve | 2-10 |
| **chunk_size** | Document chunk size | 1000 chars |
| **chunk_overlap** | Overlap between chunks | 200 chars |
| **distance_metric** | Cosine, L2, dot product | Cosine |
| **score_threshold** | Minimum similarity | None (use k) |

### 3.5 When to Use

**Use Dense Retrieval when:**
- ✅ Documents have semantic variation
- ✅ Queries are natural language
- ✅ You have computational budget
- ✅ Quality is more important than speed
- ✅ Multi-lingual support needed

**Don't use when:**
- ❌ Exact keyword matching required
- ❌ Very strict performance requirements
- ❌ Small, well-curated datasets
- ❌ Budget extremely limited

---

## 4. SPARSE RETRIEVAL (BM25)

### 4.1 BM25 Overview

**BM25 (Best Match 25)** é um ranking function usado em information retrieval, baseado no probabilistic relevance framework.

**Fórmula:**
```
BM25 = sum_over_terms IDF(term) * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * document_length))
```

**Onde:**
- `IDF`: Inverse Document Frequency
- `k1`: Saturation parameter (typically 1.2-2.0)
- `b`: Length normalization (typically 0.75)

### 4.2 Como Funciona

**Sparse Retrieval** representa documentos como sparse vectors (bag-of-words), focando em exact term matching.

**Processo:**
1. Tokenize documents e queries
2. Calculate term frequencies (TF)
3. Calculate inverse document frequencies (IDF)
4. Compute BM25 score
5. Rank documents by score

### 4.3 Vantagens

- ✅ **Fast**: No embeddings needed
- ✅ **Simple**: Easy to understand e debug
- ✅ **Deterministic**: Same input = same output
- ✅ **Exact matching**: Good for precise terms
- ✅ **No ML dependency**: Doesn't depend on models
- ✅ **Transparent**: Can see why document retrieved

### 4.4 Desvantagens

- ❌ **Keyword dependency**: Exact term matching only
- ❌ **No semantic understanding**: Can't handle synonyms
- ❌ **OOV issues**: Out-of-vocabulary problems
- ❌ **Language dependent**: Typically language-specific
- ❌ **Context insensitive**: Ignores semantic context

### 4.5 Implementation (LlamaIndex)

```python
from llama_index.core.extractors import BaseExtractor
from llama_index.core.extractors.dictionary import DictionaryExtractor
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import BM25Retriever

# Create documents
docs = [Document(text="Your document text here")]

# Create BM25 retriever
retriever = BM25Retriever.from_documents(
    docs,
    similarity_top_k=5,
    tokenizer=None  # Use default
)

# Query
results = retriever.retrieve("What is machine learning?")
```

### 4.6 Parameters

| Parâmetro | Descrição | Typical Value |
|-----------|-----------|---------------|
| **similarity_top_k** | Number of results | 5-10 |
| **k1** | Saturation parameter | 1.5 |
| **b** | Length normalization | 0.75 |
| **lowercase** | Convert to lowercase | True |
| **stopwords** | Remove stopwords | True |

### 4.7 When to Use

**Use BM25 when:**
- ✅ Exact keyword matching needed
- ✅ Speed is critical
- ✅ Small computational budget
- ✅ Deterministic results required
- ✅ Technical documents (specific terms)

**Don't use when:**
- ❌ Natural language queries
- ❌ Need semantic understanding
- ❌ Multi-lingual support
- ❌ Synonyms common in domain

---

## 5. HYBRID SEARCH

### 5.1 Como Funciona

**Hybrid Search** combina dense retrieval (semantic) e sparse retrieval (keyword-based) para obter melhor coverage e precision.

**Arquitetura:**
```
Query → [Dense Branch] → Score₁
        [Sparse Branch] → Score₂
        ↓
    [Score Fusion] → Combined Score
        ↓
    [Top-K Results]
```

### 5.2 Fusion Strategies

#### 5.2.1 Score Normalization

**Min-Max Normalization:**
```
normalized_score = (score - min_score) / (max_score - min_score)
```

**Z-Score Normalization:**
```
normalized_score = (score - mean) / std_dev
```

#### 5.2.2 Weighted Fusion

**Linear Combination:**
```
final_score = α * dense_score + (1-α) * sparse_score
```

**Typical α values:**
- α = 0.7: Dense-biased hybrid
- α = 0.5: Balanced hybrid
- α = 0.3: Sparse-biased hybrid

#### 5.2.3 Reciprocal Rank Fusion (RRF)

```
final_score = Σ (dense_score / (rank₁ + β)) + Σ (sparse_score / (rank₂ + β))
```

Where β is a constant (typically 60).

### 5.3 Implementation

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.types import RESPONSE_TYPE
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.extractors import BaseExtractor

class HybridRetriever(BaseRetriever):
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha  # Weight for dense (0-1)

    def _retrieve(self, query, **kwargs) -> RESPONSE_TYPE:
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query)

        # Normalize scores
        max_dense = max([r.score for r in dense_results]) if dense_results else 0
        max_sparse = max([r.score for r in sparse_results]) if sparse_results else 0

        # Combine
        combined = {}
        for r in dense_results:
            norm_score = r.score / max_dense if max_dense > 0 else 0
            combined[r.node_id] = self.alpha * norm_score

        for r in sparse_results:
            norm_score = r.score / max_sparse if max_sparse > 0 else 0
            combined[r.node_id] = combined.get(r.node_id, 0) + (1 - self.alpha) * norm_score

        # Sort and get top-k
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        k = kwargs.get("similarity_top_k", 10)

        return [dense_results[0] for node_id, score in sorted_results[:k]]
```

### 5.4 When to Use

**Use Hybrid Search when:**
- ✅ Want best of both worlds
- ✅ Queries vary (some exact, some semantic)
- ✅ Dataset has mixed content
- ✅ Can afford complexity
- ✅ Need robust performance

**Don't use when:**
- ❌ Very constrained computational budget
- ❌ Clear preference for one method
- ❌ Simple use case (use what works best)

---

## 6. RERANKING

### 6.1 Cross-Encoders

**Cross-encoders** re-rank o top-k results de retrievers initial usando um modelo specially trained para relevance ranking.

**Arquitetura:**
```
[Query + Doc] → [Cross-Encoder] → [Relevance Score]
```

**Como Funciona:**
1. Retrieve top-N (e.g., 50) documents using dense/sparse
2. For each query-document pair, compute cross-encoder score
3. Re-rank by cross-encoder scores
4. Return top-k (e.g., 10) final results

**Vantagens:**
- ✅ Better precision: More accurate relevance scoring
- ✅ Query-specific: Considers query-document interaction
- ✅ State-of-the-art: Best quality for ranking

**Desvantagens:**
- ❌ Slower: Requires N model inferences
- ❌ Expensive: More compute cost
- ❌ Limited k: Typically for small k only

### 6.2 ColBERT

**ColBERT (Contextualized Late Interaction)** é um reranking model que achieves хороший balance entre quality e speed.

**Características:**
- Late interaction: Computes token-level similarities
- Efficient: Faster than cross-encoders
- Scalable: Can handle large k values
- Good quality: Near cross-encoder performance

**Exemplo Implementation:**
```python
from colbert import Searcher

# Initialize
searcher = Searcher(
    index="path/to/index",
    root="path/to/index"
)

# Search
results = searcher.search(
    query="machine learning",
    k=100  # Retrieve 100
)

# Rerank
reranked = searcher.rerank(
    query="machine learning",
    k=10   # Return top 10
)
```

### 6.3 RankGPT

**RankGPT** uses LLMs (GPT-4, etc.) para re-ranking.

**Processo:**
1. Get top-N results from retriever
2. Create prompt with query + docs
3. Ask LLM to rank docs
4. Return ranked results

**Vantagens:**
- ✅ High quality: LLM understanding
- ✅ Flexible: Can use any LLM
- ✅ Contextual: Full document context

**Desvantagens:**
- ❌ Very slow: LLM inference
- ❌ Expensive: API costs
- ❌ Token limits: Cannot handle very large N

### 6.4 Reranking Pipeline

```python
class RerankRetriever:
    def __init__(self, retriever, reranker, k_initial=50, k_final=10):
        self.retriever = retriever
        self.reranker = reranker
        self.k_initial = k_initial
        self.k_final = k_final

    def retrieve(self, query, **kwargs):
        # Step 1: Initial retrieval
        initial_results = self.retriever.retrieve(
            query,
            similarity_top_k=self.k_initial
        )

        # Step 2: Rerank
        reranked = self.reranker.rerank(
            query,
            initial_results,
            k=self.k_final
        )

        return reranked
```

### 6.5 When to Use Reranking

**Use Reranking when:**
- ✅ Need maximum precision
- ✅ Can afford computational cost
- ✅ Quality > speed
- ✅ Initial retrieval is good but can be better
- ✅ Enterprise/production use

**Don't use when:**
- ❌ Speed is critical
- ❌ Budget extremely limited
- ❌ Initial retrieval is poor (fix retrieval first)
- ❌ Real-time applications

---

## 7. QUERY EXPANSION

### 7.1 Query Rewriting

**Query Rewriting** expands or reformulates the query to improve retrieval.

**Techniques:**

1. **Synonym Expansion**
   ```python
   # Add synonyms
   query = "car"
   expanded = "car OR automobile OR vehicle OR auto"
   ```

2. **Semantic Expansion**
   ```python
   # Use LLM to expand
   prompt = f"Generate 3 alternative queries for: {query}"
   expansions = llm.generate(prompt)
   ```

3. **Multi-Query Fusion**
   ```python
   # Multiple query versions
   queries = [
       "What is AI?",
       "Define artificial intelligence",
       "Explain AI technology"
   ]
   results = [retriever.retrieve(q) for q in queries]
   fused = fuse_results(results)
   ```

### 7.2 Pseudo-Relevance Feedback (PRF)

**PRF** assumes top results are relevant e uses them to expand query.

**Process:**
1. Get initial retrieval results
2. Extract terms from top documents
3. Add important terms to query
4. Re-run retrieval

### 7.3 Query Expansion with LLMs

```python
def expand_query(query, llm):
    prompt = f"""
    Expand this query to improve retrieval.
    Add relevant synonyms, related terms, and variations.

    Original: {query}

    Expanded:"""

    expanded = llm.generate(prompt)
    return expanded
```

### 7.4 When to Use

**Use Query Expansion when:**
- ✅ Query is too short
- ✅ Low recall in initial results
- ✅ Domain has synonyms/variations
- ✅ User queries are poor quality

**Don't use when:**
- ❌ Query is already good
- ❌ High precision required (may reduce)
- ❌ Results are already good

---

## 8. PERFORMANCE OPTIMIZATION

### 8.1 Caching

**Query Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query_hash):
    return retriever.retrieve(query)
```

**Result Caching:**
```python
import redis

def cache_results(query, results, ttl=3600):
    r = redis.Redis()
    r.setex(f"query:{hash(query)}", ttl, json.dumps(results))

def get_cached_results(query):
    r = redis.Redis()
    cached = r.get(f"query:{hash(query)}")
    if cached:
        return json.loads(cached)
    return None
```

### 8.2 Approximate Nearest Neighbor

**FAISS Indexing:**
```python
import faiss
import numpy as np

# Create index
d = 384  # dimension
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)

# Train and add
index.train(embeddings)
index.add(embeddings)

# Search
D, I = index.search(query_embeddings, k=10)
```

**HNSW (Hierarchical Navigable Small World):**
- Fast approximate search
- Good recall at high speed
- Used by many vector databases

### 8.3 Parallel Retrieval

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_retrieve(queries, retriever, max_workers=4):
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, retriever.retrieve, query)
            for query in queries
        ]
        results = await asyncio.gather(*tasks)

    return results
```

### 8.4 Pre-filtering

**Metadata Filtering:**
```python
# Filter by metadata before similarity search
filtered_docs = [
    doc for doc in all_docs
    if doc.metadata["category"] == "research"
]

# Then do similarity search on filtered set
```

### 8.5 Optimization Checklist

- [ ] Use appropriate index type (IVF, HNSW, Flat)
- [ ] Normalize embeddings for cosine similarity
- [ ] Implement caching for frequent queries
- [ ] Use batch processing for multiple queries
- [ ] Parallelize independent operations
- [ ] Pre-filter using metadata when possible
- [ ] Monitor and profile performance
- [ ] Use approximate NN for large scale

---

## 9. COMPARISON MATRIX

| Method | Quality | Speed | Cost | Semantic | Best For |
|--------|---------|-------|------|----------|----------|
| **Dense** | 🟢🟢🟢 | 🟡🟡 | 🟡 | ✅ | Semantic queries |
| **Sparse (BM25)** | 🟡🟡 | 🟢🟢🟢 | 🟢 | ❌ | Exact keywords |
| **Hybrid** | 🟢🟢🟢 | 🟡 | 🟡 | 🟡 | Robust retrieval |
| **Reranking** | 🟢🟢🟢 | 🔴 | 🔴 | ✅ | Maximum precision |

---

## 10. DECISION TREE

```
QUERY TYPE
├─ Natural language / semantic?
│   ├─ SIM → Dense retrieval
│   └─ NÃO → Exact keywords?
│       ├─ SIM → BM25
│       └─ NÃO → Hybrid (try both)
│
QUALITY REQUIREMENT
├─ Maximum precision needed?
│   ├─ SIM → Add reranking (cross-encoder/ColBERT)
│   └─ NÃO → Continue
│
PERFORMANCE CONSTRAINT
├─ Speed critical?
│   ├─ SIM → BM25 ou dense optimized (HNSW)
│   └─ NÃO → Dense ou hybrid
│
FINAL RECOMMENDATION
├─ Start: Dense (BGE embeddings)
├─ If issues: Try hybrid (dense + BM25)
├─ If need precision: Add reranking
└─ If need speed: BM25 ou optimize dense (HNSW)
```

---

## 11. IMPLEMENTATION EXAMPLES

### 11.1 Complete RAG with Optimization

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import numpy as np

class HybridRetriever(BaseRetriever):
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def _retrieve(self, query, **kwargs) -> List[Document]:
        # Get results from both
        dense_results = self.dense_retriever.get_relevant_documents(query)
        sparse_results = self.sparse_retriever.get_relevant_documents(query)

        # Normalize scores
        all_docs = {}

        # Process dense
        if dense_results:
            max_score = max([r.metadata.get("score", 0) for r in dense_results])
            for doc in dense_results:
                score = doc.metadata.get("score", 0) / max_score
                all_docs[doc.page_content] = {
                    "doc": doc,
                    "score": self.alpha * score
                }

        # Process sparse
        if sparse_results:
            max_score = max([r.metadata.get("score", 0) for r in sparse_results])
            for doc in sparse_results:
                if doc.page_content in all_docs:
                    all_docs[doc.page_content]["score"] += (1 - self.alpha) * (
                        doc.metadata.get("score", 0) / max_score
                    )
                else:
                    all_docs[doc.page_content] = {
                        "doc": doc,
                        "score": (1 - self.alpha) * (doc.metadata.get("score", 0) / max_score)
                    }

        # Sort and return top-k
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        k = kwargs.get("k", 10)

        return [item["doc"] for item in sorted_docs[:k]]

# Usage
dense_retriever = InMemoryVectorStore(embeddings).as_retriever(search_k=20)
sparse_retriever = TFIDFRetriever.from_texts(texts)

hybrid = HybridRetriever(dense_retriever, sparse_retriever, alpha=0.7)
results = hybrid.get_relevant_documents("What is AI?", k=5)
```

### 11.2 With Reranking

```python
from langchain_community.cross_encoders import CrossEncoder

class RerankRetriever(BaseRetriever):
    def __init__(self, base_retriever, reranker_model, k_initial=50, k_final=10):
        self.base_retriever = base_retriever
        self.reranker = CrossEncoder(reranker_model)
        self.k_initial = k_initial
        self.k_final = k_final

    def _retrieve(self, query, **kwargs) -> List[Document]:
        # Get initial results
        initial_results = self.base_retriever.get_relevant_documents(
            query, k=self.k_initial
        )

        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in initial_results]

        # Rerank
        scores = self.reranker.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(initial_results, scores):
            doc.metadata["rerank_score"] = score

        # Sort by rerank score
        reranked = sorted(initial_results, key=lambda x: x.metadata["rerank_score"], reverse=True)

        # Return top-k
        return reranked[:self.k_final]

# Usage
base_retriever = hybrid  # From above
reranker = RerankRetriever(
    base_retriever,
    "BAAI/bge-reranker-base",
    k_initial=50,
    k_final=10
)

results = reranker.get_relevant_documents("What is AI?", k=10)
```

---

## 12. EVALUATION

### 12.1 Metrics

**Retrieval Metrics:**
- **Recall@k**: % of relevant docs retrieved in top-k
- **Precision@k**: % of retrieved docs that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant doc
- **nDCG@k**: Normalized Discounted Cumulative Gain

**Reranking Metrics:**
- **MAP**: Mean Average Precision
- **nDCG**: For graded relevance

### 12.2 Evaluation Pipeline

```python
def evaluate_retriever(retriever, test_queries, relevant_docs):
    results = {}

    for query in test_queries:
        retrieved = retriever.get_relevant_documents(query, k=10)
        retrieved_ids = [doc.metadata.get("doc_id") for doc in retrieved]

        relevant = relevant_docs[query]

        # Calculate metrics
        recall = len(set(retrieved_ids) & set(relevant)) / len(relevant)
        precision = len(set(retrieved_ids) & set(relevant)) / len(retrieved_ids)

        results[query] = {
            "recall": recall,
            "precision": precision,
            "retrieved": retrieved_ids
        }

    return results
```

---

## 13. WINDOWS-SPECIFIC CONSIDERATIONS

### 13.1 Performance Tips

```python
import os

# Set number of threads for OpenMP (FAISS, etc.)
os.environ["OMP_NUM_THREADS"] = "4"

# Use all CPU cores for batch processing
from concurrent.futures import ThreadPoolExecutor
max_workers = min(32, os.cpu_count())
```

### 13.2 PowerShell Optimization Script

```powershell
# Enable optimizations
$env:PYTORCH_HIP_ALLOCATOR] = "pool"
$env:TOKENIZERS_PARALLELISM] = "true"
$env:OMP_NUM_THREADS] = "4"
```

---

## 14. BEST PRACTICES

### 14.1 Start Simple
1. Start with dense retrieval
2. Evaluate quality
3. Add complexity if needed (hybrid, reranking)
4. Always evaluate after each change

### 14.2 Know Your Use Case
- **FAQ/KB**: BM25 often sufficient
- **Research/Academic**: Dense or hybrid
- **Enterprise Search**: Hybrid with reranking
- **Real-time**: BM25 or optimized dense

### 14.3 Monitor Performance
- Track latency per query
- Monitor recall/precision
- A/B test different approaches
- Set up alerts for quality degradation

### 14.4 Optimize Iteratively
1. Measure baseline
2. Try one optimization at a time
3. Measure improvement
4. Keep or revert based on results
5. Repeat

---

## 15. COMMON PITFALLS

### 15.1 Not Evaluating
❌ **Problem**: Not measuring retrieval quality
✅ **Solution**: Use metrics, A/B test, track over time

### 15.2 Over-Engineering
❌ **Problem**: Using reranking when simple retrieval works
✅ **Solution**: Start simple, add complexity only if needed

### 15.3 Ignoring Speed
❌ **Problem**: Only optimizing for quality
✅ **Solution**: Balance quality vs speed, monitor latency

### 15.4 No Caching
❌ **Problem**: Re-computing same queries
✅ **Solution**: Cache frequent queries and results

### 15.5 Wrong k Value
❌ **Problem**: Using arbitrary k
✅ **Solution**: Tune k based on your use case and evaluation

---

## 16. RESEARCH GAPS

### 16.1 To Research
- [ ] Cross-encoder model comparison
- [ ] ColBERT vs cross-encoders benchmark
- [ ] RankGPT performance analysis
- [ ] Query expansion effectiveness study
- [ ] Fusion strategy comparison
- [ ] Approximate NN impact on quality

### 16.2 Advanced Topics
- [ ] Multi-vector retrieval
- [ ] Graph-based retrieval
- [ ] Learning to rank
- [ ] Query planning and routing
- [ ] Personalization in retrieval
- [ ] Real-time learning and adaptation

---

## 17. RECOMMENDATIONS

### 17.1 For Beginners
**Start here**: Dense retrieval with BGE embeddings
- Simple to implement
- Good quality
- Well-documented

### 17.2 For Production
**Recommended**: Hybrid retrieval with reranking
- Dense: BGE-large-en-v1.5
- Sparse: BM25
- Reranking: BGE-reranker or ColBERT
- Fusion: Weighted (α=0.7)

### 17.3 For Speed
**Recommended**: Optimized BM25
- Fast inference
- Good for exact matching
- Low computational cost

### 17.4 For Quality
**Recommended**: Dense + Reranking
- Dense for recall
- Cross-encoder for precision
- Best quality for critical applications

---

**Status**: ✅ Base para Retrieval Optimization coletada
**Próximo**: Seção 06 - Evaluation & Benchmarks
**Data Conclusão**: 09/11/2025

# Quick Start: Retrieval Optimization

**Tempo estimado:** 15-30 minutos
**Nível:** Intermediário
**Pré-requisitos:** Vector DB já configurado

## Objetivo
Otimizar o retrieval para melhor qualidade de busca

## O que é Retrieval Optimization?
Técnicas para melhorar a precisão e relevância dos documentos recuperados:
```
Query → Embedding → Search → Rerank → Top-K Results
```

## Estratégias Principais

### 1. Dense Retrieval (Semantic)
Padrão - busca por similaridade vetorial:
```python
from langchain.vectorstores import Chroma

# Dense retrieval
vectorstore = Chroma.from_documents(chunks, embeddings)
docs = vectorstore.similarity_search(query, k=4)
```

### 2. Sparse Retrieval (Keyword)
Busca por palavras-chave (BM25):
```python
from langchain.retrievers import BM25Retriever

# Sparse retrieval
retriever = BM25Retriever.from_texts(texts)
docs = retriever.get_relevant_documents(query, k=4)
```

### 3. Hybrid Search (Dense + Sparse)
Combina semantic + keyword:
```python
from langchain.retrievers import EnsembleRetriever

dense_retriever = vectorstore.as_retriever(search_k=4)
sparse_retriever = BM25Retriever.from_texts(texts)

# Ensemble
ensemble = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # 70% dense, 30% sparse
)

docs = ensemble.get_relevant_documents(query)
```

### 4. Reranking
Melhorar ordem dos resultados:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cross-encoder reranker
model_name = "BAAI/bge-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Rerank
def rerank(query, documents, top_k=3):
    pairs = [(query, doc.page_content) for doc in documents]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze()

    # Sort by score
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]

# Usage
initial_docs = vectorstore.similarity_search(query, k=10)
reranked_docs = rerank(query, initial_docs, top_k=3)
```

## Comparison

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| **Dense** | Fast | High | Semantic search |
| **Sparse** | Fast | Medium | Keyword matching |
| **Hybrid** | Medium | Very High | General purpose |
| **Rerank** | Slow | Very High | Maximum quality |

## Exemplo Completo

```python
from langchain.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.embeddings import OpenAIEmbeddings

# Setup
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Dense retriever
dense = vectorstore.as_retriever(search_k=5)

# Sparse retriever
sparse = BM25Retriever.from_texts([doc.page_content for doc in chunks])

# Hybrid retriever
hybrid = EnsembleRetriever(
    retrievers=[dense, sparse],
    weights=[0.7, 0.3]
)

# Search
query = "O que é RAG?"
docs = hybrid.get_relevant_documents(query)

print(f"Found {len(docs)} relevant documents:")
for doc in docs:
    print(f"- {doc.page_content[:100]}...")
```

## Parameters

### k (Number of results)
- **k=2-3:** Minimal context
- **k=4-5:** Balanced
- **k=6-8:** Rich context (may confuse LLM)

### Score threshold
```python
# Filter by score
docs_and_scores = vectorstore.similarity_search_with_score(
    query,
    k=5,
    score_threshold=0.7  # Only return high-scoring
)
```

### MMR (Maximal Marginal Relevance)
```python
# Diversify results
docs = vectorstore.max_marginal_relevance_search(
    query,
    k=5,
    lambda_mult=0.5  # 0=diversity, 1=relevance
)
```

## Quando Usar Cada Estratégia

### Dense Retrieval
- ✅ Queries natural language
- ✅ Semantic understanding needed
- ✅ General knowledge

### Sparse Retrieval
- ✅ Exact keyword matching
- ✅ Professional/technical terms
- ✅ Code/IDs

### Hybrid Search
- ✅ Best of both worlds
- ✅ General purpose
- ✅ Complex queries

### Reranking
- ✅ Quality critical
- ✅ Small result set (k<20)
- ✅ Can accept slower search

## Troubleshooting

### Poor relevance
**Soluções:**
1. Increase k
2. Use hybrid search
3. Add reranking
4. Adjust weight ratios

### Slow performance
**Soluções:**
1. Reduce k
2. Skip reranking
3. Cache frequent queries
4. Use simpler retriever

### Diverse results needed
**Soluções:**
```python
# MMR for diversity
docs = vectorstore.max_marginal_relevance_search(
    query,
    k=5,
    lambda_mult=0.6  # Higher = more diverse
)
```

## Próximos Passos

- 💻 **Code Examples:** [Exemplos Completos](../code-examples/)
- 🔧 **Troubleshooting:** [Problemas Comuns](../troubleshooting/common-issues.md)
- 📊 **Evaluation:** [Guia 06 - Evaluation](../06-Evaluation-Benchmarks/README.md)

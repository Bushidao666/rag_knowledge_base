#!/usr/bin/env python3
"""
Example 01: Retrieval Strategies Comparison
===========================================

Compara diferentes estratégias de retrieval.

Uso:
    python example-01-retrieval-strategies.py
"""

import time
from typing import List
from langchain.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def create_sample_data():
    """Criar sample documents"""
    texts = [
        "RAG combines retrieval and generation for better AI",
        "Retrieval finds relevant documents using embeddings",
        "Generation creates human-like responses",
        "Embeddings represent text as vectors",
        "Vector databases store embeddings for fast search",
        "Similarity search finds related content",
        "Chunking divides documents into smaller pieces",
        "Document processing loads and cleans data",
        "LLMs generate text based on input prompts",
        "QA systems answer user questions",
        "Vector similarity uses cosine or dot product",
        "Dense retrieval uses semantic similarity",
        "Sparse retrieval uses keyword matching",
        "Hybrid search combines dense and sparse",
        "Reranking improves result order"
    ]
    return [Document(page_content=text, metadata={"source": f"doc{i}"})
            for i, text in enumerate(texts, 1)]


def dense_retrieval(vectorstore, query, k=5):
    """Dense retrieval (semantic)"""
    start = time.time()
    docs = vectorstore.similarity_search(query, k=k)
    duration = time.time() - start
    return docs, duration


def sparse_retrieval(texts, query, k=5):
    """Sparse retrieval (BM25)"""
    retriever = BM25Retriever.from_texts([doc.page_content for doc in texts])
    start = time.time()
    docs = retriever.get_relevant_documents(query, k=k)
    duration = time.time() - start
    return docs, duration


def hybrid_retrieval(vectorstore, texts, query, k=5):
    """Hybrid retrieval (dense + sparse)"""
    dense = vectorstore.as_retriever(search_k=k)
    sparse = BM25Retriever.from_texts([doc.page_content for doc in texts])

    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.7, 0.3]
    )

    start = time.time()
    docs = ensemble.get_relevant_documents(query, k=k)
    duration = time.time() - start
    return docs, duration


def main():
    """Função principal"""
    print("="*60)
    print("RETRIEVAL STRATEGIES COMPARISON")
    print("="*60)

    # Sample data
    documents = create_sample_data()
    query = "Como RAG melhora a busca de informações?"

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Test different strategies
    print(f"\nQuery: {query}\n")

    # Dense
    docs_dense, time_dense = dense_retrieval(vectorstore, query, k=5)
    print(f"DENSE RETRIEVAL:")
    print(f"  Time: {time_dense:.4f}s")
    print(f"  Results: {len(docs_dense)}")
    for i, doc in enumerate(docs_dense, 1):
        print(f"  {i}. {doc.page_content[:80]}...")

    # Sparse
    docs_sparse, time_sparse = sparse_retrieval(documents, query, k=5)
    print(f"\nSPARSE RETRIEVAL (BM25):")
    print(f"  Time: {time_sparse:.4f}s")
    print(f"  Results: {len(docs_sparse)}")
    for i, doc in enumerate(docs_sparse, 1):
        print(f"  {i}. {doc.page_content[:80]}...")

    # Hybrid
    docs_hybrid, time_hybrid = hybrid_retrieval(vectorstore, documents, query, k=5)
    print(f"\nHYBRID RETRIEVAL:")
    print(f"  Time: {time_hybrid:.4f}s")
    print(f"  Results: {len(docs_hybrid)}")
    for i, doc in enumerate(docs_hybrid, 1):
        print(f"  {i}. {doc.page_content[:80]}...")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}\n")

    print(f"{'Strategy':<15} {'Time (s)':<10} {'Quality':<15}")
    print("-" * 40)
    print(f"{'Dense':<15} {time_dense:<10.4f} {'High':<15}")
    print(f"{'Sparse':<15} {time_sparse:<10.4f} {'Medium':<15}")
    print(f"{'Hybrid':<15} {time_hybrid:<10.4f} {'Very High':<15}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")

    recommendations = """
🎯 Dense Retrieval:
   • Best for: Natural language queries
   • Pros: Semantic understanding
   • Cons: May miss exact keywords

🎯 Sparse Retrieval:
   • Best for: Exact keyword matching
   • Pros: Fast, precise matching
   • Cons: No semantic understanding

🎯 Hybrid Retrieval:
   • Best for: General purpose
   • Pros: Best of both worlds
   • Cons: Slightly slower

⚡ For speed: Sparse
🎯 For quality: Hybrid
🚀 For general use: Dense
"""
    print(recommendations)

    print("="*60)
    print("Comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()

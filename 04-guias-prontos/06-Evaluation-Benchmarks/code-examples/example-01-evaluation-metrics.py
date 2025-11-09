#!/usr/bin/env python3
"""
Example 01: Evaluation Metrics
==============================

Demonstra métricas de avaliação para RAG.

Uso:
    python example-01-evaluation-metrics.py
"""

import time
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import numpy as np


def calculate_recall_k(retrieved, relevant, k):
    """Calculate Recall@K"""
    top_k_retrieved = retrieved[:k]
    relevant_retrieved = len(
        set([doc.page_content for doc in top_k_retrieved]) &
        set(relevant)
    )
    return relevant_retrieved / len(relevant) if relevant else 0


def calculate_precision_k(retrieved, relevant, k):
    """Calculate Precision@K"""
    top_k_retrieved = retrieved[:k]
    relevant_retrieved = len(
        set([doc.page_content for doc in top_k_retrieved]) &
        set(relevant)
    )
    return relevant_retrieved / k if k > 0 else 0


def calculate_dcg(relevance_scores, k):
    """Calculate DCG@K"""
    dcg = 0
    for i in range(min(k, len(relevance_scores))):
        dcg += (2**relevance_scores[i] - 1) / np.log2(i + 2)
    return dcg


def calculate_ndcg_k(retrieved, relevant, k):
    """Calculate nDCG@K"""
    # Assume all relevant have relevance score 2, non-relevant 0
    relevance_scores = [2 if doc.page_content in relevant else 0
                       for doc in retrieved[:k]]

    dcg = calculate_dcg(relevance_scores, k)

    # IDCG - ideal DCG
    ideal_scores = [2] * min(k, len(relevant)) + [0] * max(0, k - len(relevant))
    idcg = calculate_dcg(ideal_scores, k)

    return dcg / idcg if idcg > 0 else 0


def llm_evaluate_faithfulness(question, answer, context):
    """LLM-as-judge para Faithfulness"""
    # Simple heuristic - em produção usar LLM
    context_lower = context.lower()
    answer_lower = answer.lower()

    # Count how many facts from context are in answer
    context_facts = set(context_lower.split('.'))
    answer_facts = set(answer_lower.split('.'))

    matches = len(context_facts & answer_facts)
    faithfulness = min(1.0, matches / max(1, len(context_facts)))

    return faithfulness


def test_retrieval_quality():
    """Test retrieval quality metrics"""
    print("="*60)
    print("RETRIEVAL QUALITY TEST")
    print("="*60)

    # Sample data
    documents = [
        "RAG combines retrieval and generation",
        "Retrieval finds relevant documents",
        "Generation creates responses",
        "Embeddings represent text as vectors",
        "Vector databases store embeddings"
    ]

    query = "What is RAG?"
    relevant_docs = documents[:2]  # First 2 are relevant

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vectorstore = Chroma.from_texts(documents, embeddings)

    # Retrieve
    retrieved_docs = vectorstore.similarity_search(query, k=5)

    # Calculate metrics
    recall = calculate_recall_k(retrieved_docs, relevant_docs, 3)
    precision = calculate_precision_k(retrieved_docs, relevant_docs, 3)
    ndcg = calculate_ndcg_k(retrieved_docs, relevant_docs, 3)

    print(f"\nQuery: {query}")
    print(f"Relevant documents: {len(relevant_docs)}")
    print(f"Retrieved: {len(retrieved_docs)}")
    print(f"\nMetrics:")
    print(f"  Recall@3: {recall:.3f}")
    print(f"  Precision@3: {precision:.3f}")
    print(f"  nDCG@3: {ndcg:.3f}")

    return {
        "recall": recall,
        "precision": precision,
        "ndcg": ndcg
    }


def test_rag_faithfulness():
    """Test RAG faithfulness"""
    print("\n" + "="*60)
    print("RAG FAITHFULNESS TEST")
    print("="*60)

    # RAG setup
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = OpenAI(temperature=0)

    documents = [
        "RAG is Retrieval-Augmented Generation",
        "RAG improves factual accuracy",
        "RAG reduces hallucinations"
    ]

    vectorstore = Chroma.from_texts(documents, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # Test question
    question = "What is RAG?"
    answer = qa.run(question)
    context = "\n".join(documents)

    # Evaluate
    faithfulness = llm_evaluate_faithfulness(question, answer, context)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nFaithfulness Score: {faithfulness:.3f}")

    return {"faithfulness": faithfulness}


def benchmark_rag_performance():
    """Benchmark RAG performance"""
    print("\n" + "="*60)
    print("RAG PERFORMANCE BENCHMARK")
    print("="*60)

    # Setup
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = OpenAI(temperature=0)

    documents = [f"Document {i} about topic {i}" for i in range(100)]

    vectorstore = Chroma.from_documents(documents, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # Benchmark
    queries = [
        "What is RAG?",
        "How to implement RAG?",
        "What are embeddings?",
        "What is vector search?",
        "How to optimize retrieval?"
    ]

    results = []
    for query in queries:
        start = time.time()
        answer = qa.run(query)
        duration = time.time() - start

        results.append({
            "query": query,
            "answer": answer[:100] + "...",
            "latency": duration
        })

    print(f"\nProcessed {len(queries)} queries")
    print(f"Average latency: {np.mean([r['latency'] for r in results]):.3f}s")

    for result in results:
        print(f"\n  Query: {result['query']}")
        print(f"  Latency: {result['latency']:.3f}s")

    return results


def main():
    """Função principal"""
    print("="*60)
    print("EVALUATION & BENCHMARKS")
    print("="*60)

    # Test retrieval quality
    retrieval_metrics = test_retrieval_quality()

    # Test RAG faithfulness
    rag_metrics = test_rag_faithfulness()

    # Benchmark performance
    performance_results = benchmark_rag_performance()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\nRetrieval Metrics:")
    print(f"  Recall: {retrieval_metrics['recall']:.3f}")
    print(f"  Precision: {retrieval_metrics['precision']:.3f}")
    print(f"  nDCG: {retrieval_metrics['ndcg']:.3f}")

    print(f"\nRAG Quality:")
    print(f"  Faithfulness: {rag_metrics['faithfulness']:.3f}")

    print(f"\nPerformance:")
    avg_latency = np.mean([r['latency'] for r in performance_results])
    print(f"  Average latency: {avg_latency:.3f}s")

    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()

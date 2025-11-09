# Code Examples: Fase 3 (Seções 05-06)

### Data: 09/11/2025
### Status: Executáveis no Windows
### Foco: Retrieval Optimization + Evaluation

---

## EXAMPLE 1: Hybrid Search Implementation

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai sentence-transformers
```

### Complete Hybrid Retrieval

```python
"""
Hybrid Search: Dense + Sparse Retrieval
Combina semantic search (BGE) com keyword search (BM25)
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import numpy as np
import time

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and sparse retrieval."""

    def __init__(self, dense_retriever, sparse_retriever, alpha=0.7):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha  # Weight for dense retrieval (0-1)

    def _retrieve(self, query, **kwargs) -> List[Document]:
        """Retrieve documents using hybrid approach."""
        # Get results from both retrievers
        dense_results = self.dense_retriever.get_relevant_documents(query, k=20)
        sparse_results = self.sparse_retriever.get_relevant_documents(query, k=20)

        # Normalize scores
        max_dense = max([r.metadata.get("score", 0) for r in dense_results]) if dense_results else 1
        max_sparse = max([r.metadata.get("score", 0) for r in sparse_results]) if sparse_results else 1

        # Combine results
        combined_docs = {}

        # Process dense results
        for doc in dense_results:
            norm_score = doc.metadata.get("score", 0) / max_dense if max_dense > 0 else 0
            doc_id = doc.page_content[:100]  # Use first 100 chars as ID
            combined_docs[doc_id] = {
                "doc": doc,
                "score": self.alpha * norm_score
            }

        # Process sparse results
        for doc in sparse_results:
            norm_score = doc.metadata.get("score", 0) / max_sparse if max_sparse > 0 else 0
            doc_id = doc.page_content[:100]
            if doc_id in combined_docs:
                combined_docs[doc_id]["score"] += (1 - self.alpha) * norm_score
            else:
                combined_docs[doc_id] = {
                    "doc": doc,
                    "score": (1 - self.alpha) * norm_score
                }

        # Sort by combined score
        sorted_docs = sorted(combined_docs.values(), key=lambda x: x["score"], reverse=True)

        # Get top-k
        k = kwargs.get("k", 10)
        return [item["doc"] for item in sorted_docs[:k]]

def create_hybrid_retriever(texts, model_name="BAAI/bge-large-en-v1.5", alpha=0.7):
    """Create hybrid retriever with dense and sparse components."""

    # Create dense retriever
    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents([Document(text=text) for text in texts])
    dense_retriever = vectorstore.as_retriever(search_k=20)

    # Create sparse retriever
    print("Creating BM25 retriever")
    sparse_retriever = TFIDFRetriever.from_texts(texts, k=20)

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        alpha=alpha
    )

    return hybrid_retriever

# Example usage
if __name__ == "__main__":
    # Sample documents
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret visual data",
        "Python is a popular programming language for data science",
        "TensorFlow and PyTorch are frameworks for deep learning",
        "Supervised learning uses labeled training data",
        "Unsupervised learning finds patterns in unlabeled data"
    ]

    # Create hybrid retriever
    hybrid_retriever = create_hybrid_retriever(texts, alpha=0.7)

    # Test queries
    test_queries = [
        "What is machine learning?",
        "Python programming language",
        "Neural networks",
        "Data science tools"
    ]

    print("\n" + "="*60)
    print("HYBRID RETRIEVAL RESULTS")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        start = time.time()
        results = hybrid_retriever.get_relevant_documents(query, k=3)
        elapsed = time.time() - start

        print(f"Top 3 results (time: {elapsed*1000:.1f}ms):")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Score: {doc.metadata.get('score', 0):.3f}")
            print(f"   Text: {doc.page_content}")
```

---

## EXAMPLE 2: Reranking with Cross-Encoders

### Prerequisites

```bash
pip install sentence-transformers langchain langchain-community
```

### Reranking Implementation

```python
"""
Reranking Implementation
Uses cross-encoders to re-rank initial retrieval results
"""

from langchain_community.cross_encoders import CrossEncoder
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import time

class RerankRetriever(BaseRetriever):
    """Retriever with cross-encoder reranking."""

    def __init__(self, base_retriever, reranker_model, k_initial=50, k_final=10):
        self.base_retriever = base_retriever
        self.reranker = CrossEncoder(reranker_model)
        self.k_initial = k_initial
        self.k_final = k_final

    def _retrieve(self, query, **kwargs) -> List[Document]:
        """Retrieve with reranking."""
        # Step 1: Initial retrieval
        print(f"Initial retrieval (k={self.k_initial})...")
        initial_results = self.base_retriever.get_relevant_documents(
            query, k=self.k_initial
        )

        if not initial_results:
            return []

        # Step 2: Prepare query-document pairs for reranking
        query_doc_pairs = [
            (query, doc.page_content) for doc in initial_results
        ]

        # Step 3: Rerank using cross-encoder
        print(f"Reranking {len(query_doc_pairs)} documents...")
        start = time.time()
        scores = self.reranker.predict(query_doc_pairs)
        rerank_time = time.time() - start

        print(f"Reranking time: {rerank_time*1000:.1f}ms")

        # Step 4: Attach scores and sort
        for doc, score in zip(initial_results, scores):
            doc.metadata["rerank_score"] = float(score)

        # Sort by rerank score (descending)
        reranked = sorted(initial_results, key=lambda x: x.metadata["rerank_score"], reverse=True)

        # Return top-k
        return reranked[:self.k_final]

def create_rerank_pipeline(texts, reranker_model="BAAI/bge-reranker-base"):
    """Create complete retriever pipeline with reranking."""

    # Initial retriever (simple BM25)
    base_retriever = TFIDFRetriever.from_texts(texts, k=50)

    # Reranker
    reranker = RerankRetriever(
        base_retriever=base_retriever,
        reranker_model=reranker_model,
        k_initial=50,
        k_final=10
    )

    return reranker

# Example: Compare with and without reranking
if __name__ == "__main__":
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Neural networks are computational models inspired by the brain",
        "Python is a programming language used in machine learning",
        "TensorFlow is a framework for deep learning",
        "Convolutional neural networks are used for image recognition",
        "Recurrent neural networks handle sequential data",
        "Transformers are a type of neural network architecture"
    ]

    # Base retriever (no reranking)
    base_retriever = TFIDFRetriever.from_texts(texts, k=10)

    # Rerank retriever
    rerank_retriever = create_rerank_pipeline(texts)

    # Test query
    query = "What are neural networks?"

    print("\n" + "="*60)
    print("RERANKING COMPARISON")
    print("="*60)

    # Without reranking
    print(f"\nQuery: {query}")
    print("\nWithout Reranking:")
    print("-" * 60)
    start = time.time()
    base_results = base_retriever.get_relevant_documents(query, k=5)
    base_time = time.time() - start

    for i, doc in enumerate(base_results, 1):
        print(f"{i}. {doc.page_content}")

    print(f"Time: {base_time*1000:.1f}ms")

    # With reranking
    print("\nWith Reranking:")
    print("-" * 60)
    start = time.time()
    rerank_results = rerank_retriever.get_relevant_documents(query, k=5)
    rerank_time = time.time() - start

    for i, doc in enumerate(rerank_results, 1):
        print(f"{i}. [Score: {doc.metadata.get('rerank_score', 0):.3f}] {doc.page_content}")

    print(f"Time: {rerank_time*1000:.1f}ms")

    # Show score improvement
    print("\nScore Comparison:")
    print("-" * 60)
    print("Without Reranking (BM25 scores):")
    for i, doc in enumerate(base_results, 1):
        print(f"  {i}. {doc.metadata.get('score', 0):.3f}")

    print("\nWith Reranking (Cross-Encoder scores):")
    for i, doc in enumerate(rerank_results, 1):
        print(f"  {i}. {doc.metadata.get('rerank_score', 0):.3f}")
```

---

## EXAMPLE 3: Evaluation with RAGAS

### Prerequisites

```bash
pip install ragas datasets
```

### Complete RAGAS Evaluation

```python
"""
RAGAS Evaluation Framework
Evaluate RAG systems with RAG-specific metrics
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevance,
    aspect_critique
)
from datasets import Dataset
import json
import numpy as np

class RAGEvaluator:
    """RAG evaluator using RAGAS."""

    def __init__(self):
        self.test_data = []
        self.results = None

    def add_test_case(self, question, answer, contexts, reference=None):
        """Add a test case."""
        self.test_data.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": reference  # Ground truth answer (optional)
        })

    def load_test_set(self, file_path):
        """Load test set from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            self.add_test_case(
                question=item["question"],
                answer=item["answer"],
                contexts=item["contexts"],
                reference=item.get("reference")
            )

        print(f"Loaded {len(self.test_data)} test cases")

    def run_evaluation(self, metrics=None):
        """Run RAGAS evaluation."""
        if not self.test_data:
            print("No test data. Add test cases first.")
            return None

        if metrics is None:
            metrics = [
                faithfulness,
                context_precision,
                context_recall,
                answer_relevance
            ]

        # Prepare data
        questions = [item["question"] for item in self.test_data]
        answers = [item["answer"] for item in self.test_data]
        contexts = [item["contexts"] for item in self.test_data]

        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }

        dataset = Dataset.from_dict(dataset_dict)

        print(f"\nRunning evaluation on {len(self.test_data)} test cases...")
        print("Metrics:", [m.name for m in metrics])

        # Run evaluation
        self.results = evaluate(dataset, metrics=metrics)

        return self.results

    def print_results(self):
        """Print evaluation results."""
        if self.results is None:
            print("No results. Run evaluation first.")
            return

        print("\n" + "="*70)
        print("RAGAS EVALUATION RESULTS")
        print("="*70)

        # Overall scores
        for metric_name in self.results.get_metric_names():
            scores = self.results[metric_name]
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            print(f"\n{metric_name}:")
            print(f"  Mean: {mean_score:.4f}")
            print(f"  Std:  {std_score:.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")

        # Detailed results per test case
        print("\n" + "="*70)
        print("DETAILED RESULTS")
        print("="*70)

        for i, test_case in enumerate(self.test_data):
            print(f"\n[{i+1}] Question: {test_case['question'][:60]}...")
            print(f"    Answer: {test_case['answer'][:60]}...")

            for metric_name in self.results.get_metric_names():
                score = self.results[metric_name][i]
                print(f"    {metric_name}: {score:.3f}")

    def save_results(self, output_file):
        """Save results to file."""
        if self.results is None:
            print("No results to save.")
            return

        results_dict = {
            "num_test_cases": len(self.test_data),
            "metrics": {}
        }

        for metric_name in self.results.get_metric_names():
            scores = self.results[metric_name]
            results_dict["metrics"][metric_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "scores": [float(s) for s in scores]
            }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {output_file}")

    def compare_versions(self, baseline_results, new_results):
        """Compare two evaluation runs."""
        print("\n" + "="*70)
        print("EVALUATION COMPARISON")
        print("="*70)

        for metric_name in baseline_results["metrics"].keys():
            if metric_name in new_results["metrics"]:
                baseline_mean = baseline_results["metrics"][metric_name]["mean"]
                new_mean = new_results["metrics"][metric_name]["mean"]
                diff = new_mean - baseline_mean
                pct_change = (diff / baseline_mean) * 100

                print(f"\n{metric_name}:")
                print(f"  Baseline: {baseline_mean:.4f}")
                print(f"  New:      {new_mean:.4f}")
                print(f"  Change:   {diff:+.4f} ({pct_change:+.1f}%)")

                if pct_change > 1:
                    print(f"  ✅ Improvement")
                elif pct_change < -1:
                    print(f"  ❌ Degradation")
                else:
                    print(f"  ➖ No significant change")

# Create sample test data
def create_sample_test_set():
    """Create sample test set for demonstration."""
    test_set = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "contexts": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "It is a branch of artificial intelligence based on the idea that systems can learn from data."
            ]
        },
        {
            "question": "What is deep learning?",
            "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
            "contexts": [
                "Deep learning is part of a broader family of machine learning methods.",
                "It is based on artificial neural networks with representation learning."
            ]
        },
        {
            "question": "What is Python used for?",
            "answer": "Python is a high-level programming language widely used for web development, data science, artificial intelligence, and scientific computing.",
            "contexts": [
                "Python is a versatile programming language.",
                "It is commonly used in data science and machine learning projects."
            ]
        }
    ]

    with open("test_set.json", 'w') as f:
        json.dump(test_set, f, indent=2)

    print("Created sample test set: test_set.json")

# Usage example
if __name__ == "__main__":
    # Create sample test set
    create_sample_test_set()

    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Load test set
    evaluator.load_test_set("test_set.json")

    # Run evaluation
    results = evaluator.run_evaluation()

    # Print results
    evaluator.print_results()

    # Save results
    evaluator.save_results("ragas_results.json")

    # Create baseline for comparison
    baseline = {
        "num_test_cases": 3,
        "metrics": {
            "faithfulness": {
                "mean": 0.85,
                "std": 0.10,
                "min": 0.70,
                "max": 0.95,
                "scores": [0.80, 0.90, 0.85]
            },
            "context_precision": {
                "mean": 0.75,
                "std": 0.15,
                "min": 0.60,
                "max": 0.90,
                "scores": [0.70, 0.80, 0.75]
            },
            "answer_relevance": {
                "mean": 0.80,
                "std": 0.10,
                "min": 0.70,
                "max": 0.90,
                "scores": [0.75, 0.85, 0.80]
            }
        }
    }

    # Load current results for comparison
    with open("ragas_results.json") as f:
        current = json.load(f)

    # Compare
    evaluator.compare_versions(baseline, current)
```

---

## EXAMPLE 4: Custom Metrics Implementation

### Prerequisites

```bash
pip install langchain-openai
```

### Custom Evaluation Metrics

```python
"""
Custom Evaluation Metrics
Implement domain-specific evaluation metrics
"""

import numpy as np
from typing import List, Dict, Any
from langchain_openai import OpenAI
import re

class CustomRAGMetrics:
    """Custom evaluation metrics for RAG systems."""

    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(api_key=openai_api_key)

    def factual_correctness(self, answer: str, contexts: List[str]) -> float:
        """
        Check if facts in answer are supported by contexts.
        Returns score between 0 and 1.
        """
        # Simple approach: Check for fact patterns
        facts_in_answer = re.findall(r'\b\d{4}\b', answer)  # Find years

        if not facts_in_answer:
            return 1.0  # No specific facts, assume correct

        # Check if facts appear in contexts
        supported_facts = 0
        for fact in facts_in_answer:
            if any(fact in context for context in contexts):
                supported_facts += 1

        return supported_facts / len(facts_in_answer)

    def coherence_score(self, answer: str) -> float:
        """
        Evaluate coherence of answer using LLM-as-judge.
        Returns score between 0 and 1.
        """
        prompt = f"""
        Rate the coherence of the following answer on a scale of 0-1,
        where 1 is highly coherent and 0 is not coherent at all.

        Answer: {answer}

        Coherence score (0-1):"""

        try:
            response = self.llm.generate(prompt)
            score_text = response.generations[0][0].text.strip()
            # Extract number from response
            score = float(re.search(r'0?\.\d+|1\.0', score_text).group())
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        except:
            return 0.5  # Default if error

    def completeness_score(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if answer addresses all aspects of question.
        Returns score between 0 and 1.
        """
        prompt = f"""
        Given the following question and answer, rate the completeness of the answer
        on a scale of 0-1, where 1 means the answer fully addresses all aspects of the question.

        Question: {question}
        Answer: {answer}

        Completeness score (0-1):"""

        try:
            response = self.llm.generate(prompt)
            score_text = response.generations[0][0].text.strip()
            score = float(re.search(r'0?\.\d+|1\.0', score_text).group())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5

    def information_density(self, answer: str) -> float:
        """
        Calculate information density (useful information / total words).
        Returns score between 0 and 1.
        """
        # Remove stop words for density calculation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        words = answer.lower().split()
        content_words = [w for w in words if w not in stop_words and len(w) > 3]

        if not words:
            return 0.0

        return len(content_words) / len(words)

    def evaluate_rag_response(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Evaluate a single RAG response with multiple metrics.
        """
        results = {
            "factual_correctness": self.factual_correctness(answer, contexts),
            "coherence": self.coherence_score(answer),
            "completeness": self.completeness_score(question, answer, contexts),
            "information_density": self.information_density(answer)
        }

        # Calculate overall score
        weights = {
            "factual_correctness": 0.4,
            "coherence": 0.2,
            "completeness": 0.3,
            "information_density": 0.1
        }

        overall = sum(results[metric] * weights[metric] for metric in weights)
        results["overall"] = overall

        return results

def evaluate_test_set(evaluator: CustomRAGMetrics, test_data: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate entire test set.
    """
    all_results = {
        "factual_correctness": [],
        "coherence": [],
        "completeness": [],
        "information_density": [],
        "overall": []
    }

    print(f"\nEvaluating {len(test_data)} responses...")

    for i, item in enumerate(test_data):
        print(f"Processing {i+1}/{len(test_data)}", end="\r")

        results = evaluator.evaluate_rag_response(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"]
        )

        for metric, score in results.items():
            all_results[metric].append(score)

    print()  # New line after progress

    # Calculate statistics
    stats = {}
    for metric, scores in all_results.items():
        stats[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }

    return stats

# Usage example
if __name__ == "__main__":
    import os

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    if api_key == "your-api-key-here":
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        exit(1)

    # Initialize evaluator
    evaluator = CustomRAGMetrics(openai_api_key=api_key)

    # Test data
    test_data = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions.",
            "contexts": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "It is a branch of artificial intelligence based on the idea that systems can learn from data."
            ]
        },
        {
            "question": "Explain deep learning",
            "answer": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. These networks can automatically learn hierarchical representations of data, making them particularly effective for complex tasks like image recognition and natural language processing.",
            "contexts": [
                "Deep learning is based on artificial neural networks with representation learning.",
                "The adjective 'deep' in deep learning refers to the use of multiple layers in the network."
            ]
        },
        {
            "question": "What is Python used for?",
            "answer": "Python is a high-level programming language that is widely used in web development, data science, artificial intelligence, scientific computing, and automation. Its simple syntax and extensive library ecosystem make it popular among developers.",
            "contexts": [
                "Python is a versatile programming language with a simple syntax.",
                "It is commonly used in data science and machine learning projects."
            ]
        }
    ]

    # Run evaluation
    results = evaluate_test_set(evaluator, test_data)

    # Print results
    print("\n" + "="*70)
    print("CUSTOM METRICS EVALUATION RESULTS")
    print("="*70)

    for metric, stats in results.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  Max:    {stats['max']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
```

---

## EXAMPLE 5: Production Monitoring & A/B Testing

### Prerequisites

```bash
pip install langchain langchain-openai scikit-learn
```

### Production Monitoring System

```python
"""
Production Monitoring for RAG Systems
A/B testing, performance tracking, and alerting
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import statistics
import random

@dataclass
class QueryLog:
    """Log entry for a single query."""
    timestamp: datetime
    query: str
    answer: str
    latency_ms: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    version: str = "v1.0"  # RAG system version
    context_count: int = 0
    user_rating: Optional[int] = None  # 1-5 scale
    user_feedback: Optional[str] = None

class RAGMonitoringSystem:
    """Production monitoring for RAG systems."""

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.query_logs: List[QueryLog] = []
        self.metrics_history: List[Dict] = []

    def log_query(
        self,
        query: str,
        answer: str,
        latency_ms: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        version: str = "v1.0",
        context_count: int = 0,
        user_rating: Optional[int] = None,
        user_feedback: Optional[str] = None
    ):
        """Log a query to the monitoring system."""
        log = QueryLog(
            timestamp=datetime.now(),
            query=query,
            answer=answer,
            latency_ms=latency_ms,
            user_id=user_id,
            session_id=session_id,
            version=version,
            context_count=context_count,
            user_rating=user_rating,
            user_feedback=user_feedback
        )

        self.query_logs.append(log)

    def get_recent_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics from recent time window."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_logs = [log for log in self.query_logs if log.timestamp >= cutoff]

        if not recent_logs:
            return {}

        # Calculate metrics
        latencies = [log.latency_ms for log in recent_logs]
        ratings = [log.user_rating for log in recent_logs if log.user_rating is not None]

        metrics = {
            "time_window_hours": hours,
            "num_queries": len(recent_logs),
            "latency": {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "min_ms": min(latencies),
                "max_ms": max(latencies)
            },
            "user_satisfaction": {
                "num_ratings": len(ratings),
                "mean_rating": statistics.mean(ratings) if ratings else None,
                "median_rating": statistics.median(ratings) if ratings else None
            }
        }

        return metrics

    def detect_anomalies(self, threshold_p95_ms: float = 2000, threshold_rating: float = 3.0):
        """Detect anomalies in recent metrics."""
        metrics = self.get_recent_metrics(hours=1)

        anomalies = []

        # Check latency
        if "latency" in metrics:
            if metrics["latency"]["p95_ms"] > threshold_p95_ms:
                anomalies.append({
                    "type": "high_latency",
                    "message": f"P95 latency ({metrics['latency']['p95_ms']:.1f}ms) exceeded threshold ({threshold_p95_ms}ms)",
                    "severity": "warning"
                })

        # Check user satisfaction
        if "user_satisfaction" in metrics and metrics["user_satisfaction"]["num_ratings"] > 0:
            if metrics["user_satisfaction"]["mean_rating"] < threshold_rating:
                anomalies.append({
                    "type": "low_satisfaction",
                    "message": f"User satisfaction ({metrics['user_satisfaction']['mean_rating']:.2f}) below threshold ({threshold_rating})",
                    "severity": "warning"
                })

        return anomalies

    def generate_report(self, hours: int = 24) -> str:
        """Generate a text report of recent metrics."""
        metrics = self.get_recent_metrics(hours)

        if not metrics:
            return f"No data for last {hours} hours"

        report = f"\n{'='*70}\n"
        report += f"RAG System Monitoring Report - {self.system_name}\n"
        report += f"{'='*70}\n"
        report += f"Time window: Last {hours} hours\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*70}\n"

        # Query volume
        report += f"\nQuery Volume:\n"
        report += f"  Total queries: {metrics['num_queries']}\n"

        # Latency
        report += f"\nLatency (ms):\n"
        report += f"  Mean:   {metrics['latency']['mean_ms']:.1f}\n"
        report += f"  Median: {metrics['latency']['median_ms']:.1f}\n"
        report += f"  P95:    {metrics['latency']['p95_ms']:.1f}\n"
        report += f"  P99:    {metrics['latency']['p99_ms']:.1f}\n"
        report += f"  Range:  {metrics['latency']['min_ms']:.1f} - {metrics['latency']['max_ms']:.1f}\n"

        # User satisfaction
        if metrics["user_satisfaction"]["num_ratings"] > 0:
            report += f"\nUser Satisfaction:\n"
            report += f"  Ratings collected: {metrics['user_satisfaction']['num_ratings']}\n"
            report += f"  Mean rating: {metrics['user_satisfaction']['mean_rating']:.2f}/5.0\n"
            report += f"  Median rating: {metrics['user_satisfaction']['median_rating']:.2f}/5.0\n"
        else:
            report += f"\nUser Satisfaction: No ratings collected\n"

        # Anomalies
        anomalies = self.detect_anomalies()
        if anomalies:
            report += f"\n⚠️  Anomalies Detected:\n"
            for anomaly in anomalies:
                report += f"  - {anomaly['message']}\n"
        else:
            report += f"\n✅ No anomalies detected\n"

        report += f"{'='*70}\n"

        return report

class ABTestManager:
    """A/B testing manager for comparing RAG system versions."""

    def __init__(self, test_name: str, control_version: str, treatment_version: str, traffic_split: float = 0.5):
        self.test_name = test_name
        self.control_version = control_version
        self.treatment_version = treatment_version
        self.traffic_split = traffic_split
        self.control_logs: List[QueryLog] = []
        self.treatment_logs: List[QueryLog] = []

    def assign_variant(self, user_id: str) -> str:
        """Assign user to control or treatment based on consistent hashing."""
        # Simple hash-based assignment
        hash_val = hash(user_id) % 100
        return "treatment" if hash_val < self.traffic_split * 100 else "control"

    def log_query(self, query: str, answer: str, latency_ms: float,
                  user_id: str, user_rating: Optional[int] = None, version: str = "v1.0"):
        """Log query with variant assignment."""
        variant = self.assign_variant(user_id)

        log = QueryLog(
            timestamp=datetime.now(),
            query=query,
            answer=answer,
            latency_ms=latency_ms,
            user_id=user_id,
            version=version
        )

        if variant == "control":
            self.control_logs.append(log)
        else:
            self.treatment_logs.append(log)

    def get_results(self) -> Dict[str, Any]:
        """Calculate A/B test results."""
        results = {
            "test_name": self.test_name,
            "control": {
                "version": self.control_version,
                "num_samples": len(self.control_logs)
            },
            "treatment": {
                "version": self.treatment_version,
                "num_samples": len(self.treatment_logs)
            }
        }

        # Compare latency
        if self.control_logs and self.treatment_logs:
            control_latencies = [log.latency_ms for log in self.control_logs]
            treatment_latencies = [log.latency_ms for log in self.treatment_logs]

            control_latency = statistics.mean(control_latencies)
            treatment_latency = statistics.mean(treatment_latencies)

            results["latency"] = {
                "control_mean_ms": control_latency,
                "treatment_mean_ms": treatment_latency,
                "difference_ms": treatment_latency - control_latency,
                "percent_change": ((treatment_latency - control_latency) / control_latency) * 100
            }

        # Compare user satisfaction
        control_ratings = [log.user_rating for log in self.control_logs if log.user_rating]
        treatment_ratings = [log.user_rating for log in self.treatment_logs if log.user_rating]

        if control_ratings and treatment_ratings:
            control_rating = statistics.mean(control_ratings)
            treatment_rating = statistics.mean(treatment_ratings)

            results["user_satisfaction"] = {
                "control_mean": control_rating,
                "treatment_mean": treatment_rating,
                "difference": treatment_rating - control_rating,
                "percent_change": ((treatment_rating - control_rating) / control_rating) * 100
            }

        return results

    def print_results(self):
        """Print A/B test results."""
        results = self.get_results()

        print(f"\n{'='*70}")
        print(f"A/B Test Results: {self.test_name}")
        print(f"{'='*70}")

        print(f"\nSample sizes:")
        print(f"  Control: {results['control']['num_samples']}")
        print(f"  Treatment: {results['treatment']['num_samples']}")

        if "latency" in results:
            print(f"\nLatency:")
            print(f"  Control:   {results['latency']['control_mean_ms']:.1f}ms")
            print(f"  Treatment: {results['latency']['treatment_mean_ms']:.1f}ms")
            print(f"  Change:    {results['latency']['difference_ms']:+.1f}ms ({results['latency']['percent_change']:+.1f}%)")

        if "user_satisfaction" in results:
            print(f"\nUser Satisfaction:")
            print(f"  Control:   {results['user_satisfaction']['control_mean']:.2f}/5.0")
            print(f"  Treatment: {results['user_satisfaction']['treatment_mean']:.2f}/5.0")
            print(f"  Change:    {results['user_satisfaction']['difference']:+.2f} ({results['user_satisfaction']['percent_change']:+.1f}%)")

        # Determine winner
        winner = None
        if "user_satisfaction" in results and results["user_satisfaction"]["treatment_mean"] > results["user_satisfaction"]["control_mean"]:
            winner = "treatment"
        elif "user_satisfaction" in results:
            winner = "control"

        if winner:
            print(f"\n{'✅' if winner == 'treatment' else '❌'} Winner: {winner.upper()}")

        print(f"{'='*70}\n")

# Example usage
if __name__ == "__main__":
    # Simulate production monitoring
    print("Simulating RAG Production Monitoring...")

    # Initialize monitoring system
    monitor = RAGMonitoringSystem("My RAG App")

    # Simulate queries
    for i in range(100):
        user_id = f"user_{i}"
        query = f"Sample query {i}"
        answer = f"Sample answer {i}"
        latency = random.uniform(100, 500)  # Random latency 100-500ms

        # Log query
        monitor.log_query(
            query=query,
            answer=answer,
            latency_ms=latency,
            user_id=user_id
        )

        # Occasionally add a rating
        if i % 20 == 0:
            rating = random.randint(3, 5)  # Mostly positive ratings
            # In real system, would update the log entry

    # Generate report
    report = monitor.generate_report(hours=1)
    print(report)

    # Check for anomalies
    anomalies = monitor.detect_anomalies()
    if anomalies:
        print("\n⚠️  Anomalies:")
        for anomaly in anomalies:
            print(f"  {anomaly['message']}")

    # A/B Testing example
    print("\n" + "="*70)
    print("A/B Testing Example")
    print("="*70)

    ab_test = ABTestManager(
        test_name="Chunk Size Test",
        control_version="chunk_1000",
        treatment_version="chunk_500",
        traffic_split=0.5
    )

    # Simulate A/B test traffic
    for i in range(200):
        user_id = f"user_{i}"
        query = f"Query {i}"
        answer = f"Answer {i}"
        latency = random.uniform(150, 400)

        ab_test.log_query(
            query=query,
            answer=answer,
            latency_ms=latency,
            user_id=user_id
        )

    # Print results
    ab_test.print_results()
```

---

## USAGE INSTRUCTIONS

### Prerequisites Installation

```powershell
# Core dependencies
pip install langchain langchain-community langchain-openai
pip install sentence-transformers
pip install ragas datasets
pip install numpy scikit-learn

# For cross-encoders
pip install sentence-transformers
```

### Running the Examples

#### Example 1: Hybrid Search
```bash
python example1_hybrid_search.py
```

#### Example 2: Reranking
```bash
# Download reranker model
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

python example2_reranking.py
```

#### Example 3: RAGAS Evaluation
```bash
python example3_ragas_evaluation.py
```

#### Example 4: Custom Metrics
```bash
# Set OpenAI API key
$env:OPENAI_API_KEY = "your-api-key-here"

python example4_custom_metrics.py
```

#### Example 5: Production Monitoring
```bash
python example5_monitoring.py
```

### Windows-Specific Notes

1. **Performance**: Use WSL2 for better performance with large models
2. **Memory**: Close other applications when running reranking
3. **API Keys**: Use environment variables in PowerShell
4. **Caching**: Enable caching for repeated evaluations

### PowerShell Automation

```powershell
# Run all examples
.\run_all_examples.ps1

# Or run individually
python example1_hybrid_search.py
python example2_reranking.py
python example3_ragas_evaluation.py
```

### Performance Tips

1. **Batch Processing**: Process multiple queries together
2. **Caching**: Cache frequent queries and results
3. **GPU**: Use CUDA if available for faster reranking
4. **Monitoring**: Set up alerts for performance degradation

---

**Status**: ✅ Code examples Fase 3 created
**Próximo**: Resumo Executivo Fase 3
**Data Conclusão**: 09/11/2025

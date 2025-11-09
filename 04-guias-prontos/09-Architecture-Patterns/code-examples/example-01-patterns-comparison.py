#!/usr/bin/env python3
"""
Example 01: Architecture Patterns Comparison
=========================================

Compara diferentes padrões arquiteturais RAG.

Uso:
    python example-01-patterns-comparison.py
"""

from typing import List
from langchain.schema import Document


class NaiveRAG:
    """Pattern 1: Naive RAG"""
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def query(self, question):
        # Direct approach
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        answer = self.llm(f"Context: {context}\nQuestion: {question}")
        return answer


class ChunkJoinRAG:
    """Pattern 2: Chunk-Join RAG"""
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def query(self, question):
        # Get chunks
        chunks = self.vectorstore.similarity_search(question, k=10)

        # Join ALL chunks
        full_documents = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id")
            if doc_id not in full_documents:
                full_documents[doc_id] = chunk.metadata["content"]

        context = "\n\n".join(full_documents.values())

        # Generate with full context
        answer = self.llm(f"Context: {context}\nQuestion: {question}")
        return answer


class ParentDocumentRAG:
    """Pattern 3: Parent-Document RAG"""
    def __init__(self, vectorstore, llm, parent_store):
        self.vectorstore = vectorstore
        self.llm = llm
        self.parent_store = parent_store

    def query(self, question):
        # Get relevant chunks
        chunks = self.vectorstore.similarity_search(question, k=5)

        # Get parent documents
        parent_ids = list(set([chunk.metadata["parent_id"] for chunk in chunks]))
        parent_docs = [self.parent_store[id] for id in parent_ids]

        context = "\n\n".join(parent_docs)

        return self.llm(f"Context: {context}\nQuestion: {question}")


def compare_patterns():
    """Compare different patterns"""
    print("="*60)
    print("ARCHITECTURE PATTERNS COMPARISON")
    print("="*60)

    patterns = {
        "Naive": {
            "complexity": "Low",
            "speed": "Fast",
            "quality": "Medium",
            "use_case": "Simple Q&A"
        },
        "Chunk-Join": {
            "complexity": "Medium",
            "speed": "Medium",
            "quality": "High",
            "use_case": "Full context needed"
        },
        "Parent-Doc": {
            "complexity": "High",
            "speed": "Medium",
            "quality": "High",
            "use_case": "Hierarchical docs"
        }
    }

    print(f"\n{'Pattern':<15} {'Complexity':<12} {'Speed':<10} {'Quality':<10} {'Use Case':<20}")
    print("-" * 70)

    for pattern, specs in patterns.items():
        print(f"{pattern:<15} {specs['complexity']:<12} "
              f"{specs['speed']:<10} {specs['quality']:<10} {specs['use_case']:<20}")


def main():
    """Função principal"""
    compare_patterns()

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")

    recommendations = """
🎯 Naive RAG:
   • Start here for most projects
   • Simple to implement
   • Good baseline

🎯 Chunk-Join RAG:
   • When full documents matter
   • Better for comprehension
   • Slower but more complete

🎯 Parent-Document RAG:
   • For structured documents
   • Preserves document hierarchy
   • Best for complex layouts

💡 Choose based on:
   1. Complexity tolerance
   2. Quality requirements
   3. Latency constraints
   4. Data structure
"""
    print(recommendations)

    print("="*60)
    print("Comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()

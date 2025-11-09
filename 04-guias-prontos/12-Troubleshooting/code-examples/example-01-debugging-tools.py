#!/usr/bin/env python3
"""
Example 01: Debugging Tools
==========================

Ferramentas para debugging de sistemas RAG.

Uso:
    python example-01-debugging-tools.py
"""

import time
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class RAGDebugger:
    """Debugger para sistemas RAG"""

    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    def inspect_vectorstore(self):
        """Inspect vector store"""
        print("="*60)
        print("VECTOR STORE INSPECTION")
        print("="*60)

        # Count
        count = self.vectorstore._collection.count()
        print(f"Total vectors: {count}")

        # Sample
        sample = self.vectorstore.get(limit=5)
        print(f"\nSample metadata:")
        for i, meta in enumerate(sample['metadatas'], 1):
            print(f"  {i}. {meta}")

    def test_embedding_consistency(self, texts: List[str]):
        """Test embedding consistency"""
        print("\n" + "="*60)
        print("EMBEDDING CONSISTENCY TEST")
        print("="*60)

        # Test same text
        text = texts[0]
        v1 = self.embeddings.embed_query(text)
        v2 = self.embeddings.embed_query(text)

        if v1 == v2:
            print("✅ Embeddings are consistent")
        else:
            print("❌ Embeddings are NOT consistent!")

        # Show dimensions
        print(f"Embedding dimension: {len(v1)}")

    def test_similarity(self, query: str):
        """Test similarity search"""
        print("\n" + "="*60)
        print("SIMILARITY SEARCH TEST")
        print("="*60)

        start = time.time()
        docs = self.vectorstore.similarity_search(query, k=3)
        duration = time.time() - start

        print(f"Query: {query}")
        print(f"Time: {duration:.3f}s")
        print(f"Results: {len(docs)}")

        for i, doc in enumerate(docs, 1):
            score = doc.metadata.get('score', 'N/A')
            print(f"\n{i}. Score: {score}")
            print(f"   Content: {doc.page_content[:100]}...")

    def profile_query(self, query: str, iterations: int = 10):
        """Profile query performance"""
        print("\n" + "="*60)
        print(f"QUERY PROFILE ({iterations} iterations)")
        print("="*60)

        times = []
        for i in range(iterations):
            start = time.time()
            docs = self.vectorstore.similarity_search(query, k=3)
            duration = time.time() - start
            times.append(duration)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"Average: {avg_time:.3f}s")
        print(f"Min: {min_time:.3f}s")
        print(f"Max: {max_time:.3f}s")

        if avg_time > 1.0:
            print("⚠️  Query is slow (avg > 1s)")
        else:
            print("✅ Query is fast (avg < 1s)")

    def validate_chunking(self, chunks: List[Document]):
        """Validate chunking quality"""
        print("\n" + "="*60)
        print("CHUNKING VALIDATION")
        print("="*60)

        # Check sizes
        sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)

        print(f"Total chunks: {len(chunks)}")
        print(f"Avg size: {avg_size:.1f} chars")
        print(f"Min size: {min_size} chars")
        print(f"Max size: {max_size} chars")

        # Check for empty chunks
        empty_chunks = [c for c in chunks if len(c.page_content.strip()) == 0]
        if empty_chunks:
            print(f"⚠️  Found {len(empty_chunks)} empty chunks!")
        else:
            print("✅ No empty chunks")

        # Check metadata
        with_metadata = [c for c in chunks if c.metadata]
        if len(with_metadata) == len(chunks):
            print("✅ All chunks have metadata")
        else:
            missing = len(chunks) - len(with_metadata)
            print(f"⚠️  {missing} chunks missing metadata")


def main():
    """Função principal"""
    print("="*60)
    print("RAG DEBUGGING TOOLS")
    print("="*60)

    # Setup
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = [
        Document(page_content=f"Document {i} content here", metadata={"id": i})
        for i in range(100)
    ]
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create debugger
    debugger = RAGDebugger(vectorstore, embeddings)

    # Run diagnostics
    debugger.inspect_vectorstore()
    debugger.test_embedding_consistency([d.page_content for d in documents])
    debugger.test_similarity("What is RAG?")
    debugger.profile_query("What is RAG?", iterations=5)
    debugger.validate_chunking(documents)

    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)

    print(f"""
📊 Quick Diagnosis:

1. Vector Store: Check count and metadata
2. Embeddings: Verify consistency
3. Similarity: Test search quality
4. Performance: Profile latency
5. Chunking: Validate sizes and metadata

🔧 Common Issues:
   • Low quality → Check chunking
   • Slow speed → Profile queries
   • Inconsistency → Check embeddings
   • Missing results → Verify metadata
""")

    print("="*60)


if __name__ == "__main__":
    main()

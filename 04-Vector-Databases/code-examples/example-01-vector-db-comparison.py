#!/usr/bin/env python3
"""
Example 01: Vector Database Comparison
======================================

Compara diferentes vector databases em funcionalidades e performance.

Uso:
    python example-01-vector-db-comparison.py
"""

import os
import time
from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import (
    Chroma, FAISS, Pinecone, Weaviate
)


def create_sample_data():
    """Criar sample data"""
    texts = [
        "RAG combines retrieval and generation",
        "Embeddings represent text as vectors",
        "Vector databases store and search vectors",
        "Similarity search finds related content",
        "Chunking divides documents into pieces",
        "Document processing loads and cleans data",
        "LLMs generate human-like responses",
        "Retrieval finds relevant information",
        "Generation creates final answer",
        "QA systems answer user questions"
    ]
    return texts


def test_chroma(texts: List[str]):
    """Test Chroma"""
    print(f"\n{'='*60}")
    print("Testing: Chroma (Local)")
    print(f"{'='*60}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create
        start = time.time()
        vectorstore = Chroma.from_texts(texts, embeddings)
        creation_time = time.time() - start

        # Search
        start = time.time()
        docs = vectorstore.similarity_search("RAG", k=3)
        search_time = time.time() - start

        print(f"✅ Created successfully")
        print(f"  Creation time: {creation_time:.3f}s")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Results: {len(docs)}")

        # Cleanup
        vectorstore.delete_collection()

        return {
            "name": "Chroma",
            "creation_time": creation_time,
            "search_time": search_time,
            "success": True
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "name": "Chroma",
            "error": str(e),
            "success": False
        }


def test_faiss(texts: List[str]):
    """Test FAISS"""
    print(f"\n{'='*60}")
    print("Testing: FAISS (Local)")
    print(f"{'='*60}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create
        start = time.time()
        vectorstore = FAISS.from_texts(texts, embeddings)
        creation_time = time.time() - start

        # Search
        start = time.time()
        docs = vectorstore.similarity_search("RAG", k=3)
        search_time = time.time() - start

        print(f"✅ Created successfully")
        print(f"  Creation time: {creation_time:.3f}s")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Results: {len(docs)}")

        return {
            "name": "FAISS",
            "creation_time": creation_time,
            "search_time": search_time,
            "success": True
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "name": "FAISS",
            "error": str(e),
            "success": False
        }


def test_pinecone(texts: List[str]):
    """Test Pinecone"""
    print(f"\n{'='*60}")
    print("Testing: Pinecone (Cloud)")
    print(f"{'='*60}")

    try:
        if not os.getenv("PINECONE_API_KEY"):
            print("⚠️  PINECONE_API_KEY not set, skipping")
            return {"name": "Pinecone", "skipped": True}

        import pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV", "us-west1-gcp")
        )

        # Create index
        index_name = "test-rag-" + str(int(time.time()))
        pinecone.create_index(
            name=index_name,
            dimension=384,
            metric="cosine"
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create
        start = time.time()
        vectorstore = Pinecone.from_texts(
            texts,
            embeddings,
            index_name=index_name
        )
        creation_time = time.time() - start

        # Search
        start = time.time()
        docs = vectorstore.similarity_search("RAG", k=3)
        search_time = time.time() - start

        print(f"✅ Created successfully")
        print(f"  Creation time: {creation_time:.3f}s")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Results: {len(docs)}")

        # Cleanup
        pinecone.delete_index(index_name)

        return {
            "name": "Pinecone",
            "creation_time": creation_time,
            "search_time": search_time,
            "success": True
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "name": "Pinecone",
            "error": str(e),
            "success": False
        }


def test_weaviate(texts: List[str]):
    """Test Weaviate"""
    print(f"\n{'='*60}")
    print("Testing: Weaviate (Cloud/Self-hosted)")
    print(f"{'='*60}")

    try:
        if not os.getenv("WEAVIATE_API_KEY"):
            print("⚠️  WEAVIATE_API_KEY not set, skipping")
            return {"name": "Weaviate", "skipped": True}

        import weaviate
        client = weaviate.Client(
            "https://your-cluster.weaviate.network",
            auth_client_secret=weaviate.AuthApiKey(
                api_key=os.getenv("WEAVIATE_API_KEY")
            )
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create
        start = time.time()
        vectorstore = Weaviate.from_texts(
            texts,
            embeddings,
            client=client
        )
        creation_time = time.time() - start

        # Search
        start = time.time()
        docs = vectorstore.similarity_search("RAG", k=3)
        search_time = time.time() - start

        print(f"✅ Created successfully")
        print(f"  Creation time: {creation_time:.3f}s")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Results: {len(docs)}")

        return {
            "name": "Weaviate",
            "creation_time": creation_time,
            "search_time": search_time,
            "success": True
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "name": "Weaviate",
            "error": str(e),
            "success": False
        }


def main():
    """Função principal"""
    print("="*60)
    print("VECTOR DATABASE COMPARISON")
    print("="*60)

    texts = create_sample_data()

    # Test all databases
    results = []
    results.append(test_chroma(texts))
    results.append(test_faiss(texts))
    results.append(test_pinecone(texts))
    results.append(test_weaviate(texts))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Database':<15} {'Creation':<12} {'Search':<10} {'Status':<10}")
    print("-" * 50)

    for result in results:
        name = result["name"]
        if result.get("skipped"):
            print(f"{name:<15} {'N/A':<12} {'N/A':<10} {'Skipped':<10}")
        elif result.get("success"):
            creation = f"{result['creation_time']:.3f}s"
            search = f"{result['search_time']:.3f}s"
            print(f"{name:<15} {creation:<12} {search:<10} {'✅':<10}")
        else:
            print(f"{name:<15} {'Error':<12} {'Error':<10} {'❌':<10}")

    # Comparison matrix
    print(f"\n{'='*60}")
    print("COMPARISON MATRIX")
    print(f"{'='*60}\n")

    matrix = """
┌────────────┬──────────────┬─────────────┬────────────┬─────────────┐
│ Database   │ Deployment   │ Cost        │ Scale      │ Features    │
├────────────┼──────────────┼─────────────┼────────────┼─────────────┤
│ Chroma     │ Local/Cloud  │ Free        │ Small-Med  │ Basic       │
│ FAISS      │ Local        │ Free        │ Small-Med  │ Fast        │
│ Pinecone   │ Cloud        │ $$$         │ Large-Ent  │ Full        │
│ Weaviate   │ Both         │ Free-$$     │ Med-Large  │ Rich        │
│ Qdrant     │ Both         │ Free-$$     │ Med-Large  │ Fast        │
└────────────┴──────────────┴─────────────┴────────────┴─────────────┘
"""
    print(matrix)

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")

    recommendations = """
🚀 Development/Prototyping:
   • Chroma: Easy local setup, free
   • FAISS: Very fast, library-based

💼 Production (Small-Medium):
   • Weaviate: Good features, flexible
   • Qdrant: High performance, Rust

🏢 Production (Large Scale):
   • Pinecone: Enterprise features, managed
   • Milvus: Scalable, open source

🎓 Research/Academic:
   • FAISS: Custom indexing, fast
   • Chroma: Simple, local-first
"""

    print(recommendations)

    print("="*60)
    print("Comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()

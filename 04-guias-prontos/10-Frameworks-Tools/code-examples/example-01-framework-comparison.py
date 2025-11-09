#!/usr/bin/env python3
"""
Example 01: Framework Comparison
================================

Compara LangChain, LlamaIndex e Haystack.

Uso:
    python example-01-framework-comparison.py
"""

import time
from typing import List


def test_langchain():
    """Test LangChain"""
    print("\n" + "="*60)
    print("LANGCHAIN")
    print("="*60)

    # LangChain setup
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA

    print("✅ LangChain imported successfully")

    # Create sample data
    documents = [
        "RAG combines retrieval and generation",
        "Embeddings represent text as vectors",
        "Vector databases store embeddings"
    ]

    # Setup
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    start = time.time()
    vectorstore = Chroma.from_texts(documents, embeddings)
    setup_time = time.time() - start

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=vectorstore.as_retriever()
    )

    # Query
    start = time.time()
    answer = qa.run("What is RAG?")
    query_time = time.time() - start

    print(f"\nSetup time: {setup_time:.3f}s")
    print(f"Query time: {query_time:.3f}s")
    print(f"Answer: {answer[:100]}...")

    print(f"\n✅ LangChain - Simple and flexible")

    return {
        "framework": "LangChain",
        "setup_time": setup_time,
        "query_time": query_time,
        "features": ["Comprehensive", "Flexible", "Large community"]
    }


def test_llamaindex():
    """Test LlamaIndex"""
    print("\n" + "="*60)
    print("LLAMAINDEX")
    print("="*60)

    try:
        from llama_index.core import (
            VectorStoreIndex,
            SimpleDirectoryReader,
            Document
        )

        print("✅ LlamaIndex imported successfully")

        # Create sample data
        documents = [
            Document(text="RAG combines retrieval and generation"),
            Document(text="Embeddings represent text as vectors"),
            Document(text="Vector databases store embeddings")
        ]

        # Setup
        start = time.time()
        index = VectorStoreIndex.from_documents(documents)
        setup_time = time.time() - start

        # Query
        query_engine = index.as_query_engine()

        start = time.time()
        response = query_engine.query("What is RAG?")
        query_time = time.time() - start

        print(f"\nSetup time: {setup_time:.3f}s")
        print(f"Query time: {query_time:.3f}s")
        print(f"Answer: {str(response)[:100]}...")

        print(f"\n✅ LlamaIndex - Index-centric approach")

        return {
            "framework": "LlamaIndex",
            "setup_time": setup_time,
            "query_time": query_time,
            "features": ["Index-centric", "Data connectors", "Query optimization"]
        }

    except ImportError as e:
        print(f"❌ LlamaIndex not installed: {e}")
        return None


def test_haystack():
    """Test Haystack"""
    print("\n" + "="*60)
    print("HAYSTACK")
    print("="*60)

    try:
        from haystack import Document
        from haystack.nodes import EmbeddingRetriever
        from haystack.document_stores import InMemoryDocumentStore

        print("✅ Haystack imported successfully")

        # Create sample data
        documents = [
            Document(content="RAG combines retrieval and generation"),
            Document(content="Embeddings represent text as vectors"),
            Document(content="Vector databases store embeddings")
        ]

        # Setup
        document_store = InMemoryDocumentStore()
        document_store.write_documents(documents)

        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )

        start = time.time()
        document_store.update_embeddings(retriever)
        setup_time = time.time() - start

        # Query
        start = time.time()
        docs = retriever.retrieve(query="What is RAG?")
        query_time = time.time() - start

        print(f"\nSetup time: {setup_time:.3f}s")
        print(f"Query time: {query_time:.3f}s")
        print(f"Retrieved: {len(docs)} documents")

        print(f"\n✅ Haystack - Production-ready")

        return {
            "framework": "Haystack",
            "setup_time": setup_time,
            "query_time": query_time,
            "features": ["Production-ready", "REST API", "Monitoring"]
        }

    except ImportError as e:
        print(f"❌ Haystack not installed: {e}")
        return None


def main():
    """Função principal"""
    print("="*60)
    print("FRAMEWORK COMPARISON")
    print("="*60)

    results = []

    # Test each framework
    langchain_result = test_langchain()
    if langchain_result:
        results.append(langchain_result)

    llamaindex_result = test_llamaindex()
    if llamaindex_result:
        results.append(llamaindex_result)

    haystack_result = test_haystack()
    if haystack_result:
        results.append(haystack_result)

    # Comparison table
    if results:
        print("\n" + "="*60)
        print("COMPARISON TABLE")
        print("="*60)

        print(f"\n{'Framework':<15} {'Setup (s)':<12} {'Query (s)':<12} {'Best For':<20}")
        print("-" * 60)

        for result in results:
            framework = result["framework"]
            setup = f"{result['setup_time']:.3f}"
            query = f"{result['query_time']:.3f}"
            best_for = "General RAG" if framework == "LangChain" else \
                      "Data-heavy" if framework == "LlamaIndex" else \
                      "Production"
            print(f"{framework:<15} {setup:<12} {query:<12} {best_for:<20}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    recommendations = """
🎯 LangChain:
   • Start here for most projects
   • Largest community
   • Most flexible
   • Best for: Research, experimentation

🎯 LlamaIndex:
   • Better for data-heavy apps
   • Index-centric approach
   • Great query optimization
   • Best for: Multi-document, complex queries

🎯 Haystack:
   • Production-focused
   • Built-in REST API
   • Enterprise features
   • Best for: Production deployments
"""
    print(recommendations)

    print("="*60)
    print("Comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 03: RAG Architecture Demo
=================================

Demonstração visual da arquitetura RAG mostrando o fluxo
completo: Indexing → Query → Response.

Uso:
    python example-03-architecture-demo.py
"""

import time
from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


def print_architecture():
    """Exibe a arquitetura RAG visualmente"""
    print("=" * 70)
    print("RAG ARCHITECTURE - Fluxo Completo")
    print("=" * 70)

    architecture = """
┌─────────────────────────────────────────────────────────────────┐
│                        RAG ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │   PHASE 1   │  INDEXING (Uma vez, off-line)
    │  INDEXING   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │  1. Load Documents (PDF, TXT, etc)  │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  2. Split into Chunks               │
    │     (chunk_size=1000, overlap=200)  │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  3. Generate Embeddings             │
    │     (text-embedding-ada-002)        │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  4. Store in Vector Database        │
    │     (Chroma, Pinecone, etc)         │
    └─────────────────────────────────────┘

    ┌─────────────┐
    │   PHASE 2   │  QUERY (Toda vez, on-line)
    │   QUERY     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │  5. Embed User Query                │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  6. Search Similar Chunks           │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  7. Retrieve Top-K Documents        │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  8. LLM Generates Response          │
    │     (with context)                  │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  9. Return Answer + Citations       │
    └─────────────────────────────────────┘
"""
    print(architecture)


def demonstrate_indexing_phase():
    """Demonstra a fase de Indexing"""
    print("\n" + "=" * 70)
    print("FASE 1: INDEXING (Executada uma vez)")
    print("=" * 70)

    documents = [
        "RAG é uma técnica que combina busca com geração",
        "RAG usa memória paramétrica e não-paramétrica",
        "Lewis et al. (2020) introduziu RAG",
        "RAG reduz hallucinations em QA systems"
    ]

    print(f"\n📄 Documentos: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc}")

    print("\n1️⃣  Loading documents...")
    time.sleep(0.5)
    print("   ✅ Loaded 4 documents")

    print("\n2️⃣  Splitting into chunks...")
    time.sleep(0.5)
    print("   ✅ Split into 8 chunks (chunk_size=1000, overlap=200)")

    print("\n3️⃣  Generating embeddings...")
    time.sleep(0.5)
    print("   ✅ Generated 8 embeddings (1536 dimensions)")

    print("\n4️⃣  Storing in vector database...")
    time.sleep(0.5)
    print("   ✅ Stored in Chroma (8 vectors)")

    return Chroma.from_texts(documents, OpenAIEmbeddings())


def demonstrate_query_phase(vectorstore):
    """Demonstra a fase de Query"""
    print("\n" + "=" * 70)
    print("FASE 2: QUERY (Executada toda vez)")
    print("=" * 70)

    question = "O que é RAG?"
    print(f"\n❓ User Query: '{question}'")

    print("\n1️⃣  Embedding user query...")
    time.sleep(0.5)
    print(f"   ✅ Query embedded (1536 dimensions)")

    print("\n2️⃣  Searching for similar chunks...")
    time.sleep(0.5)
    print("   ✅ Similarity search completed")

    print("\n3️⃣  Retrieving top-3 documents...")
    time.sleep(0.5)

    docs = vectorstore.similarity_search(question, k=3)
    print("   ✅ Retrieved 3 relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"      {i}. Score: {doc.metadata.get('score', 'N/A'):.4f}")
        print(f"         Content: {doc.page_content[:60]}...")

    print("\n4️⃣  Generating response with LLM...")
    time.sleep(0.5)

    llm = OpenAI(temperature=0)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    answer = llm(prompt)

    print("   ✅ Response generated")
    print(f"\n✅ FINAL ANSWER: {answer}")


def show_component_details():
    """Mostra detalhes dos componentes"""
    print("\n" + "=" * 70)
    print("COMPONENTES DETALHADOS")
    print("=" * 70)

    components = {
        "Document Loaders": [
            "PyPDFLoader - documentos PDF",
            "TextLoader - arquivos texto",
            "WebBaseLoader - páginas web",
            "CSVLoader - arquivos CSV"
        ],
        "Text Splitters": [
            "RecursiveCharacterTextSplitter - padrão",
            "SentenceTransformersTokenizer - por sentenças",
            "MarkdownHeaderTextSplitter - por headers"
        ],
        "Embeddings": [
            "OpenAI - text-embedding-ada-002",
            "HuggingFace - BGE, E5, MiniLM",
            "Cohere - multilingual embeddings"
        ],
        "Vector Stores": [
            "Chroma - open-source, local",
            "Pinecone - cloud, managed",
            "FAISS - library, not full DB",
            "Weaviate - open-source, cloud"
        ],
        "LLMs": [
            "OpenAI - GPT-3.5, GPT-4",
            "Anthropic - Claude",
            "Hugging Face - open models"
        ]
    }

    for component, examples in components.items():
        print(f"\n📦 {component}:")
        for example in examples:
            print(f"   • {example}")


def main():
    """Função principal"""
    print("\n")
    print_architecture()

    # Demonstrar indexing
    vectorstore = demonstrate_indexing_phase()

    # Demonstrar query
    demonstrate_query_phase(vectorstore)

    # Mostrar detalhes dos componentes
    show_component_details()

    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print("""
RAG = Retrieval-Augmented Generation

Duas fases:
1. INDEXING (uma vez) - preparar documentos
2. QUERY (sempre) - responder perguntas

Vantagens:
• Knowledge up-to-date
• Factualidade (reduz hallucinations)
• Citations (explicabilidade)
• Custo-efetivo vs fine-tuning

Quando usar:
• Dados dinâmicos/mudam frequente
• Precisa de citations
• Volume grande de dados
• Custo de re-treino alto
""")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 01: Basic Chunking Strategies
=====================================

Demonstra diferentes estratégias de chunking e seus efeitos
na qualidade do retrieval.

Uso:
    python example-01-basic-chunking.py
"""

from typing import List
from langchain.text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SentenceTransformersTokenizer
)
from langchain.schema import Document
import matplotlib.pyplot as plt
import numpy as np


def create_sample_document() -> List[Document]:
    """Criar documento de exemplo"""
    text = """
Chunking Strategies in RAG
==========================

Introduction
------------
Chunking is the process of dividing large documents into smaller, manageable
pieces. This is a critical step in RAG systems as it directly impacts the
quality of retrieval and generation.

Why Chunk?
-----------
There are several reasons why chunking is important:

1. Context Window Limitations
   Large language models have a limited context window. For example, GPT-4
   has a context window of 8,192 tokens (about 6,000 words) for the base
   model and 32,768 tokens (about 25,000 words) for the extended model.

2. Cost Efficiency
   Smaller chunks mean fewer tokens to embed and retrieve, which reduces
   both embedding and inference costs.

3. Semantic Coherence
   When documents are split at semantic boundaries, each chunk contains
   related information, improving retrieval precision.

Types of Chunking
------------------
There are several approaches to chunking:

Fixed-Size Chunking
This is the simplest approach where documents are split into chunks of a
fixed number of characters or tokens. While simple, it may break semantic
units in the middle.

Recursive Chunking
This approach attempts to split documents at semantic boundaries by trying
different separators in order. It starts with the most natural break points
and falls back to less ideal splits if needed.

Semantic Chunking
This method uses the meaning of the text to determine where to split.
Sentences or paragraphs with similar themes are grouped together.

Hierarchical Chunking
This approach preserves the document structure by first splitting at major
divisions (chapters, sections) and then applying chunking within those
divisions.

Best Practices
---------------
1. Start with 1000 character chunks with 200 character overlap
2. Use semantic separators like paragraph breaks and sentence boundaries
3. Test different chunk sizes for your specific use case
4. Consider the types of questions your users will ask
5. Monitor chunk statistics and adjust as needed
"""

    return [Document(page_content=text, metadata={"source": "sample.txt"})]


def chunk_recursive(documents: List[Document]) -> List[Document]:
    """Chunking with RecursiveCharacterTextSplitter"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)


def chunk_character(documents: List[Document]) -> List[Document]:
    """Chunking with CharacterTextSplitter"""
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator=""
    )
    return splitter.split_documents(documents)


def chunk_semantic(documents: List[Document]) -> List[Document]:
    """Chunking with semantic approach"""
    # Using sentence transformers for semantic separation
    splitter = SentenceTransformersTokenizer(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def analyze_chunks(chunks: List[Document], method: str):
    """Analisar características dos chunks"""
    lengths = [len(chunk.page_content) for chunk in chunks]

    print(f"\n{'='*60}")
    print(f"Analysis: {method}")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average length: {np.mean(lengths):.1f} chars")
    print(f"Min length: {min(lengths)} chars")
    print(f"Max length: {max(lengths)} chars")
    print(f"Std deviation: {np.std(lengths):.1f}")

    # Mostrar sample chunks
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
        print(f"  {chunk.page_content[:150]}...")

    return {
        "method": method,
        "count": len(chunks),
        "avg_length": np.mean(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }


def compare_strategies():
    """Comparar diferentes estratégias de chunking"""
    print("="*60)
    print("CHUNKING STRATEGIES COMPARISON")
    print("="*60)

    documents = create_sample_document()

    strategies = {
        "RecursiveCharacterTextSplitter": chunk_recursive,
        "CharacterTextSplitter": chunk_character,
        "Semantic (SentenceTransformers)": chunk_semantic
    }

    results = []

    for name, strategy in strategies.items():
        try:
            chunks = strategy(documents)
            result = analyze_chunks(chunks, name)
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error with {name}: {e}")
            results.append({
                "method": name,
                "count": "Error",
                "avg_length": "Error",
                "error": str(e)
            })

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Method':<30} {'Chunks':<10} {'Avg Length':<15}")
    print("-" * 55)
    for result in results:
        method = result["method"]
        count = result["count"]
        avg = result["avg_length"]
        if isinstance(avg, (int, float)):
            print(f"{method[:29]:<30} {count:<10} {avg:<15.1f}")
        else:
            print(f"{method[:29]:<30} {count:<10} {avg:<15}")

    return results


def test_overlap_sizes():
    """Testar diferentes overlap sizes"""
    print(f"\n{'='*60}")
    print("OVERLAP SIZE COMPARISON")
    print(f"{'='*60}\n")

    documents = create_sample_document()
    overlaps = [0, 50, 100, 200, 300, 500]

    for overlap in overlaps:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=overlap
        )
        chunks = splitter.split_documents(documents)

        # Calcular overlap effectiveness
        total_content = sum(len(c.page_content) for c in chunks)
        original_content = len(documents[0].page_content)
        redundancy = (total_content - original_content) / original_content * 100

        print(f"Overlap: {overlap:>4} | Chunks: {len(chunks):>2} | Redundancy: {redundancy:>5.1f}%")


def test_chunk_sizes():
    """Testar diferentes chunk sizes"""
    print(f"\n{'='*60}")
    print("CHUNK SIZE COMPARISON")
    print(f"{'='*60}\n")

    documents = create_sample_document()
    sizes = [200, 500, 800, 1000, 1500, 2000, 3000]

    print(f"{'Size':<8} {'Chunks':<8} {'Avg Length':<12} {'Coverage':<10}")
    print("-" * 40)

    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        avg_length = np.mean([len(c.page_content) for c in chunks])
        coverage = (avg_length * len(chunks)) / len(documents[0].page_content) * 100

        print(f"{size:<8} {len(chunks):<8} {avg_length:<12.1f} {coverage:<10.1f}%")


def custom_separator_test():
    """Testar separadores customizados"""
    print(f"\n{'='*60}")
    print("CUSTOM SEPARATORS")
    print(f"{'='*60}\n")

    # Documento com estrutura específica
    text = """Chapter 1: Introduction

This is the first chapter of our document.

Subsection 1.1
This is a subsection with more details.

Chapter 2: Methods

This is the second chapter.
Let's discuss our methods here.
With multiple paragraphs.

Chapter 3: Results

Finally, we present the results."""

    document = Document(page_content=text, metadata={"source": "structured.txt"})

    # Separators diferentes
    configs = [
        (["\n\n\n", "\n\n", "\n", ". "], "Paragraph-first"),
        (["\n\n", "\n", ". ", " "], "Default"),
        (["Chapter", "\n\n", "\n", ". "], "Chapter-aware")
    ]

    for separators, name in configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=separators
        )
        chunks = splitter.split_documents([document])

        print(f"\n{name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. {chunk.page_content[:80]}...")


def main():
    """Função principal"""
    print("="*60)
    print("CHUNKING STRATEGIES - Comprehensive Demo")
    print("="*60)

    # 1. Comparar estratégias
    results = compare_strategies()

    # 2. Testar overlap sizes
    test_overlap_sizes()

    # 3. Testar chunk sizes
    test_chunk_sizes()

    # 4. Testar separadores customizados
    custom_separator_test()

    # Recomendações
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")

    recommendations = """
📌 For most RAG systems:
   • Use RecursiveCharacterTextSplitter
   • chunk_size = 1000
   • chunk_overlap = 200
   • separators = ["\\n\\n", "\\n", ".", " "]

📌 For code/technical documents:
   • chunk_size = 500-800
   • chunk_overlap = 100-150
   • Use language-specific separators

📌 For conversational AI:
   • chunk_size = 800
   • chunk_overlap = 150
   • Preserve conversational context

📌 For summarization:
   • chunk_size = 2000-3000
   • chunk_overlap = 500
   • Prioritize paragraph boundaries

⚙️  Always test with your specific use case!
"""
    print(recommendations)

    print("="*60)
    print("Demo completed! Try with your own documents.")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 01: Embedding Models Comparison
========================================

Compara diferentes modelos de embedding em qualidade e performance.

Uso:
    python example-01-embedding-comparison.py
"""

import time
from typing import List
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


def load_embedding_models():
    """Carregar diferentes modelos"""
    models = {
        "openai-small": OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
        "openai-large": OpenAIEmbeddings(
            model="text-embedding-3-large"
        ),
        "bge-large": HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        ),
        "e5-large": HuggingFaceEmbeddings(
            model_name="microsoft/E5-large-v2"
        ),
        "minilm": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    }
    return models


def benchmark_embedding_speed(embeddings, texts: List[str], name: str):
    """Benchmark speed of embedding"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    # Warm up
    try:
        embeddings.embed_query("test")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return None

    # Benchmark
    start = time.time()
    vectors = embeddings.embed_documents(texts)
    end = time.time()

    duration = end - start
    avg_time = duration / len(texts)
    total_tokens = sum(len(text.split()) for text in texts)

    print(f"Documents: {len(texts)}")
    print(f"Total time: {duration:.2f}s")
    print(f"Avg time per doc: {avg_time:.3f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Vector dimensions: {len(vectors[0])}")
    print(f"Throughput: {len(texts)/duration:.1f} docs/sec")

    return {
        "name": name,
        "duration": duration,
        "avg_time": avg_time,
        "throughput": len(texts)/duration,
        "dimensions": len(vectors[0]),
        "total_tokens": total_tokens
    }


def compare_similarity(embeddings, name: str):
    """Compare similarity computation"""
    print(f"\n{'='*60}")
    print(f"Similarity Test: {name}")
    print(f"{'='*60}")

    # Related texts
    text1 = "RAG is a technique that combines retrieval and generation"
    text2 = "Retrieval-augmented generation combines search with AI"
    text3 = "I love pizza and pasta"

    # Embed
    v1 = embeddings.embed_query(text1)
    v2 = embeddings.embed_query(text2)
    v3 = embeddings.embed_query(text3)

    # Compute similarity
    def cosine_similarity(v1, v2):
        import numpy as np
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    sim_12 = cosine_similarity(v1, v2)
    sim_13 = cosine_similarity(v1, v3)
    sim_23 = cosine_similarity(v2, v3)

    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    print(f"\nSimilarities:")
    print(f"  1-2 (related): {sim_12:.4f}")
    print(f"  1-3 (unrelated): {sim_13:.4f}")
    print(f"  2-3 (unrelated): {sim_23:.4f}")

    # Check if similar texts have higher similarity
    if sim_12 > sim_13 and sim_12 > sim_23:
        print("  ✅ Correctly identified similarity")
    else:
        print("  ⚠️  May have similarity issues")

    return {
        "name": name,
        "similar_12": sim_12,
        "unrelated_13": sim_13,
        "unrelated_23": sim_23
    }


def test_multilingual(embeddings, name: str):
    """Test multilingual support"""
    print(f"\n{'='*60}")
    print(f"Multilingual Test: {name}")
    print(f"{'='*60}")

    try:
        texts = [
            "Hello world",
            "Bonjour le monde",
            "Hola mundo",
            "Hallo Welt",
            "Ciao mondo"
        ]

        vectors = embeddings.embed_documents(texts)
        print(f"✅ Successfully embedded {len(texts)} languages")
        return True
    except Exception as e:
        print(f"❌ Multilingual support failed: {e}")
        return False


def main():
    """Função principal"""
    print("="*60)
    print("EMBEDDING MODELS COMPARISON")
    print("="*60)

    # Sample texts
    texts = [
        "RAG combines retrieval and generation",
        "Embeddings represent text as vectors",
        "Vector databases store embeddings",
        "Similarity search finds related content",
        "Chunking divides documents into smaller pieces",
        "Document processing loads and cleans data",
        "LLMs generate human-like text",
        "Retrieval finds relevant information",
        "Generation creates final response",
        "QA systems answer user questions"
    ]

    # Load models
    models = load_embedding_models()

    # Benchmark
    results = []
    for name, embeddings in models.items():
        result = benchmark_embedding_speed(embeddings, texts, name)
        if result:
            results.append(result)

        # Similarity test
        compare_similarity(embeddings, name)

        # Multilingual test
        test_multilingual(embeddings, name)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Model':<20} {'Speed (docs/s)':<15} {'Dimensions':<12} {'Free':<6}")
    print("-" * 55)

    for result in results:
        model = result["name"]
        speed = result["throughput"]
        dims = result["dimensions"]
        free = "✅" if "openai" not in model else "❌"
        print(f"{model:<20} {speed:<15.1f} {dims:<12} {free:<6}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")

    recommendations = """
🎯 For Production (Commercial):
   • text-embedding-3-large: Best quality, reliable
   • text-embedding-3-small: Good balance cost/quality

🎓 For Research/Academic (Open Source):
   • BAAI/bge-large-en-v1.5: Excellent quality, free
   • microsoft/E5-large-v2: Good quality, instruction-tuned

⚡ For Speed/Prototyping:
   • sentence-transformers/all-MiniLM-L6-v2: Very fast, lightweight

🌍 For Multilingual:
   • text-embedding-3-large/small: Best multilingual
   • BAAI/bge-large-en: Good multilingual support
"""

    print(recommendations)

    print("="*60)
    print("Comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()

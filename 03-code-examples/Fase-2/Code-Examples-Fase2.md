# Code Examples: Fase 2 (Seções 03-04)

### Data: 09/11/2025
### Status: Executáveis no Windows
### Foco: Embedding Models + Vector Databases

---

## EXAMPLE 1: Embedding Models Comparison

### Prerequisites

```bash
pip install sentence-transformers torch transformers
```

### Complete Comparison Script

```python
"""
Embedding Models Comparison
Compare multiple models: BGE, E5, MiniLM, MPNet, Jina
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime

class EmbeddingModelBenchmark:
    """Benchmark different embedding models."""

    def __init__(self):
        self.models = {
            'BGE-large': 'BAAI/bge-large-en-v1.5',
            'E5-large': 'intfloat/e5-large-v2',
            'MiniLM': 'all-MiniLM-L6-v2',
            'MPNet': 'all-mpnet-base-v2',
            'Jina-base': 'jinaai/jina-embeddings-v2-base-en'
        }
        self.loaded_models = {}

    def load_model(self, name):
        """Load model with caching."""
        if name not in self.loaded_models:
            print(f"Loading {name}...")
            model_name = self.models[name]
            kwargs = {}
            if name == 'Jina-base':
                kwargs['trust_remote_code'] = True
            self.loaded_models[name] = SentenceTransformer(model_name, **kwargs)
        return self.loaded_models[name]

    def encode_texts(self, texts, model_name, batch_size=32):
        """Encode texts with a model."""
        model = self.load_model(model_name)
        start_time = time.time()

        # Add E5 prefix
        if model_name == 'E5-large':
            texts = ["query: " + text for text in texts]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        elapsed = time.time() - start_time
        return embeddings, elapsed

    def benchmark_models(self, texts):
        """Benchmark all models."""
        results = {}

        for model_name in self.models.keys():
            try:
                print(f"\n{'='*60}")
                print(f"Benchmarking: {model_name}")
                print(f"{'='*60}")

                embeddings, elapsed = self.encode_texts(texts, model_name)

                # Calculate similarity matrix
                sim_matrix = cosine_similarity(embeddings)

                # Calculate metrics
                avg_similarity = np.mean(sim_matrix)
                std_similarity = np.std(sim_matrix)
                throughput = len(texts) / elapsed

                results[model_name] = {
                    'embeddings': embeddings,
                    'elapsed_time': elapsed,
                    'avg_similarity': avg_similarity,
                    'std_similarity': std_similarity,
                    'throughput': throughput,
                    'embedding_dim': embeddings.shape[1],
                    'memory_mb': embeddings.nbytes / (1024 * 1024)
                }

                print(f"✅ Model: {model_name}")
                print(f"   Dimensão: {embeddings.shape[1]}")
                print(f"   Tempo: {elapsed:.2f}s")
                print(f"   Throughput: {throughput:.1f} texts/s")
                print(f"   Memória: {embeddings.nbytes / (1024 * 1024):.1f} MB")
                print(f"   Similaridade média: {avg_similarity:.3f} ± {std_similarity:.3f}")

            except Exception as e:
                print(f"❌ Erro com {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        return results

    def recommend_model(self, requirements):
        """Recommend model based on requirements."""
        recommendations = []

        # Quality priority
        if requirements.get('quality') == 'max':
            recommendations.append(('BGE-large', 'State-of-the-art quality, MIT license'))
            recommendations.append(('MPNet', 'Good quality, Apache-2.0, faster than BGE'))

        # Speed priority
        if requirements.get('speed') == 'max':
            recommendations.append(('MiniLM', 'Ultra-fast, good for prototyping'))
            recommendations.append(('Jina-base', 'Fast with long sequence support'))

        # Multilingual
        if requirements.get('multilingual'):
            recommendations.append(('E5-large', 'English only - need multilingual model'))

        # Production
        if requirements.get('use_case') == 'production':
            recommendations.append(('BGE-large', 'Production recommended'))
            recommendations.append(('MPNet', 'Production alternative'))

        return recommendations

# Usage Example
if __name__ == "__main__":
    # Test texts
    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Python is a popular programming language for data science",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret visual data"
    ]

    # Run benchmark
    benchmark = EmbeddingModelBenchmark()
    results = benchmark.benchmark_models(test_texts)

    # Show summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, data in results.items():
        if 'error' not in data:
            print(f"{name:15} | Dim: {data['embedding_dim']:4} | "
                  f"Time: {data['elapsed_time']:6.2f}s | "
                  f"Throughput: {data['throughput']:6.1f} texts/s | "
                  f"Memory: {data['memory_mb']:6.1f} MB")

    # Get recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    requirements = {'quality': 'max', 'use_case': 'production'}
    recommendations = benchmark.recommend_model(requirements)
    for model, reason in recommendations:
        print(f"✅ {model}: {reason}")
```

---

## EXAMPLE 2: RAG with Multiple Embeddings

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai sentence-transformers
```

### Multi-Model RAG Pipeline

```python
"""
RAG Pipeline com diferentes Embedding Models
Teste a mesma query com modelos diferentes
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

class MultiModelRAG:
    """RAG with different embedding models."""

    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

    def load_and_split(self, file_path):
        """Load and split document."""
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        return self.splitter.split_documents(docs)

    def create_vectorstore(self, splits, model_name):
        """Create vector store with specified embedding model."""
        embedding_models = {
            'BGE': 'BAAI/bge-large-en-v1.5',
            'MiniLM': 'all-MiniLM-L6-v2',
            'MPNet': 'all-mpnet-base-v2',
            'E5': 'intfloat/e5-large-v2',
            'Jina': 'jinaai/jina-embeddings-v2-base-en'
        }

        if model_name not in embedding_models:
            raise ValueError(f"Model {model_name} not supported")

        print(f"Loading {model_name} embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_models[model_name],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = InMemoryVectorStore(embeddings)
        vectorstore.add_documents(splits)

        return vectorstore

    def create_rag_chain(self, vectorstore):
        """Create RAG chain."""
        prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

Context: {context}

Question: {question}

Provide a detailed answer and cite the source.
""")

        chain = (
            {
                "context": vectorstore.as_retriever(search_k=4),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def compare_models(self, file_path, questions):
        """Compare RAG performance with different models."""
        print("Loading and splitting document...")
        splits = self.load_and_split(file_path)
        print(f"Created {len(splits)} chunks\n")

        models = ['BGE', 'MiniLM', 'MPNet']
        results = {}

        for model_name in models:
            print(f"{'='*60}")
            print(f"Testing with {model_name}")
            print(f"{'='*60}")

            try:
                vectorstore = self.create_vectorstore(splits, model_name)
                rag_chain = self.create_rag_chain(vectorstore)

                model_results = []
                for question in questions:
                    print(f"\n❓ Question: {question}")
                    answer = rag_chain.invoke(question)
                    print(f"💡 Answer: {answer[:200]}...")
                    model_results.append({'question': question, 'answer': answer})

                results[model_name] = model_results
                print(f"✅ {model_name} completed\n")

            except Exception as e:
                print(f"❌ Error with {model_name}: {e}\n")
                results[model_name] = {'error': str(e)}

        return results

    def analyze_results(self, results, questions):
        """Analyze results from different models."""
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")

        for i, question in enumerate(questions):
            print(f"\n📝 Question {i+1}: {question}")
            print(f"{'-'*60}")

            for model_name, model_results in results.items():
                if 'error' in model_results:
                    print(f"{model_name:10} | Error: {model_results['error']}")
                else:
                    answer = model_results[i]['answer']
                    # Simple analysis: answer length
                    print(f"{model_name:10} | Length: {len(answer):4} chars | "
                          f"Preview: {answer[:100]}...")

# Usage
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    if OPENAI_API_KEY == "your-api-key-here":
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        exit(1)

    rag_system = MultiModelRAG(OPENAI_API_KEY)

    # Path to your document
    document_path = r"C:\Users\Bushido\Documents\sample_document.txt"

    # Test questions
    questions = [
        "What is the main topic of the document?",
        "What are the key findings?",
        "What conclusions are drawn?"
    ]

    # Run comparison
    results = rag_system.compare_models(document_path, questions)

    # Analyze
    rag_system.analyze_results(results, questions)
```

---

## EXAMPLE 3: Vector Database Comparison

### Prerequisites

```bash
pip install chromadb qdrant-client weaviate-client
```

### Multi-Vector DB Testing

```python
"""
Vector Database Comparison
Test Chroma, Qdrant, Weaviate with same data
"""

import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import weaviate
import numpy as np
from datetime import datetime
import time

class VectorDBComparator:
    """Compare different vector databases."""

    def __init__(self):
        self.collections = {}
        self.results = {}

    def setup_chroma(self, collection_name="test_collection", dim=384):
        """Setup ChromaDB."""
        print("Setting up ChromaDB...")
        client = chromadb.Client()
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.collections['chroma'] = collection
        return collection

    def setup_qdrant(self, collection_name="test_collection", dim=384):
        """Setup Qdrant (requires running instance)."""
        print("Setting up Qdrant...")
        try:
            client = QdrantClient(url="http://localhost:6333")
            # Create collection (idempotent)
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            self.collections['qdrant'] = {'client': client, 'collection': collection_name}
            return self.collections['qdrant']
        except Exception as e:
            print(f"⚠️  Qdrant not running: {e}")
            return None

    def setup_weaviate(self, class_name="TestDocument"):
        """Setup Weaviate (requires running instance)."""
        print("Setting up Weaviate...")
        try:
            client = weaviate.Client("http://localhost:8080")

            # Create schema
            schema = {
                "classes": [{
                    "class": class_name,
                    "properties": [
                        {"name": "text", "dataType": ["text"]},
                        {"name": "source", "dataType": ["string"]}
                    ],
                    "vectorizer": "none"  # We'll add vectors manually
                }]
            }

            client.schema.create(schema)
            self.collections['weaviate'] = {'client': client, 'class_name': class_name}
            return self.collections['weaviate']
        except Exception as e:
            print(f"⚠️  Weaviate not running: {e}")
            return None

    def generate_test_data(self, num_docs=1000, dim=384):
        """Generate test documents and embeddings."""
        print(f"Generating {num_docs} test documents...")

        # Generate texts
        texts = [f"This is document {i} about topic X with content {i * 10}"
                 for i in range(num_docs)]

        # Generate random embeddings
        embeddings = np.random.rand(num_docs, dim).astype(np.float32)

        # Normalize (for cosine similarity)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        metadata = [{"source": f"file_{i}.txt", "id": i} for i in range(num_docs)]

        return texts, embeddings, metadata

    def test_chroma(self, texts, embeddings, metadata, k=10, num_queries=10):
        """Test ChromaDB performance."""
        print("\nTesting ChromaDB...")
        start_time = time.time()

        collection = self.collections['chroma']

        # Add documents
        ids = [f"doc_{i}" for i in range(len(texts))]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadata
        )

        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f}s")

        # Run queries
        query_embeddings = embeddings[:num_queries]
        query_times = []

        for i, query_emb in enumerate(query_embeddings):
            start = time.time()
            results = collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=k
            )
            query_times.append(time.time() - start)

        avg_query_time = np.mean(query_times)
        print(f"   Avg query time (k={k}): {avg_query_time*1000:.2f}ms")
        print(f"   QPS: {1/avg_query_time:.1f}")

        self.results['chroma'] = {
            'load_time': load_time,
            'avg_query_time': avg_query_time,
            'qps': 1/avg_query_time
        }

    def test_qdrant(self, texts, embeddings, metadata, k=10, num_queries=10):
        """Test Qdrant performance."""
        if 'qdrant' not in self.collections:
            return

        print("\nTesting Qdrant...")
        qdrant_config = self.collections['qdrant']
        client = qdrant_config['client']
        collection_name = qdrant_config['collection']

        start_time = time.time()

        # Prepare points
        points = []
        for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadata)):
            points.append({
                "id": i,
                "vector": emb.tolist(),
                "payload": {"text": text, **meta}
            })

        # Upload
        client.upsert(collection_name=collection_name, points=points)

        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f}s")

        # Run queries
        query_embeddings = embeddings[:num_queries]
        query_times = []

        for query_emb in query_embeddings:
            start = time.time()
            results = client.search(
                collection_name=collection_name,
                query_vector=query_emb.tolist(),
                limit=k
            )
            query_times.append(time.time() - start)

        avg_query_time = np.mean(query_times)
        print(f"   Avg query time (k={k}): {avg_query_time*1000:.2f}ms")
        print(f"   QPS: {1/avg_query_time:.1f}")

        self.results['qdrant'] = {
            'load_time': load_time,
            'avg_query_time': avg_query_time,
            'qps': 1/avg_query_time
        }

    def test_weaviate(self, texts, embeddings, metadata, k=10, num_queries=10):
        """Test Weaviate performance."""
        if 'weaviate' not in self.collections:
            return

        print("\nTesting Weaviate...")
        weaviate_config = self.collections['weaviate']
        client = weaviate_config['client']
        class_name = weaviate_config['class_name']

        start_time = time.time()

        # Add data
        for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadata)):
            client.data_object.create(
                data_object={"text": text, **meta},
                class_name=class_name,
                vector=emb.tolist()
            )

        load_time = time.time() - start_time
        print(f"   Load time: {load_time:.2f}s")

        # Run queries
        query_embeddings = embeddings[:num_queries]
        query_times = []

        for query_emb in query_embeddings:
            start = time.time()
            results = client.query.get(
                class_name,
                ["text", "source"]
            ).with_near_vector({
                "vector": query_emb.tolist()
            }).with_limit(k).do()
            query_times.append(time.time() - start)

        avg_query_time = np.mean(query_times)
        print(f"   Avg query time (k={k}): {avg_query_time*1000:.2f}ms")
        print(f"   QPS: {1/avg_query_time:.1f}")

        self.results['weaviate'] = {
            'load_time': load_time,
            'avg_query_time': avg_query_time,
            'qps': 1/avg_query_time
        }

    def run_comparison(self, num_docs=1000, dim=384, k=10):
        """Run full comparison."""
        print("="*60)
        print("VECTOR DATABASE COMPARISON")
        print("="*60)
        print(f"Documents: {num_docs}, Dimension: {dim}, k={k}\n")

        # Setup databases
        self.setup_chroma(dim=dim)
        self.setup_qdrant(dim=dim)
        self.setup_weaviate()

        # Generate test data
        texts, embeddings, metadata = self.generate_test_data(num_docs, dim)

        # Test each database
        if 'chroma' in self.collections:
            self.test_chroma(texts, embeddings, metadata, k)

        if 'qdrant' in self.collections:
            self.test_qdrant(texts, embeddings, metadata, k)

        if 'weaviate' in self.collections:
            self.test_weaviate(texts, embeddings, metadata, k)

        # Show summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for db_name, results in self.results.items():
            print(f"\n{db_name.upper()}:")
            print(f"  Load time:  {results['load_time']:.2f}s")
            print(f"  Query time: {results['avg_query_time']*1000:.2f}ms")
            print(f"  QPS:        {results['qps']:.1f}")

# Usage
if __name__ == "__main__":
    # Note: Requires running Qdrant and Weaviate services
    # Run with: docker run -p 6333:6333 qdrant/qdrant
    #          docker run -p 8080:8080 semitechnologies/weaviate

    comparator = VectorDBComparator()
    comparator.run_comparison(num_docs=1000, dim=384, k=10)
```

---

## EXAMPLE 4: RAG Production with Pinecone

### Prerequisites

```bash
pip install langchain langchain-pinecone pinecone-client openai
```

### Production RAG with Pinecone

```python
"""
Production RAG with Pinecone
Cloud-based vector database with auto-scaling
"""

import os
import pinecone
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ProductionRAG:
    """Production RAG with Pinecone."""

    def __init__(self, openai_api_key, pinecone_api_key, environment, index_name):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.environment = environment
        self.index_name = index_name

        # Initialize Pinecone
        pinecone.init(
            api_key=pinecone_api_key,
            environment=environment
        )

        # Check if index exists, if not create it
        if index_name not in pinecone.list_indexes():
            print(f"Creating index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
            print("⏳ Waiting for index to be ready...")
            import time
            time.sleep(30)  # Wait for index creation

        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            temperature=0
        )

        # Create vector store
        self.vectorstore = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )

    def index_document(self, file_path, metadata_filter=None):
        """Index a document to Pinecone."""
        print(f"📄 Indexing document: {file_path}")

        # Load and split
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = splitter.split_documents(docs)

        # Add metadata
        for i, split in enumerate(splits):
            split.metadata.update({
                'chunk_id': i,
                'source_file': os.path.basename(file_path),
                'indexed_at': datetime.now().isoformat()
            })
            if metadata_filter:
                split.metadata.update(metadata_filter)

        # Upload to Pinecone
        print(f"   Uploading {len(splits)} chunks to Pinecone...")
        self.vectorstore.add_documents(splits)
        print(f"   ✅ Indexed successfully")

        return len(splits)

    def create_rag_chain(self):
        """Create RAG chain with Pinecone."""
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the context to answer the question.

Context: {context}

Question: {question}

Instructions:
- Answer based only on the context
- If the context doesn't contain the answer, say so
- Cite the source file in your answer

Answer:""")

        chain = (
            {
                "context": self.vectorstore.as_retriever(
                    search_k=5,
                    filter=metadata_filter if 'metadata_filter' in locals() else None
                ),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def ask_question(self, question, filter_dict=None):
        """Ask a question with optional metadata filter."""
        print(f"❓ Question: {question}")
        if filter_dict:
            print(f"   Filter: {filter_dict}")

        # Temporarily set filter if provided
        if filter_dict:
            retriever = self.vectorstore.as_retriever(
                search_k=5,
                filter=filter_dict
            )
        else:
            retriever = self.vectorstore.as_retriever(search_k=5)

        # Create chain with retriever
        prompt = ChatPromptTemplate.from_template("""
Answer based on context:

{context}

Question: {question}
""")

        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)
        print(f"💡 Answer: {answer}")
        print("-" * 80)

        return answer

    def get_stats(self):
        """Get Pinecone index statistics."""
        index = pinecone.Index(self.index_name)
        stats = index.describe_index_stats()
        return stats

# Usage
if __name__ == "__main__":
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = "us-west1-gcp"  # Your environment
    INDEX_NAME = "production-rag"

    if not all([OPENAI_API_KEY, PINECONE_API_KEY]):
        print("⚠️  Please set OPENAI_API_KEY and PINECONE_API_KEY")
        exit(1)

    # Initialize
    rag = ProductionRAG(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
        index_name=INDEX_NAME
    )

    # Index document
    document_path = r"C:\Users\Bushido\Documents\production_doc.txt"
    chunks = rag.index_document(document_path, {"category": "research"})

    # Create RAG chain
    print("\n🔗 Creating RAG chain...")
    # Chain will be created in ask_question

    # Ask questions
    questions = [
        "What is the main topic?",
        "What are the key findings?",
        "What methodology was used?"
    ]

    for question in questions:
        rag.ask_question(question)

    # Get stats
    print("\n📊 Index Statistics:")
    stats = rag.get_stats()
    print(f"   Total vector count: {stats.total_vector_count}")
    print(f"   Dimension: {stats.dimension}")

    print(f"\n✅ Production RAG with Pinecone is ready!")
```

---

## EXAMPLE 5: Batch Embedding Processing

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai sentence-transformers pandas tqdm
```

### Batch Processing Pipeline

```python
"""
Batch Embedding Processing
Process large datasets efficiently with batching and caching
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

class BatchEmbeddingProcessor:
    """Process large documents in batches with caching."""

    def __init__(self, model_name="BAAI/bge-large-en-v1.5", cache_dir="./embeddings_cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        print(f"Loading model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_cache_path(self, text_hash):
        """Get cache file path for a text hash."""
        return self.cache_dir / f"{text_hash}.npy"

    def get_text_hash(self, text):
        """Generate hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def load_from_cache(self, text):
        """Load embedding from cache if exists."""
        text_hash = self.get_text_hash(text)
        cache_path = self.get_cache_path(text_hash)

        if cache_path.exists():
            return np.load(cache_path)
        return None

    def save_to_cache(self, text, embedding):
        """Save embedding to cache."""
        text_hash = self.get_text_hash(text)
        cache_path = self.get_cache_path(text_hash)
        np.save(cache_path, embedding)

    def encode_batch(self, texts, batch_size=100, use_cache=True):
        """Encode texts in batches with optional caching."""
        print(f"Encoding {len(texts)} texts...")
        print(f"Batch size: {batch_size}, Model: {self.model_name}")
        print(f"Cache: {'Enabled' if use_cache else 'Disabled'}\n")

        all_embeddings = []
        cached_count = 0
        processed_count = 0

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch:
                # Try cache first
                if use_cache:
                    cached_emb = self.load_from_cache(text)
                    if cached_emb is not None:
                        batch_embeddings.append(cached_emb)
                        cached_count += 1
                        continue

                # Encode
                emb = self.embeddings.embed_query(text)
                batch_embeddings.append(emb)

                # Cache it
                if use_cache:
                    self.save_to_cache(text, emb)

                processed_count += 1

            all_embeddings.extend(batch_embeddings)

        print(f"\n✅ Encoding complete!")
        print(f"   Cached: {cached_count} texts")
        print(f"   Processed: {processed_count} texts")
        print(f"   Total: {len(texts)} texts")
        print(f"   Embedding dimension: {len(all_embeddings[0])}")

        return np.array(all_embeddings)

    def process_directory(self, directory_path, output_file, batch_size=100):
        """Process all text files in a directory."""
        directory = Path(directory_path)
        all_texts = []

        print(f"Scanning directory: {directory}")
        text_files = list(directory.glob("*.txt"))

        if not text_files:
            print(f"⚠️  No .txt files found in {directory}")
            return

        print(f"Found {len(text_files)} files\n")

        # Load all texts
        for file_path in tqdm(text_files, desc="Loading files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_texts.append({
                        'file': str(file_path),
                        'text': text,
                        'word_count': len(text.split())
                    })
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")

        # Extract texts for encoding
        texts = [item['text'] for item in all_texts]

        # Encode
        embeddings = self.encode_batch(texts, batch_size=batch_size)

        # Combine results
        results = []
        for i, item in enumerate(all_texts):
            results.append({
                'file': item['file'],
                'text': item['text'][:100] + "...",  # Truncate for storage
                'word_count': item['word_count'],
                'embedding': embeddings[i].tolist()
            })

        # Save to file
        output_path = Path(output_file)
        print(f"\n💾 Saving results to {output_path}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': self.model_name,
                'total_files': len(results),
                'embedding_dim': len(embeddings[0]),
                'processed_at': datetime.now().isoformat(),
                'files': results
            }, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(results)} embeddings to {output_path}")

        # Save embeddings separately (for faster loading)
        embeddings_file = output_path.with_suffix('.npy')
        print(f"💾 Saving embeddings array to {embeddings_file}...")
        np.save(embeddings_file, embeddings)
        print(f"✅ Saved embeddings array")

        return results

    def load_embeddings(self, embeddings_file):
        """Load pre-saved embeddings."""
        embeddings_file = Path(embeddings_file)
        if embeddings_file.suffix == '.npy':
            return np.load(embeddings_file)
        else:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                embeddings = [item['embedding'] for item in data['files']]
                return np.array(embeddings)

    def search_similar(self, query, embeddings_file, top_k=5):
        """Search for similar texts."""
        embeddings = self.load_embeddings(embeddings_file)

        # Encode query
        query_emb = self.embeddings.embed_query(query)

        # Calculate similarities
        similarities = np.dot(embeddings, query_emb)

        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        print(f"\n🔍 Query: {query}")
        print(f"Top {top_k} similar texts:")
        print("-" * 80)

        for i, idx in enumerate(top_indices):
            print(f"\n{i+1}. Similarity: {similarities[idx]:.3f}")
            print(f"   Text: {self.truncate_text(embeddings_file, idx)}")

        return top_indices

    def truncate_text(self, embeddings_file, idx):
        """Helper to truncate text from saved file."""
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data['files'][idx]['text']
            return text[:200] + "..."

# Usage
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "all-MiniLM-L6-v2"  # Fast model for demo
    DIRECTORY = r"C:\Users\Bushido\Documents\texts"
    OUTPUT_FILE = r"C:\Users\Bushido\Documents\embeddings_output.json"

    # Initialize processor
    processor = BatchEmbeddingProcessor(model_name=MODEL_NAME)

    # Process directory
    results = processor.process_directory(
        directory_path=DIRECTORY,
        output_file=OUTPUT_FILE,
        batch_size=50
    )

    # Example search
    query = "artificial intelligence"
    processor.search_similar(query, OUTPUT_FILE, top_k=3)
```

---

## USAGE INSTRUCTIONS

### Prerequisites Installation

```powershell
# Core dependencies
pip install langchain langchain-community langchain-openai
pip install sentence-transformers torch transformers

# Vector databases (choose what you need)
pip install chromadb qdrant-client weaviate-client pinecone-client

# For batch processing
pip install pandas tqdm numpy

# For Windows-specific
# Install Docker Desktop for running vector databases
```

### Running the Examples

#### Example 1: Embedding Model Comparison
```bash
python example1_embedding_comparison.py
```

#### Example 2: Multi-Model RAG
```bash
# Set OpenAI API key
$env:OPENAI_API_KEY = "your-api-key-here"

python example2_multi_rag.py
```

#### Example 3: Vector DB Comparison
```bash
# Start services first (in separate terminals)
# Qdrant: docker run -p 6333:6333 qdrant/qdrant
# Weaviate: docker run -p 8080:8080 semitechnologies/weaviate

python example3_vector_db_comparison.py
```

#### Example 4: Production RAG with Pinecone
```bash
# Set API keys
$env:OPENAI_API_KEY = "your-openai-key"
$env:PINECONE_API_KEY = "your-pinecone-key"

python example4_pinecone_rag.py
```

#### Example 5: Batch Processing
```bash
python example5_batch_processing.py
```

### Windows-Specific Notes

1. **Docker Services**: Use Docker Desktop to run vector databases
2. **Paths**: Always use raw strings `r"C:\path"` or forward slashes
3. **Environment Variables**: Use `$env:VAR_NAME = "value"` in PowerShell
4. **Memory**: Close other applications when running large batch processing
5. **CUDA**: Install PyTorch with CUDA for GPU acceleration

### PowerShell Setup Script

```powershell
# Create a setup.ps1 file
@"
# Install required packages
pip install langchain langchain-community langchain-openai
pip install sentence-transformers torch transformers
pip install chromadb qdrant-client weaviate-client pinecone-client
pip install pandas tqdm numpy

# Start vector database services
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name weaviate -p 8080:8080 semitechnologies/weaviate

Write-Host "Setup complete!" -ForegroundColor Green
"@ | Out-File -FilePath "setup.ps1" -Encoding UTF8

# Run it
.\setup.ps1
```

### Common Issues

1. **Import Errors**:
   ```bash
   pip install --upgrade pip
   pip install --upgrade langchain
   ```

2. **CUDA Errors**:
   ```python
   # Force CPU usage
   embeddings = HuggingFaceEmbeddings(
       model_name="...",
       model_kwargs={'device': 'cpu'}
   )
   ```

3. **Memory Errors**:
   ```python
   # Reduce batch size
   batch_size = 10  # instead of 100
   ```

4. **Permission Errors**:
   ```powershell
   # Run PowerShell as Administrator
   # Or use: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Next Steps

1. **Experiment** with different embedding models
2. **Compare** vector databases for your use case
3. **Scale** to production with Pinecone or Weaviate Cloud
4. **Optimize** batch processing for large datasets
5. **Deploy** to cloud with proper monitoring

---

**Status**: ✅ Code examples Fase 2 created
**Próximo**: Resumo Executivo Fase 2
**Data Conclusão**: 09/11/2025

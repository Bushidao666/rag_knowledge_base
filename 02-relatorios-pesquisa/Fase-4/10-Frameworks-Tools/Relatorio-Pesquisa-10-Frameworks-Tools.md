# Relatório de Pesquisa: Seção 10 - Frameworks & Tools

### Data: 09/11/2025
### Status: Fase 4 - Advanced Topics

---

## 1. RESUMO EXECUTIVO

Frameworks e Tools são essenciais para desenvolvimento eficiente de sistemas RAG. A escolha correta pode acelerar development 10x e simplificar production deployment.

**Insights Chave:**
- **LangChain**: Comprehensive, mature, 100+ integrations
- **LlamaIndex**: Index-centric, query-focused, data connectors
- **Haystack**: Production-ready, REST API, NLP-focused
- **txtai**: Lightweight, semantic search
- **Vespa**: Big data, real-time, hybrid search
- **ChromaDB**: Developer-friendly, embedding-native

---

## 2. LANGCHAIN

### 2.1 Overview

**LangChain** é o framework mais popular para LLM applications, com strong RAG support.

**Versão Atual**: 0.1+ (LTS)

**Características:**
- Chain-based architecture
- 100+ integrations
- Multiple programming languages (Python, JavaScript, Go)
- Large community
- Comprehensive documentation

### 2.2 Componentes Principais

**Document Loaders:**
```python
from langchain_community.document_loaders import (
    WebBaseLoader, TextLoader, PyMuPDFLoader,
    Docx2txtLoader, CSVLoader
)

# Web
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# PDF
loader = PyMuPDFLoader("document.pdf")
docs = loader.load()

# Word
loader = Docx2txtLoader("document.docx")
docs = loader.load()
```

**Text Splitters:**
```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersSplit,
    MarkdownHeaderTextSplitter
)

# Recursive (recommended)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Semantic
splitter = SentenceTransformersSplit()

# Markdown-aware
splitter = MarkdownHeaderTextSplitter()
```

**Embedding Models:**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)
```

**Vector Stores:**
```python
from langchain_chroma import Chroma
from langchain_pinecone import Pinecone
from langchain_weaviate import Weaviate

# Chroma
vectorstore = Chroma.from_documents(docs, embeddings)

# Pinecone
vectorstore = Pinecone.from_documents(
    docs, embeddings,
    index_name="my-index"
)

# Weaviate
vectorstore = Weaviate.from_documents(docs, embeddings)
```

**Retrievers:**
```python
from langchain_core.retrievers import VectorStoreRetriever

# Basic
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# MMR (Maximal Marginal Relevance)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)
```

**RAG Chains:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context:

Context: {context}

Question: {question}
""")

# Create chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Use chain
response = chain.invoke("What is AI?")
```

### 2.3 RAG Agents

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun

# Define tools
tools = [DuckDuckGoSearchRun()]

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Use
result = agent_executor.invoke({
    "input": "What is the weather in Paris?"
})
```

### 2.4 LangSmith (Evaluation & Tracing)

```python
import os
from langsmith import Client

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# Create dataset
client = Client()
dataset = client.create_dataset(
    name="rag-eval",
    description="RAG evaluation dataset"
)

# Add examples
client.create_examples(
    inputs=[
        {"question": "What is AI?"},
        {"question": "How does RAG work?"}
    ],
    outputs=[
        {"answer": "AI is..."},
        {"answer": "RAG works by..."}
    ],
    dataset_id=dataset.id
)
```

### 2.5 Vantagens

- ✅ Comprehensive ecosystem
- ✅ Large community
- ✅ Excellent documentation
- ✅ Production-ready features
- ✅ Multi-language support
- ✅ LangSmith integration

### 2.6 Desvantagens

- ❌ Can be overkill for simple use cases
- ❌ Steeper learning curve
- ❌ Performance overhead
- ❌ Frequent breaking changes

---

## 3. LLAMAINDEX

### 3.1 Overview

**LlamaIndex** é index-centric framework focado em data ingestion e query optimization.

**Versão Atual**: 0.10+

**Características:**
- Index-centric design
- Multiple index types
- Data connectors ecosystem
- Query engine abstraction
- Modular architecture

### 3.2 Index Types

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.extractors import BaseExtractor
from llama_index.text_splitter import SentenceSplitter

# Create index
doc = Document(text="Your text here")
index = VectorStoreIndex.from_documents([doc])

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
```

**Index Types:**

**1. VectorStoreIndex** (mais popular)
```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
```

**2. ListIndex** (for lists)
```python
from llama_index.core import ListIndex

index = ListIndex(nodes)
```

**3. TreeIndex** (hierarchical)
```python
from llama_index.core import TreeIndex

index = TreeIndex(nodes)
```

**4. KeywordTableIndex** (keyword-based)
```python
from llama_index.core import KeywordTableIndex

index = KeywordTableIndex(nodes)
```

**5. KGIndex** (knowledge graph)
```python
from llama_index.core import KnowledgeGraphIndex

index = KnowledgeGraphIndex(nodes)
```

### 3.3 Query Engines

```python
from llama_index.core.response import ResponseMode
from llama_index.core.types import RESPONSE_TYPE

# Basic query engine
query_engine = index.as_query_engine()

# Custom response mode
query_engine = index.as_query_engine(
    response_mode=ResponseMode.COMPACT
)

# With retriever
from llama_index.core.retrievers import VectorIndexRetriever

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3
)

query_engine = RetrieverQueryEngine(retriever)
```

### 3.4 Data Connectors (LlamaHub)

```python
from llama_index.core import download_loader

# Web
WebPageReader = download_loader("WebPageReader")
loader = WebPageReader()
docs = loader.load_data(url="https://example.com")

# Database
SimpleMongoDBReader = download_loader("SimpleMongoDBReader")
loader = SimpleMongoDBReader(
    host="localhost",
    port=27017,
    dbname="mydb",
    collection_name="mycollection"
)
docs = loader.load_data()
```

### 3.5 Vantagens

- ✅ Index-centric approach
- ✅ Multiple index types
- ✅ Data connectors ecosystem
- ✅ Good for data-heavy apps
- ✅ Query optimization
- ✅ Modular

### 3.6 Desvantagens

- ❌ Smaller community than LangChain
- ❌ Less integrations
- ❌ Different mental model
- ❌ Documentation gaps

---

## 4. HAYSTACK

### 4.1 Overview

**Haystack** é framework production-ready focado em NLP, com REST API built-in.

**Versão Atual**: 2.0+

**Características:**
- Production-focused
- REST API
- NLP background
- Component-based
- Scalable architecture

### 4.2 Components

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator

# Create pipeline
pipeline = Pipeline()

# Add components
pipeline.add_component("doc_embedder", SentenceTransformersDocumentEmbedder())
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", InMemoryEmbeddingRetriever())
pipeline.add_component("generator", OpenAIGenerator())

# Connect
pipeline.connect("doc_embedder", "retriever")
pipeline.connect("text_embedder", "retriever")
pipeline.connect("retriever", "generator")

# Run
result = pipeline.run({
    "doc_embedder": {"documents": documents},
    "text_embedder": {"text": "What is AI?"}
})
```

### 4.3 REST API

```python
# Start server
from haystack import __version__
from haystack.server import HaystackServer

server = HaystackServer()
server.start()

# REST API available at http://localhost:11434
# Endpoints:
# /search, /query, /embed, /eval
```

### 4.4 Vantagens

- ✅ Production-ready
- ✅ REST API
- ✅ Scalable
- ✅ NLP-focused
- ✅ Good documentation

### 4.5 Desvantagens

- ❌ Python-focused
- ❌ Less flexible than LangChain
- ❌ Smaller ecosystem
- ❌ Steeper learning curve

---

## 5. TXT.AI

### 5.1 Overview

**txtai** é lightweight semantic search engine.

**Características:**
- Lightweight
- Simple API
- Multiple backends (SQLite, Weaviate, etc.)
- Semantic search
- Extensible

### 5.2 Implementation

```python
from txtai import Embedding

# Create embedding instance
embeddings = Embedding(
    path="path/to/index"
)

# Add documents
embeddings.add([
    (1, "AI is artificial intelligence", ""),
    (2, "ML is machine learning", ""),
    (3, "DL is deep learning", "")
])

# Search
embeddings.index()

# Query
results = embeddings.search("What is AI?", 3)
# Returns: [(1, 0.89), ...]
```

### 5.3 Vantagens

- ✅ Lightweight
- ✅ Simple API
- ✅ Multiple backends
- ✅ Fast development
- ✅ Good for simple use cases

### 5.4 Desvantagens

- ❌ Limited features
- ❌ Less ecosystem
- ❌ Not for complex use cases
- ❌ Fewer integrations

---

## 6. VESPA

### 6.1 Overview

**Vespa** é big data serving engine com hybrid search.

**Características:**
- Big data scale
- Real-time
- Hybrid search
- Structured and unstructured data
- Low latency

### 6.2 Implementation

```python
from vespa.application import Vespa
from vespa.io import VespaResponse

# Connect to Vespa
app = Vespa(url="http://localhost", port=8080)

# Query
response = app.query(
    yql="select * from sources * where userQuery()",
    query="What is AI?",
    hits=5
)

# Add documents
response = app.feed_data_point(
    data_id="1",
    fields={
        "title": "AI Article",
        "content": "AI is...",
        "embedding": [0.1, 0.2, ...]
    }
)
```

### 6.3 Vantagens

- ✅ Big data scale
- ✅ Real-time
- ✅ Hybrid search
- ✅ Structured + unstructured
- ✅ Low latency

### 6.4 Desvantagens

- ❌ Complex setup
- ❌ Java-focused
- ❌ Steeper learning curve
- ❌ Overkill for small apps

---

## 7. CHROMADB

### 7.1 Overview

**ChromaDB** é embedding-native vector database.

**Características:**
- Developer-friendly
- Embedding-native
- Simple API
- Python-first
- Local-first

### 7.2 Implementation

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# Add documents
collection.add(
    ids=["1", "2", "3"],
    documents=["Doc 1", "Doc 2", "Doc 3"],
    metadatas=[{"source": "book"}, {"source": "web"}, {"source": "paper"}]
)

# Query
results = collection.query(
    query_texts=["What is AI?"],
    n_results=3
)
```

### 7.3 Vantagens

- ✅ Developer-friendly
- ✅ Simple API
- ✅ Embedding-native
- ✅ Good for prototyping
- ✅ Python-first

### 7.4 Desvantagens

- ❌ Not for large scale
- ❌ Limited features
- ❌ Basic querying
- ❌ Less ecosystem

---

## 8. COMPARISON MATRIX

| Framework | Ease of Use | Flexibility | Scalability | Ecosystem | Production Ready | Best For |
|-----------|-------------|-------------|-------------|-----------|----------------|----------|
| **LangChain** | 🟡 | ✅✅✅ | 🟡 | ✅✅✅ | ✅✅ | General use |
| **LlamaIndex** | 🟡 | ✅✅ | ✅ | ✅✅ | ✅ | Data-heavy |
| **Haystack** | 🟡 | 🟡 | ✅✅ | ✅ | ✅✅ | Production |
| **txtai** | ✅✅ | 🟡 | 🟡 | 🟡 | 🟡 | Simple apps |
| **Vespa** | 🟡 | ✅ | ✅✅ | 🟡 | ✅✅ | Big data |
| **ChromaDB** | ✅✅ | 🟡 | 🟡 | 🟡 | 🟡 | Prototyping |

### Framework Selection Guide

**Choose LangChain if:**
- Need comprehensive features
- Large community support
- Multiple integrations
- Complex workflows

**Choose LlamaIndex if:**
- Data-heavy applications
- Multiple index types
- Query optimization focus
- Custom data connectors

**Choose Haystack if:**
- Production deployment
- REST API needed
- NLP-focused
- Scalability important

**Choose txtai if:**
- Simple use case
- Quick development
- Lightweight
- Python apps

**Choose Vespa if:**
- Big data scale
- Real-time requirements
- Hybrid search
- Enterprise scale

**Choose ChromaDB if:**
- Prototyping
- Local development
- Simple API
- Embedding-native

---

## 9. NEW FRAMEWORKS (2024-2025)

### 9.1 Emerging Frameworks

**1. Phidata**
- Build, deploy and monitor AI applications
- Python-first
- Focus on production

**2. Autogen**
- Microsoft framework
- Multi-agent conversations
- Code execution

**3. Graphlit**
- Graph-based RAG
- Knowledge graphs
- Relationship understanding

**4. Kotaemon**
- Japanese RAG framework
- Multi-modal support
- Local-first

### 9.2 Specialized Tools

**Evaluation:**
- Ragas (RAG-specific)
- TruLens (comprehensive)
- DeepEval (testing)
- LangSmith (tracing)

**Vector DBs:**
- Qdrant (Rust-based)
- Weaviate (AI-native)
- Milvus (GenAI-focused)
- Pinecone (managed)

---

## 10. FRAMEWORK-SPECIFIC PATTERNS

### 10.1 LangChain Patterns

**Chain of Thought:**
```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
```

**Retrieval-Augmented Generation:**
```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

**Agent with Tools:**
```python
agent = create_react_agent(llm, tools, prompt)
```

### 10.2 LlamaIndex Patterns

**Custom Query Engine:**
```python
from llama_index.core.query_engine import CustomQueryEngine

class MyQueryEngine(CustomQueryEngine):
    def custom_query(self, query):
        # Custom logic
        return Response()

query_engine = MyQueryEngine()
```

**Reranking:**
```python
from llama_index.core.postprocessor import SentenceRerank

postprocessor = SentenceRerank(top_k=3)
```

### 10.3 Haystack Patterns

**Pipeline with multiple retrievers:**
```python
pipeline = Pipeline()
pipeline.add_component("bm25", BM25Retriever())
pipeline.add_component("embedding", EmbeddingRetriever())
pipeline.add_component("ranker", Ranker())

pipeline.connect("bm25", "ranker")
pipeline.connect("embedding", "ranker")
```

---

## 11. INTEGRATION EXAMPLES

### 11.1 LangChain + Pinecone + OpenAI

```python
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo")
vectorstore = Pinecone.from_existing_index("my-index", embeddings)

# RAG
prompt = ChatPromptTemplate.from_template(...)
chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = chain.invoke("What is AI?")
```

### 11.2 LlamaIndex + Weaviate + Cohere

```python
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.embeddings import CohereEmbedding

# Setup
embedder = CohereEmbedding(cohere_api_key="...", model_name="...")
vector_store = WeaviateVectorStore(weaviate_client=client, index_name="Documents")
index = VectorStoreIndex.from_documents(docs, embedder=embedder, vector_store=vector_store)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is AI?")
```

### 11.3 Haystack + FAISS + Transformers

```python
from haystack.components.embedders import TransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.generators import TransformersGenerator

# Setup
embedder = TransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = InMemoryEmbeddingRetriever()
generator = TransformersGenerator(model="microsoft/DialoGPT-medium")

# Pipeline
pipeline = Pipeline()
pipeline.add_component("embedder", embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("generator", generator)

pipeline.connect("embedder", "retriever")
pipeline.connect("retriever", "generator")
```

---

## 12. BEST PRACTICES

### 12.1 Framework Selection

1. **Start with LangChain** para general use
2. **Use LlamaIndex** para data-heavy apps
3. **Choose Haystack** para production REST API
4. **Pick txtai/Chroma** para simple/prototyping
5. **Use Vespa** para big data

### 12.2 Development Tips

1. **Start simple**, add complexity
2. **Use templates/boilerplates**
3. **Leverage ecosystem**
4. **Test thoroughly**
5. **Monitor performance**

### 12.3 Production Deployment

1. **Containerize** applications
2. **Use managed services** (Pinecone, Weaviate Cloud)
3. **Set up monitoring** (LangSmith, custom)
4. **Version control** prompts and configs
5. **A/B test** different approaches

---

## 13. TRENDING 2024-2025

### 13.1 Features

- **Agentic RAG** integration
- **Multimodal** support
- **Streaming** responses
- **Caching** layers
- **Observability** tools

### 13.2 Performance

- **Faster inference**
- **Lower latency**
- **Better scaling**
- **Cost optimization**
- **Efficient batching**

### 13.3 Developer Experience

- **Better DX**
- **IDE plugins**
- **Debugging tools**
- **Visualization**
- **Type safety**

---

## 14. RESEARCH GAPS

### 14.1 To Research
- [ ] Framework benchmarks
- [ ] Performance comparisons
- [ ] Developer experience studies
- [ ] Production case studies
- [ ] Cost analysis
- [ ] Integration patterns

### 14.2 Future Directions
- [ ] Unified frameworks
- [ ] Auto-optimization
- [ ] Self-healing systems
- [ ] Domain-specific frameworks
- [ ] Low-code/no-code
- [ ] AI-assisted development

---

**Status**: ✅ Base para Frameworks & Tools coletada
**Próximo**: Seção 11 - Production Deployment
**Data Conclusão**: 09/11/2025

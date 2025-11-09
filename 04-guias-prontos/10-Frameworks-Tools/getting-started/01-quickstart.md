# Quick Start: Frameworks & Tools

**Tempo estimado:** 15-30 minutos
**Nível:** Iniciante
**Pré-requisitos:** Conhecimentos básicos de RAG

## Objetivo
Selecionar e usar frameworks para RAG

## Frameworks Populares

### 1. LangChain
Mais popular e abrangente:
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# RAG completo em poucas linhas
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)
vectorstore = Chroma.from_documents(chunks, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

answer = qa.run("O que é RAG?")
```

### 2. LlamaIndex
Index-centric, bom para data-heavy:
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("O que é RAG?")
```

### 3. Haystack
Production-ready, NLP-focused:
```python
from haystack import Document
from haystack.nodes import EmbeddingRetriever, FARMReader

# Indexing
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
retriever.retrieve(query="RAG?")

# Querying
reader = FARMReader("deepset/roberta-base-squad2")
answer = reader.predict(query="O que é RAG?", documents=docs)
```

## Comparison

| Framework | Strengths | Weaknesses | Best For |
|-----------|-----------|------------|----------|
| **LangChain** | Comprehensive, flexible, large community | Complex, steep learning curve | General RAG, research |
| **LlamaIndex** | Index-centric, data connectors, query optimization | Smaller community, less flexible | Data-heavy apps |
| **Haystack** | Production-ready, REST API, monitoring | Less flexible, NLP-focused | Enterprise, production |

## Ecosystem Tools

### Vector Databases
- **Chroma** - Simple, local
- **Pinecone** - Cloud, managed
- **Weaviate** - Open source, cloud
- **FAISS** - Library, research

### Embeddings
- **OpenAI** - Commercial, high quality
- **HuggingFace** - Open source models
- **Cohere** - Commercial, multilingual

### Evaluation
- **RAGAS** - RAG-specific metrics
- **Trulens** - Production monitoring
- **LangSmith** - Tracing and evaluation

## Selection Guide

### Use LangChain when:
- ✅ Building complex RAG systems
- ✅ Need flexibility
- ✅ Research/experimentation
- ✅ Large community support

### Use LlamaIndex when:
- ✅ Index-centric approach
- ✅ Multiple data sources
- ✅ Query optimization needed
- ✅ Complex data connectors

### Use Haystack when:
- ✅ Production deployment
- ✅ REST API needed
- ✅ NLP-focused
- ✅ Enterprise features

## Quick Comparison

```python
# LangChain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Simple RAG
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever()
)

# LlamaIndex
from llama_index.core import VectorStoreIndex

# Index-centric
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Haystack
from haystack.pipelines import Pipeline

# Pipeline-based
pipeline = Pipeline()
pipeline.add_node("retriever", retriever)
pipeline.add_node("reader", reader)
result = pipeline.run(query="RAG?")
```

## Which to Choose?

### For Beginners
**Start with LangChain**
- Largest community
- Best documentation
- Most examples

### For Data-Heavy Apps
**Consider LlamaIndex**
- Better data connectors
- Index optimization
- Query engines

### For Production
**Look at Haystack**
- REST API built-in
- Production monitoring
- Enterprise features

## Learning Path

1. **Start** with LangChain
2. **Understand** core concepts
3. **Try** LlamaIndex for data-heavy
4. **Explore** Haystack for production
5. **Combine** tools as needed

## Próximos Passos

- 💻 **Code Examples:** [Comparações](../code-examples/)
- 🔧 **Troubleshooting:** [Frameworks Issues](../troubleshooting/common-issues.md)
- 📊 **Evaluation:** [Guia 06 - Evaluation](../06-Evaluation-Benchmarks/README.md)

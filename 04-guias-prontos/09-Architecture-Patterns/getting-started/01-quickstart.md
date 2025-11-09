# Quick Start: Architecture Patterns

**Tempo estimado:** 15-30 minutos
**Nível:** Avançado
**Pré-requisitos:** Experiência com RAG

## Objetivo
Conhecer padrões arquiteturais para sistemas RAG

## Padrões Principais

### 1. Naive RAG
Mais simples, direto ao ponto:
```python
# Query → Embed → Search → Generate
query = "What is RAG?"
query_embedding = embeddings.embed_query(query)
docs = vectorstore.similarity_search(query_embedding, k=3)
context = "\n".join([doc.page_content for doc in docs])
answer = llm(f"Context: {context}\nQuestion: {query}")
```

### 2. Chunk-Join RAG
Preserva contexto completo:
```python
class ChunkJoinRAG:
    def query(self, question):
        # Get chunks
        chunks = self.retriever.get_relevant_chunks(question)

        # Join all chunks
        context = self.join_chunks(chunks)

        # Generate with full context
        answer = self.generate(question, context)
        return answer
```

### 3. Parent-Document RAG
Hierárquico, parent-child:
```python
class ParentDocumentRAG:
    def query(self, question):
        # Get relevant chunks
        chunks = self.vectorstore.similarity_search(question, k=5)

        # Get parent documents
        parents = [chunk.metadata["parent_id"] for chunk in chunks]

        # Use parent docs as context
        context = self.get_parent_content(parents)

        return self.generate(question, context)
```

### 4. Routing RAG
Multi-index, roteamento inteligente:
```python
class RoutingRAG:
    def __init__(self):
        self.indices = {
            "technical": technical_vectorstore,
            "legal": legal_vectorstore,
            "general": general_vectorstore
        }

    def query(self, question):
        # Route to appropriate index
        index = self.route(question)

        # Query routed index
        return index.query(question)
```

### 5. Agentic RAG
Agentes com multi-step reasoning:
```python
from langchain.agents import create_agent

class AgenticRAG:
    def __init__(self, tools):
        self.agent = create_agent(llm, tools)

    def query(self, question):
        return self.agent.run(question)
```

### 6. Fusion RAG
Multi-query fusion:
```python
class FusionRAG:
    def query(self, question):
        # Generate multiple queries
        queries = self.expand_query(question)

        # Retrieve from all
        results = []
        for q in queries:
            docs = self.search(q)
            results.append(docs)

        # Fusion results
        fused_docs = self.fuse_results(results)

        return self.generate(question, fused_docs)
```

## Comparison

| Pattern | Complexity | Quality | Speed | Use Case |
|---------|------------|---------|-------|----------|
| **Naive** | Low | Medium | Fast | Simple Q&A |
| **Chunk-Join** | Medium | High | Medium | Full context |
| **Parent-Doc** | Medium | High | Medium | Hierarchical docs |
| **Routing** | High | Very High | Medium | Multi-domain |
| **Agentic** | Very High | Very High | Slow | Complex reasoning |
| **Fusion** | High | Very High | Slow | Comprehensive |

## When to Use

- **Naive:** Start here, simple use cases
- **Chunk-Join:** Need full document context
- **Parent-Doc:** Structured documents
- **Routing:** Multiple knowledge bases
- **Agentic:** Complex multi-step tasks
- **Fusion:** Maximum recall, comprehensive

## Example: Architecture Selection

```python
def select_architecture(use_case):
    if use_case == "simple_qa":
        return NaiveRAG()
    elif use_case == "full_context":
        return ChunkJoinRAG()
    elif use_case == "multi_domain":
        return RoutingRAG()
    elif use_case == "complex_reasoning":
        return AgenticRAG()
    else:
        return NaiveRAG()  # Default
```

## Design Considerations

### Scalability
- Horizontal scaling
- Load balancing
- Caching layers

### Monitoring
- Latency tracking
- Quality metrics
- Error rates

### Maintenance
- Versioning
- A/B testing
- Rollback capability

## Próximos Passos

- 💻 **Code Examples:** [Padrões Avançados](../code-examples/)
- 🚀 **Production:** [Guia 11 - Production Deployment](../11-Production-Deployment/README.md)

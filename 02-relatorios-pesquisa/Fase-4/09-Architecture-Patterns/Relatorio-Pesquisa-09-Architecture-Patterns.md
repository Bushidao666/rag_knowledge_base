# Relatório de Pesquisa: Seção 09 - Architecture Patterns

### Data: 09/11/2025
### Status: Fase 4 - Advanced Topics

---

## 1. RESUMO EXECUTIVO

Architecture Patterns definem como organizar e estruturar sistemas RAG para diferentes requisitos, domínios e constraints. Escolher o padrão correto impacta performance, qualidade, custo e manutenibilidade.

**Insights Chave:**
- **Naive RAG**: Simples, baseline para quick start
- **Chunk-Join RAG**: Melhor contexto com less chunks
- **Parent-Document RAG**: Hierarchical retrieval
- **Routing RAG**: Specialized retrieval per query type
- **Agentic RAG**: Multi-step reasoning
- **Citation RAG**: Full traceability
- **Modular RAG**: Composable, testable

---

## 2. NAIVE RAG

### 2.1 Overview

**Naive RAG** é o padrão mais simples: chunk documents, embed, retrieve, generate.

**Arquitetura:**
```
[Document] → [Chunk] → [Embed] → [Store] → [Retrieve] → [Generate]
```

### 2.2 Implementation

```python
class NaiveRAG:
    def __init__(self, embedding_model, llm, vectorstore):
        self.embedding_model = embedding_model
        self.llm = llm
        self.vectorstore = vectorstore

    def index(self, documents):
        """Index documents."""
        # Chunk
        chunks = self.chunk_documents(documents)

        # Embed
        embeddings = self.embedding_model.encode([chunk.text for chunk in chunks])

        # Store
        for chunk, embedding in zip(chunks, embeddings):
            self.vectorstore.add(
                id=chunk.id,
                vector=embedding,
                text=chunk.text,
                metadata=chunk.metadata
            )

    def query(self, question):
        """Answer question."""
        # Retrieve
        relevant_chunks = self.vectorstore.similarity_search(question, k=5)

        # Generate
        context = "\n".join([chunk.text for chunk in relevant_chunks])
        response = self.llm.generate(f"Context: {context}\nQuestion: {question}")

        return response

    def chunk_documents(self, documents):
        """Split documents into chunks."""
        # Simple character-based chunking
        chunks = []
        for doc in documents:
            text = doc.text
            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunks.append(Chunk(
                    id=f"{doc.id}_{i}",
                    text=chunk,
                    metadata=doc.metadata
                ))
        return chunks
```

### 2.3 Pros e Cons

**Pros:**
- ✅ Simple to implement
- ✅ Fast retrieval
- ✅ Good baseline
- ✅ Easy to understand

**Cons:**
- ❌ Limited context (small chunks)
- ❌ May lose document structure
- ❌ No hierarchical understanding
- ❌ May retrieve irrelevant chunks

### 2.4 When to Use

**Use Naive RAG when:**
- ✅ Quick prototyping
- ✅ Small documents
- ✅ Simple questions
- ✅ Limited resources
- ✅ First implementation

**Don't use when:**
- ❌ Large documents
- ❌ Complex questions
- ❌ Need document context
- ❌ Structured data

---

## 3. CHUNK-JOIN RAG

### 3.1 Overview

**Chunk-Join RAG** retrieves multiple related chunks e joins them para context completo.

**Arquitetura:**
```
[Document] → [Chunk] → [Embed] → [Store]
                     ↓
[Query] → [Retrieve] → [Join Chunks] → [Generate]
```

### 3.2 Implementation

```python
class ChunkJoinRAG:
    def __init__(self, embedding_model, llm, vectorstore):
        self.embedding_model = embedding_model
        self.llm = llm
        self.vectorstore = vectorstore
        self.chunk_size = 2000
        self.chunk_overlap = 200

    def index(self, documents):
        """Index documents with parent relationships."""
        for doc in documents:
            # Create chunks
            chunks = self.create_chunks(doc)

            # Store with parent info
            for chunk in chunks:
                self.vectorstore.add(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        'parent_id': doc.id,
                        'chunk_index': chunk.index
                    }
                )

    def query(self, question):
        """Answer question with joined chunks."""
        # Retrieve initial chunks
        initial_chunks = self.vectorstore.similarity_search(question, k=10)

        # Group by parent
        parent_groups = {}
        for chunk in initial_chunks:
            parent_id = chunk.metadata['parent_id']
            if parent_id not in parent_groups:
                parent_groups[parent_id] = []
            parent_groups[parent_id].append(chunk)

        # Join chunks from same parent
        joined_chunks = []
        for parent_id, chunks in parent_groups.items():
            # Sort by chunk index
            chunks.sort(key=lambda x: x.metadata['chunk_index'])

            # Join consecutive chunks
            for i in range(0, len(chunks), 2):  # Join 2 chunks at a time
                if i + 1 < len(chunks):
                    joined = chunks[i].text + "\n" + chunks[i+1].text
                    joined_chunks.append(joined)
                else:
                    joined_chunks.append(chunks[i].text)

        # Re-score joined chunks
        scored_chunks = self.rescore_chunks(question, joined_chunks)

        # Generate
        context = "\n\n".join(scored_chunks[:3])  # Top 3
        response = self.llm.generate(f"Context: {context}\nQuestion: {question}")

        return response

    def rescore_chunks(self, question, chunks):
        """Re-score joined chunks."""
        # Re-embed joined chunks
        embeddings = self.embedding_model.encode(chunks)
        question_embedding = self.embedding_model.encode([question])[0]

        # Calculate similarity
        scores = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            score = cosine_similarity([embedding], [question_embedding])[0][0]
            scores.append((chunk, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scores]
```

### 3.3 Pros e Cons

**Pros:**
- ✅ Better context preservation
- ✅ Reduces fragmentation
- ✅ More coherent answers
- ✅ Preserves document flow

**Cons:**
- ❌ More complex
- ❌ Slower retrieval
- ❌ More compute for re-scoring
- ❌ May include irrelevant info

### 3.4 When to Use

**Use Chunk-Join when:**
- ✅ Large documents
- ✅ Sequential information
- ✅ Need document flow
- ✅ Quality > speed

---

## 4. PARENT-DOCUMENT RAG

### 4.1 Overview

**Parent-Document RAG** maintains hierarchy: chunks → parent document.

**Arquitetura:**
```
[Document] → [Chunk] → [Parent Mapping]
                     ↓
[Query] → [Retrieve Chunks] → [Retrieve Parent] → [Generate]
```

### 4.2 Implementation

```python
class ParentDocumentRAG:
    def __init__(self, embedding_model, llm, vectorstore):
        self.embedding_model = embedding_model
        self.llm = llm
        self.vectorstore = vectorstore
        self.docstore = {}  # Store parent documents

    def index(self, documents):
        """Index documents with parent tracking."""
        for doc in documents:
            # Store full document
            self.docstore[doc.id] = doc

            # Create chunks
            chunks = self.create_chunks(doc)

            # Store chunks with parent reference
            for chunk in chunks:
                self.vectorstore.add(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        'parent_id': doc.id
                    }
                )

    def query(self, question):
        """Answer using parent documents."""
        # Retrieve relevant chunks
        relevant_chunks = self.vectorstore.similarity_search(question, k=10)

        # Get parent documents
        parent_ids = set(
            chunk.metadata['parent_id'] for chunk in relevant_chunks
        )

        # Retrieve full parent documents
        parent_docs = [
            self.docstore[parent_id] for parent_id in parent_ids
        ]

        # Score parent documents by chunk relevance
        scored_docs = self.score_parents(question, parent_docs, relevant_chunks)

        # Use top documents
        top_docs = scored_docs[:3]
        context = "\n\n".join([doc.text for doc in top_docs])

        # Generate
        response = self.llm.generate(f"Context: {context}\nQuestion: {question}")

        return response

    def score_parents(self, question, parent_docs, relevant_chunks):
        """Score parent documents by chunk relevance."""
        # Count chunks per parent
        chunk_counts = {}
        for chunk in relevant_chunks:
            parent_id = chunk.metadata['parent_id']
            chunk_counts[parent_id] = chunk_counts.get(parent_id, 0) + 1

        # Score by chunk count
        scored = []
        for doc in parent_docs:
            score = chunk_counts.get(doc.id, 0)
            scored.append((doc, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored]
```

### 4.3 Pros e Cons

**Pros:**
- ✅ Full document context
- ✅ Better understanding
- ✅ No chunking artifacts
- ✅ Natural for large documents

**Cons:**
- ❌ Larger context
- ❌ More expensive generation
- ❌ May include irrelevant info
- ❌ Trade-off precision vs recall

### 4.4 When to Use

**Use Parent-Document when:**
- ✅ Large documents
- ✅ Need full context
- ✅ Document-level understanding
- ✅ Budget allows larger context

---

## 5. ROUTING RAG

### 5.1 Overview

**Routing RAG** uses different retrievers based on query type.

**Arquitetura:**
```
[Query] → [Route] → [Specialized Retriever] → [Generate]
```

### 5.2 Implementation

```python
class RoutingRAG:
    def __init__(self, retrievers, router, llm):
        self.retrievers = retrievers  # {'qa': qa_retriever, 'search': search_retriever, etc.}
        self.router = router
        self.llm = llm

    def query(self, question):
        """Route query to appropriate retriever."""
        # Determine query type
        query_type = self.router.classify(question)

        # Get appropriate retriever
        retriever = self.retrievers.get(query_type, self.retrievers['default'])

        # Retrieve
        relevant_docs = retriever.search(question, k=5)

        # Generate
        context = "\n".join([doc.text for doc in relevant_docs])
        response = self.llm.generate(f"Context: {context}\nQuestion: {question}")

        return response

class QueryRouter:
    def classify(self, question):
        """Classify query type."""
        # Rule-based
        if any(keyword in question.lower() for keyword in ['what', 'when', 'where', 'who']):
            return 'factual'
        elif any(keyword in question.lower() for keyword in ['how', 'why']):
            return 'explanatory'
        elif any(keyword in question.lower() for keyword in ['compare', 'difference']):
            return 'comparative'
        elif any(keyword in question.lower() for keyword in ['code', 'function', 'class']):
            return 'code'
        else:
            return 'general'

    # Or use LLM-based classification
    def classify_llm(self, question, llm):
        """Classify using LLM."""
        prompt = f"""
        Classify the following question into one of: factual, explanatory, comparative, code, general

        Question: {question}

        Type:
        """
        response = llm.generate(prompt)
        return response.strip().lower()
```

### 5.3 Specialized Retrievers

```python
# Different retrievers for different query types
retrievers = {
    'factual': FactRetriever(
        vectorstore=vectorstore,
        use_bm25=True,  # Exact matching
        use_embeddings=False
    ),
    'explanatory': SemanticRetriever(
        vectorstore=vectorstore,
        use_embeddings=True
    ),
    'comparative': ComparativeRetriever(
        vectorstore=vectorstore,
        method='contrastive'
    ),
    'code': CodeRetriever(
        ast_index=ast_index,  # Parse code
        use_function_names=True
    ),
    'general': HybridRetriever(
        vectorstore=vectorstore,
        use_both=True
    )
}
```

### 5.4 Pros e Cons

**Pros:**
- ✅ Optimized per query type
- ✅ Better quality
- ✅ Specialized handling
- ✅ Efficient resource usage

**Cons:**
- ❌ More complex
- ❌ Need query classifier
- ❌ Multiple retrievers to maintain
- ❌ Classification errors

### 5.5 When to Use

**Use Routing when:**
- ✅ Mixed query types
- ✅ Different data sources
- ✅ Specialized domains
- ✅ Performance requirements

---

## 6. AGENTIC RAG

### 6.1 Overview

**Agentic RAG** uses autonomous agents para multi-step reasoning.

**Arquitetura:**
```
[Query] → [Agent] → [Plan] → [Tools] → [Collect] → [Synthesize] → [Response]
```

### 6.2 Implementation

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool

class AgenticRAG:
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
        self.agent = self.create_agent()

    def create_agent(self):
        """Create ReAct agent."""
        prompt = PromptTemplate.from_template("""
        You are an agent that can use tools to answer questions.

        Question: {input}
        Thought: {agent_scratchpad}
        """)

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )

    def query(self, question):
        """Answer question using agent."""
        result = self.agent.invoke({
            "input": question
        })

        return result['output']

# Define tools
tools = [
    Tool(
        name="search",
        description="Search for information",
        func=search_tool.run
    ),
    Tool(
        name="retrieve",
        description="Retrieve documents",
        func=retriever.search
    ),
    Tool(
        name="calculator",
        description="Perform calculations",
        func=calculator.calculate
    )
]
```

### 6.3 Pros e Cons

**Pros:**
- ✅ Multi-step reasoning
- ✅ Tool usage
- ✅ Flexible
- ✅ Powerful

**Cons:**
- ❌ Complex
- ❌ Unpredictable
- ❌ Expensive
- ❌ Hard to debug

### 6.4 When to Use

**Use Agentic when:**
- ✅ Complex questions
- ✅ Multiple tools needed
- ✅ Research tasks
- ✅ Budget allows

---

## 7. CITATION RAG

### 7.1 Overview

**Citation RAG** provides full traceability de sources.

**Arquitetura:**
```
[Query] → [Retrieve] → [Generate] → [Add Citations] → [Response with Sources]
```

### 7.2 Implementation

```python
class CitationRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query(self, question):
        """Answer with full citations."""
        # Retrieve
        relevant_docs = self.retriever.search(question, k=5)

        # Create context with citations
        citations = []
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            source_id = doc.metadata.get('source', 'Unknown')
            line_ref = doc.metadata.get('line_number', 'N/A')

            citations.append({
                'id': i,
                'source': source_id,
                'line': line_ref,
                'text': doc.text
            })

            context += f"[{i}] {doc.text}\n"

        # Generate with citation instruction
        prompt = f"""
        Context: {context}

        Question: {question}

        Provide an answer and cite your sources using [1], [2], etc. format.
        If information is not in the context, say so.
        """

        response = self.llm.generate(prompt)

        return {
            'answer': response,
            'citations': citations
        }

    def format_citation(self, answer, citations):
        """Format answer with numbered citations."""
        formatted = answer

        # Add reference list
        formatted += "\n\n**Sources:**\n"
        for citation in citations:
            formatted += f"[{citation['id']}] {citation['source']} (line {citation['line']})\n"
            formatted += f"   {citation['text'][:200]}...\n"

        return formatted
```

### 7.3 Pros e Cons

**Pros:**
- ✅ Full traceability
- ✅ Trustworthy
- ✅ Verifiable
- ✅ Academic standard

**Cons:**
- ❌ More verbose
- ❌ Complex formatting
- ❌ May overwhelm user
- ❌ Slower generation

### 7.4 When to Use

**Use Citation when:**
- ✅ Academic/research
- ✅ Legal documents
- ✅ Medical info
- ✅ Trust critical

---

## 8. MODULAR RAG

### 8.1 Overview

**Modular RAG** uses composable components para flexibility.

**Arquitetura:**
```
[Retriever] + [Reranker] + [Generator] = RAG Pipeline
    ↓           ↓             ↓
 Configurable  Configurable   Configurable
```

### 8.2 Implementation

```python
from dataclasses import dataclass
from typing import List, Any

@dataclass
class RAGConfig:
    retriever_type: str
    retriever_params: dict
    reranker_type: str
    reranker_params: dict
    generator_type: str
    generator_params: dict

class ModularRAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = self.create_retriever()
        self.reranker = self.create_reranker()
        self.generator = self.create_generator()

    def create_retriever(self):
        """Factory pattern para retrievers."""
        if self.config.retriever_type == 'dense':
            return DenseRetriever(**self.config.retriever_params)
        elif self.config.retriever_type == 'sparse':
            return SparseRetriever(**self.config.retriever_params)
        elif self.config.retriever_type == 'hybrid':
            return HybridRetriever(**self.config.retriever_params)

    def create_reranker(self):
        """Factory pattern para rerankers."""
        if self.config.reranker_type == 'cross_encoder':
            return CrossEncoderReranker(**self.config.reranker_params)
        elif self.config.reranker_type == 'colbert':
            return ColBERTReranker(**self.config.reranker_params)
        else:
            return None

    def create_generator(self):
        """Factory pattern para generators."""
        if self.config.generator_type == 'llm':
            return LLMGenerator(**self.config.generator_params)
        elif self.config.generator_type == 'template':
            return TemplateGenerator(**self.config.generator_params)

    def query(self, question):
        """Execute pipeline."""
        # Retrieve
        docs = self.retriever.search(question, k=20)

        # Rerank
        if self.reranker:
            docs = self.reranker.rerank(question, docs, k=5)
        else:
            docs = docs[:5]

        # Generate
        context = "\n".join([doc.text for doc in docs])
        response = self.generator.generate(question, context)

        return response

# Usage
config = RAGConfig(
    retriever_type='hybrid',
    retriever_params={'alpha': 0.7, 'k': 20},
    reranker_type='cross_encoder',
    reranker_params={'model': 'BAAI/bge-reranker-base', 'k': 5},
    generator_type='llm',
    generator_params={'model': 'gpt-3.5-turbo', 'temperature': 0}
)

rag = ModularRAG(config)
response = rag.query("What is AI?")
```

### 8.3 Pros e Cons

**Pros:**
- ✅ Highly configurable
- ✅ Easy to test
- ✅ Composable
- ✅ Maintainable

**Cons:**
- ❌ More code
- ❌ Configuration complexity
- ❌ Need component management
- ❌ Testing overhead

### 8.4 When to Use

**Use Modular when:**
- ✅ Production systems
- ✅ Need A/B testing
- ✅ Multiple configurations
- ✅ Team collaboration

---

## 9. COMPARISON TABLE

| Pattern | Complexity | Quality | Speed | Context | Best For |
|---------|------------|---------|-------|---------|----------|
| **Naive** | Low | Medium | High | Small | Quick start |
| **Chunk-Join** | Medium | High | Medium | Medium | Large docs |
| **Parent-Doc** | Medium | High | Medium | Large | Full context |
| **Routing** | High | High | High | Variable | Mixed types |
| **Agentic** | Very High | Very High | Low | Variable | Complex tasks |
| **Citation** | Medium | High | Medium | Medium | Trust/trace |
| **Modular** | High | Variable | Variable | Variable | Production |

---

## 10. DECISION TREE

```
REQUIREMENTS
├─ Need quick start?
│   └─ SIM → Naive RAG
│
├─ Large documents, need context?
│   ├─ Simple → Chunk-Join
│   └─ Full context → Parent-Document
│
├─ Mixed query types?
│   └─ SIM → Routing RAG
│
├─ Complex reasoning needed?
│   └─ SIM → Agentic RAG
│
├─ Trust/traceability critical?
│   └─ SIM → Citation RAG
│
└─ Production, need flexibility?
    └─ SIM → Modular RAG
```

---

## 11. COMBINING PATTERNS

### 11.1 Parent-Document + Citation
```python
class ParentCitationRAG(ParentDocumentRAG, CitationRAG):
    """Combines parent document with citations."""
    pass
```

### 11.2 Routing + Modular
```python
class RoutingModularRAG(RoutingRAG, ModularRAG):
    """Combines routing with modular components."""
    pass
```

### 11.3 Chunk-Join + Self-RAG
```python
class ChunkJoinSelfRAG(ChunkJoinRAG):
    """Combines chunk-join with self-reflection."""
    pass
```

---

## 12. IMPLEMENTATION GUIDELINES

### 12.1 Start Simple
1. Start with Naive RAG
2. Evaluate quality
3. Identify issues
4. Upgrade pattern

### 12.2 Pattern Evolution
```
Naive → Chunk-Join → Parent-Document → Modular
          ↓
       Routing ←→ Agentic
          ↓
    Citation + Self-RAG
```

### 12.3 Evaluation
- Quality metrics
- Latency measurements
- User satisfaction
- Cost analysis

### 12.4 Best Practices
- Keep it simple initially
- Upgrade incrementally
- Monitor performance
- Document changes
- A/B test patterns

---

## 13. RESEARCH GAPS

### 13.1 To Research
- [ ] Pattern effectiveness benchmarks
- [ ] Pattern selection algorithms
- [ ] Hybrid pattern combinations
- [ ] Pattern-specific metrics
- [ ] Cost-quality tradeoffs
- [ ] User experience studies

### 13.2 Future Directions
- [ ] Automatic pattern selection
- [ ] Self-optimizing RAG
- [ ] Pattern composition
- [ ] Domain-specific patterns
- [ ] Real-time pattern switching
- [ ] Pattern evolution learning

---

**Status**: ✅ Base para Architecture Patterns coletada
**Próximo**: Seção 10 - Frameworks & Tools
**Data Conclusão**: 09/11/2025

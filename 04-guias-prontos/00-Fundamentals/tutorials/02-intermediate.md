# Tutorial Intermediário: Comparando Abordagens RAG

**Tempo estimado:** 1-2 horas
**Nível:** Intermediário
**Pré-requisitos:** Conhecimentos básicos de Python, entender RAG básico

## Objetivo
Comparar as diferentes abordagens de implementação RAG e aprender quando usar cada uma.

## Agenda
1. RAG Chains vs RAG Agents
2. Implementação com LangChain
3. Implementação com LlamaIndex
4. Hybrid Search
5. Evaluation básico

## 1. RAG Chains vs RAG Agents

### 1.1 RAG Chains (Uma Chamada)

**Características:**
- Uma única chamada LLM
- Menor latência
- Mais barato
- Sempre busca contexto

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Context sempre injetado
template = """
Use o contexto abaixo para responder à pergunta:

Contexto: {context}

Pergunta: {question}

Resposta:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    prompt=prompt
)

# Uma única chamada LLM
result = qa.run({"query": "O que é RAG?"})
```

**Quando usar:**
- Queries frequentes similares
- Latência é crítica
- Custo deve ser baixo
- Simplicidade é importante

### 1.2 RAG Agents (Multi-Step)

**Características:**
- Múltiplas chamadas LLM
- Mais flexível
- Busca quando necessário
- Maior latência

```python
from langchain.agents import tool, create_agent
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Busca informações relevantes para responder uma pergunta."""
    docs = vectorstore.similarity_search(query, k=2)
    context = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in docs
    )
    return context, docs

model = OpenAI(temperature=0)
agent = create_agent(
    model,
    tools=[retrieve_context],
    system_prompt="Use as ferramentas para ajudar a responder perguntas."
)

# Busca apenas quando necessário, múltiplas buscas
result = agent.run("Explique RAG e dê 3 exemplos")
```

**Quando usar:**
- Queries complexas
- Precisa de múltiplas buscas
- Lógica condicional
- Latência não é crítica

## 2. Implementação com LangChain

### 2.1 Estrutura Completa

```python
import os
from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

class RAGLangChain:
    def __init__(self, data_path: str):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.data_path = data_path
        self.vectorstore = None
        self.qa = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )

    def build_index(self):
        """Criar índice a partir de documentos"""
        # 1. Load documents
        loader = PyPDFLoader(self.data_path)
        docs = loader.load()

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)

        # 3. Create vector store
        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings
        )

    def create_qa_chain(self, use_memory: bool = False):
        """Criar QA chain com ou sem memória"""
        if not self.vectorstore:
            raise ValueError("Build index first")

        if use_memory:
            # Conversational RAG
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                memory=self.memory,
                retriever=self.vectorstore.as_retriever()
            )
        else:
            # Simple RAG
            template = """
Use o contexto para responder. Cite as fontes.

Contexto: {context}

Pergunta: {question}

Resposta (com citações):"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_k=4
                ),
                prompt=prompt,
                return_source_documents=True
            )

    def query(self, question: str, chat_history: List[Dict] = None):
        """Fazer pergunta"""
        if not self.qa:
            raise ValueError("Create QA chain first")

        if isinstance(self.qa, ConversationalRetrievalChain):
            # With memory
            result = self.qa({
                "question": question,
                "chat_history": chat_history or []
            })
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", [])
            }
        else:
            # Simple
            result = self.qa({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", [])
            }

# Usage
rag = RAGLangChain("documento.pdf")
rag.build_index()
rag.create_qa_chain(use_memory=True)

chat_history = []
while True:
    question = input("\nPergunta (ou 'quit'): ")
    if question.lower() == 'quit':
        break

    result = rag.query(question, chat_history)
    print(f"\nResposta: {result['answer']}")

    if result['source_documents']:
        print("\nFontes:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"{i}. {doc.page_content[:100]}...")

    chat_history.append((question, result['answer']))
```

## 3. Implementação com LlamaIndex

### 3.1 Pipeline Index-Centric

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer
)
from llama_index.core.extractors import BaseExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.openai import OpenAI

class RAGLlamaIndex:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.llm = OpenAI(temperature=0)
        self.index = None
        self.query_engine = None

    def build_index(self):
        """Criar índice"""
        # 1. Load documents
        documents = SimpleDirectoryReader(self.data_dir).load_data()

        # 2. Create index directly
        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=[
                SentenceSplitter(
                    chunk_size=1024,
                    chunk_overlap=200
                )
            ]
        )

    def create_query_engine(self):
        """Criar query engine"""
        if not self.index:
            raise ValueError("Build index first")

        # Simple query engine
        self.query_engine = self.index.as_query_engine(
            response_mode="compact"
        )

    def query(self, question: str):
        """Fazer pergunta"""
        if not self.query_engine:
            raise ValueError("Create query engine first")

        response = self.query_engine.query(question)
        return {
            "answer": str(response),
            "source_nodes": getattr(response, 'source_nodes', [])
        }

# Usage
rag = RAGLlamaIndex("data/")
rag.build_index()
rag.create_query_engine()

result = rag.query("O que é RAG?")
print(f"Resposta: {result['answer']}")
```

### 3.2 Query Engines Avançados

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.types import RESPONSE_TYPE

class CustomResponseSynthesizer(BaseSynthesizer):
    def synthesize(self, query, nodes, **kwargs) -> RESPONSE_TYPE:
        """Synthesizer customizado"""
        context = "\n".join([node.text for node in nodes])

        prompt = f"""
Contexto: {context}

Pergunta: {query}

Responda com base no contexto, citando as fontes.
"""

        llm = OpenAI(temperature=0)
        response = llm.complete(prompt)

        return response.text

# Query engine com reranking
retriever = VectorIndexRetriever(
    index=self.index,
    similarity_top_k=5  # Buscar mais, depois filtrar
)

response_synthesizer = CustomResponseSynthesizer()
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)
```

## 4. Hybrid Search (Dense + Sparse)

### 4.1 Com LangChain

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever, TFIDFRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Dense retriever (embeddings)
dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Sparse retriever (keywords)
texts = ["texto1", "texto2", "texto3"]
sparse_retriever = BM25Retriever.from_texts(texts)

# Ensemble (fused)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # 70% dense, 30% sparse
)

# Query
docs = ensemble_retriever.get_relevant_documents("O que é RAG?")
```

### 4.2 Com LlamaIndex

```python
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response import RESPONSE_TYPE
from llama_index.core.types import MODEL_TYPE

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, weights=[0.7, 0.3]):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.weights = weights

    def retrieve(self, query):
        # Dense
        dense_nodes = self.dense_retriever.retrieve(query)

        # Sparse
        sparse_nodes = self.sparse_retriever.retrieve(query)

        # Combine scores
        all_nodes = dense_nodes + sparse_nodes
        # Remove duplicates and re-rank
        # Implementation details...

        return all_nodes

# Usage
hybrid_retriever = HybridRetriever(
    VectorIndexRetriever(index, similarity_top_k=5),
    BM25Retriever(index, similarity_top_k=5),
    weights=[0.7, 0.3]
)
```

## 5. Evaluation Básico

### 5.1 Métricas Simples

```python
import time
from typing import List, Dict

def evaluate_rag(qa_chain, test_questions: List[Dict]):
    """Avaliar RAG com métricas básicas"""
    results = []

    for item in test_questions:
        question = item["question"]
        expected_answer = item["answer"]

        # Query
        start_time = time.time()
        result = qa_chain.run(question)
        latency = time.time() - start_time

        # Métricas básicas
        answer_relevance = evaluate_answer_relevance(
            question, result, expected_answer
        )
        context_precision = evaluate_context_precision(
            question, result["source_documents"]
        )

        results.append({
            "question": question,
            "answer": result,
            "latency": latency,
            "answer_relevance": answer_relevance,
            "context_precision": context_precision
        })

    # Summary
    avg_latency = sum([r["latency"] for r in results]) / len(results)
    avg_relevance = sum([r["answer_relevance"] for r in results]) / len(results)
    avg_precision = sum([r["context_precision"] for r in results]) / len(results)

    return {
        "results": results,
        "metrics": {
            "avg_latency": avg_latency,
            "avg_answer_relevance": avg_relevance,
            "avg_context_precision": avg_precision
        }
    }

def evaluate_answer_relevance(question, answer, expected):
    """LLM-as-judge para relevância"""
    prompt = f"""
Pergunta: {question}
Resposta esperada: {expected}
Resposta gerada: {answer}

Em uma escala de 0-1, quão relevante é a resposta gerada à pergunta?
(0 = irrelevante, 1 = perfeitamente relevante)

Score:"""

    # Usar LLM para avaliar
    score = llm.invoke(prompt)
    return float(score)

# Dataset de teste
test_questions = [
    {
        "question": "O que é RAG?",
        "answer": "RAG é Retrieval-Augmented Generation, uma técnica que combina busca com geração"
    },
    {
        "question": "Como RAG reduz hallucinations?",
        "answer": "RAG reduz hallucinations fornecendo contexto factual da documentação"
    }
]

# Evaluation
metrics = evaluate_rag(qa_chain, test_questions)
print(f"Latência média: {metrics['metrics']['avg_latency']:.2f}s")
print(f"Relevância média: {metrics['metrics']['avg_answer_relevance']:.2f}")
```

## Comparação Performance

| Métrica | Chains | Agents |
|---------|--------|--------|
| **Latência** | 2-3s | 5-10s |
| **Custo por query** | Baixo | Médio-Alto |
| **Qualidade** | Alta | Muito Alta |
| **Flexibilidade** | Média | Alta |
| **Complexidade** | Baixa | Alta |

## Resumo

**Escolha RAG Chains quando:**
- Performance é crítica
- Queries são previsíveis
- Budget é limitado
- Simplicidade é importante

**Escolha RAG Agents quando:**
- Queries são complexas
- Lógica condicional é necessária
- Múltiplas fontes
- Qualidade máxima é importante

## Próximos Passos

- 🔬 **Tutorial Avançado:** [Self-RAG e Agentic RAG](03-advanced.md)
- 💻 **Exemplos Práticos:** [Code Examples](../code-examples/)
- 🎯 **Otimização:** [Guia 07 - Performance Optimization](../07-Performance-Optimization/README.md)
- 📊 **Avaliação:** [Guia 06 - Evaluation & Benchmarks](../06-Evaluation-Benchmarks/README.md)

---

**Anterior:** [Quick Start](../getting-started/01-quickstart.md) | **Próximo:** [Tutorial Avançado](03-advanced.md)

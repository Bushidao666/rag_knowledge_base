# 📚 Guia 00: RAG Fundamentals

### Data: 09/11/2025
### Versão: 1.0
### Tempo de Leitura: 45 minutos
### Nível: Iniciante a Avançado

---

## 📑 Índice

1. [Getting Started (15-30 min)](#1-getting-started-15-30-min)
2. [Tutorial Intermediário (1-2h)](#2-tutorial-intermediário-1-2h)
3. [Tutorial Avançado (3-4h)](#3-tutorial-avançado-3-4h)
4. [Implementation End-to-End (half-day)](#4-implementation-end-to-end-half-day)
5. [Best Practices](#5-best-practices)
6. [Code Examples](#6-code-examples)
7. [Decision Trees](#7-decision-trees)
8. [Troubleshooting](#8-troubleshooting)
9. [Recursos Adicionais](#9-recursos-adicionais)

---

## 1. Getting Started (15-30 min)

### 1.1 O que é RAG?

**RAG (Retrieval-Augmented Generation)** é uma técnica em NLP que combina:

1. **Memória Paramétrica** (modelos pré-treinados) - conhecimento general
2. **Memória Não-Paramétrica** (índices vetoriais) - knowledge externo

**Conceito Simples:**
```
Usuário Pergunta → Busca Relevante → LLM Responde com Contexto
```

### 1.2 Por que usar RAG?

| ✅ Vantagens | ❌ Quando NÃO usar |
|-------------|------------------|
| Knowledge up-to-date | Domínio muito restrito e estático |
| Factualidade melhorada | Precisa de performance máxima |
| Citations/explicabilidade | Tem budget para fine-tuning |
| Custo-efetivo | Queries sempre similares |
| Menos hallucinations | Não precisa de citations |

### 1.3 Arquitetura Básica

```
[Documento] → [Load] → [Split] → [Embed] → [Index]
                                    ↓
[User Query] → [Embed] → [Search] → [Top-K] → [LLM] → [Resposta]
```

**Pipeline em 2 Fases:**

**FASE 1: Indexing (Uma vez)**
1. Load documents
2. Split em chunks
3. Generate embeddings
4. Store no vector DB

**FASE 2: Query (Toda vez)**
1. Embed user query
2. Search similar chunks
3. Generate com contexto

### 1.4 Primeiro Exemplo (5 min)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. Setup
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

# 2. Create vector store (indexing)
texts = ["RAG é uma técnica que...", "Combina memória paramétrica..."]
vectorstore = Chroma.from_texts(texts, embeddings)

# 3. Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 4. Query
answer = qa.run("O que é RAG?")
print(answer)
```

### 1.5 Quando Usar RAG?

**USE RAG se:**
- ✅ Dados mudam frequentemente
- ✅ Precisa de citations
- ✅ Volume de dados é grande
- ✅ Custo de re-treino é alto
- ✅ Knowledge up-to-date é crítico

**NÃO use se:**
- ❌ Domínio estático e restrito
- ❌ Performance máxima é prioridade
- ❌ Tem budget para fine-tuning

---

## 2. Tutorial Intermediário (1-2h)

### 2.1 Comparando Abordagens

| Critério | RAG | Fine-tuning | Pure Generative | Vector Search |
|---------|-----|------------|---------------|-------------|
| **Knowledge** | External + Paramétrico | Paramétrico | Paramétrico | External |
| **Atualização** | Fácil (update index) | Caro (re-train) | Impossível | Fácil |
| **Explicabilidade** | Alta (citations) | Baixa | Baixa | Alta |
| **Custo Inicial** | Baixo-Médio | Alto | Baixo | Baixo |
| **Performance** | Alta | Muito Alta | Média | Alta |
| **Flexibilidade** | Alta | Baixa | Alta | Baixa |
| **Hallucination** | Menor | Pode ocorrer | Frequente | Não ocorre |

### 2.2 Implementação com LangChain

#### RAG Agentes (Mais Flexível)
```python
from langchain.agents import tool, create_agent
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    docs = vector_store.similarity_search(query, k=2)
    context = "\n\n".join(f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs)
    return context, docs

model = OpenAI(temperature=0)
agent = create_agent(
    model,
    tools=[retrieve_context],
    system_prompt="Use tools to help answer queries."
)

# Agente busca quando necessário, múltiplas buscas
result = agent.run("Explique RAG e dê exemplos")
```

**Vantagens:**
- ✅ Busca apenas quando necessário
- ✅ Múltiplas buscas para queries complexas
- ✅ Contexto dinâmico

**Desvantagens:**
- ❌ Duas chamadas LLM (mais caro)
- ❌ Maior latência

#### RAG Chains (Mais Rápido)
```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Context é sempre injetado
template = """Use the following context to answer the question:

Context: {context}

Question: {question}

Answer:"""

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

**Vantagens:**
- ✅ Uma única chamada LLM
- ✅ Menor latência
- ✅ Mais barato

**Desvantagens:**
- ❌ Sempre busca (mesmo para queries simples)
- ❌ Menos flexível

### 2.3 LlamaIndex: Index-Centric

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. Load documents
documents = SimpleDirectoryReader("data").load_data()

# 2. Create index
index = VectorStoreIndex.from_documents(documents)

# 3. Create query engine
query_engine = index.as_query_engine()

# 4. Query
response = query_engine.query("O que é RAG?")
print(response)
```

**Pipeline LlamaIndex:** Loading → Indexing → Querying → Storing

**Retrievers disponíveis:**
- Auto Merging
- BM25
- Router
- Ensemble

### 2.4 Melhores Práticas

**Chunking:**
- **Tamanho**: 1000 caracteres
- **Overlap**: 200 caracteres
- **Justificativa**: Equilibra contexto e precision

**Retrieval:**
- **k**: 2-5 documentos
- **Score threshold**: Filtrar resultados irrelevantes
- **Metadata**: Preservar sources

**Prompting:**
- Inclua sempre citations
- Use contexto relevante
- Especifique formato de resposta

```python
# Exemplo de prompt com citations
template = """
Use o contexto abaixo para responder a pergunta.
Se a informação não estiver no contexto, diga que não sabe.
Sempre cite a fonte.

Contexto: {context}
Pergunta: {question}
Resposta (com citations):"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
```

### 2.5 Exemplo Completo: Q&A sobre Documentos

```python
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentQA:
    def __init__(self, documents_dir):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.vectorstore = None
        self.qa = None
        self.documents_dir = documents_dir

    def build_index(self):
        """Criar índice a partir de documentos"""
        # 1. Load documents
        loader = TextLoader(self.documents_dir)
        docs = loader.load()

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # 3. Create vector store
        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings
        )

        # 4. Create QA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever()
        )

    def ask(self, question):
        """Fazer pergunta"""
        if not self.qa:
            raise ValueError("Build index first")
        return self.qa.run(question)

# Usage
qa_system = DocumentQA("meus_docs.txt")
qa_system.build_index()
answer = qa_system.ask("Quais são os principais conceitos de RAG?")
print(answer)
```

---

## 3. Tutorial Avançado (3-4h)

### 3.1 Evolução do RAG (2020-2025)

**Timeline:**
- **2020**: RAG original (Lewis et al.)
- **2021-2022**: Adoção em produção
- **2023**: Frameworks (LangChain, LlamaIndex)
- **2024**: RAG avançado (Self-RAG, Corrective RAG, Agentic RAG)
- **2025**: Multimodal RAG, Federated RAG

### 3.2 Self-RAG (2024)

**Conceito:** Sistema que se auto-avalia e melhora

```python
class SelfRAG:
    def __init__(self, retriever, generator, critic):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic

    def query(self, question, max_iterations=3):
        for i in range(max_iterations):
            # 1. Retrieve
            context = self.retriever.get_relevant_docs(question)

            # 2. Generate
            answer = self.generator.generate(question, context)

            # 3. Self-critique
            score = self.critic.evaluate(question, answer, context)

            # 4. Check quality
            if score > 0.8:  # threshold
                return answer

        return answer  # Return best effort
```

### 3.3 Agentic RAG

**Conceito:** Multi-step reasoning com tools

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.llms import OpenAI
from langchain.tools import Tool

# Tools
def search_docs(query):
    return vectorstore.similarity_search(query)

def calculate_score(answer):
    # LLM-based evaluation
    return evaluation_llm.evaluate(answer)

def expand_query(query):
    # Query expansion
    return llm.expand(query)

# Create agent
tools = [
    Tool(name="search", func=search_docs, description="Search documents"),
    Tool(name="score", func=calculate_score, description="Score answer quality"),
    Tool(name="expand", func=expand_query, description="Expand query")
]

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.run("Como RAG reduz hallucinations?")
```

### 3.4 Hybrid Search (Dense + Sparse)

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever, TFIDFRetriever

# Dense retriever (embeddings)
dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Sparse retriever (keywords)
sparse_retriever = BM25Retriever.from_texts(texts)

# Ensemble (fused)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # 70% dense, 30% sparse
)

# Query
docs = ensemble_retriever.get_relevant_documents("O que é RAG?")
```

**Vantagens:**
- Combina semantic + keyword search
- Melhor recall + precision
- Menos sensível a query formulation

### 3.5 Multimodal RAG

```python
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings

class MultimodalRAG:
    def __init__(self):
        self.text_encoder = OpenAIEmbeddings()
        self.image_encoder = CLIPEmbeddings()
        self.vectorstore = Chroma()

    def index_document(self, text, image_path=None):
        # Encode text
        text_embedding = self.text_encoder.embed_query(text)

        # Encode image if present
        embeddings = [text_embedding]
        if image_path:
            image_embedding = self.image_encoder(image_path)
            embeddings.append(image_embedding)

        # Store
        self.vectorstore.add_embeddings(
            embeddings=embeddings,
            metadatas=[{"type": "text"}, {"type": "image"}]
        )

    def query(self, text_query, image_query=None):
        # Encode query
        if image_query:
            query_embedding = self.image_encoder(image_query)
        else:
            query_embedding = self.text_encoder.embed_query(text_query)

        # Search
        return self.vectorstore.similarity_search(
            query_embedding,
            k=5
        )
```

### 3.6 Reranking com Cross-Encoders

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query, documents, top_k=3):
        # Create pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Score
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze()

        # Sort by score
        scored_docs = [
            (doc, score.item())
            for doc, score in zip(documents, scores)
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [doc for doc, _ in scored_docs[:top_k]]

# Usage
reranker = CrossEncoderReranker()
initial_docs = vectorstore.similarity_search(query, k=10)
reranked_docs = reranker.rerank(query, initial_docs, top_k=3)
```

### 3.7 Evaluation e Monitoring

```python
from langchain.evaluation import EvaluatorType
from langsmith import Client

class RAGEvaluator:
    def __init__(self):
        self.client = Client()

    def evaluate_faithfulness(self, question, answer, context):
        """Evaluate if answer is faithful to context"""
        prompt = f"""
        Question: {question}
        Context: {context}
        Answer: {answer}

        On a scale of 0-1, how faithful is the answer to the context?
        (0 = not faithful, 1 = perfectly faithful)
        """
        score = self.llm.invoke(prompt)
        return float(score)

    def evaluate_answer_relevance(self, question, answer):
        """Evaluate if answer is relevant to question"""
        # LLM-as-judge or human evaluation
        pass

    def batch_evaluate(self, dataset):
        """Evaluate on dataset"""
        results = []
        for item in dataset:
            faithfulness = self.evaluate_faithfulness(
                item["question"],
                item["answer"],
                item["context"]
            )
            results.append({
                "question": item["question"],
                "faithfulness": faithfulness
            })
        return results
```

### 3.8 Performance Optimization

```python
import time
import asyncio
from functools import lru_cache

class OptimizedRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.cache = {}

    @lru_cache(maxsize=1000)
    def get_embedding(self, text):
        """Cache embeddings"""
        return self.embeddings.embed_query(text)

    async def async_batch_search(self, queries):
        """Async batch search"""
        tasks = [
            self.vectorstore.asimilarity_search(query)
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        return results

    def batch_generate(self, questions, contexts):
        """Batch generation para melhor throughput"""
        prompts = [
            self.format_prompt(q, c)
            for q, c in zip(questions, contexts)
        ]
        # Process in batches to avoid rate limits
        batch_size = 10
        all_answers = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            answers = self.llm.generate(batch)
            all_answers.extend(answers)
        return all_answers

    def format_prompt(self, question, context):
        return f"""
        Context: {context}
        Question: {question}
        Answer:"""
```

---

## 4. Implementation End-to-End (half-day)

### 4.1 Estrutura do Projeto

```
rag_project/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── indexing.py
│   ├── retrieval.py
│   ├── generation.py
│   └── evaluation.py
├── config/
│   └── settings.yaml
├── tests/
│   ├── test_indexing.py
│   ├── test_retrieval.py
│   └── test_qa.py
├── main.py
└── requirements.txt
```

### 4.2 main.py

```python
#!/usr/bin/env python3
"""
RAG System - End-to-End Implementation
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Config
from config.settings import Settings

settings = Settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG System completo"""

    def __init__(self, data_dir: str, persist_dir: str):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)

        # Init components
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model
        )
        self.llm = OpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators
        )

        self.vectorstore = None
        self.qa_chain = None

        logger.info("RAG System initialized")

    def load_documents(self) -> List[Dict[str, Any]]:
        """Carregar todos os documentos"""
        documents = []

        for file_path in self.data_dir.rglob("*.pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {file_path}")

        for file_path in self.data_dir.rglob("*.txt"):
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} from {file_path}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def build_index(self, documents: List[Dict[str, Any]]):
        """Criar índice vetorial"""
        logger.info("Building index...")

        # Split documents
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )

        # Persist
        self.vectorstore.persist()
        logger.info(f"Index saved to {self.persist_dir}")

    def load_index(self):
        """Carregar índice existente"""
        if self.persist_dir.exists():
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            logger.info(f"Index loaded from {self.persist_dir}")
            return True
        return False

    def create_qa_chain(self):
        """Criar QA chain"""
        if not self.vectorstore:
            raise ValueError("No index available. Build or load index first.")

        # Custom prompt
        template = """Use o contexto abaixo para responder à pergunta.
Se a informação não estiver no contexto, diga que não sabe.
Sempre cite a fonte (título do documento).

Contexto: {context}

Pergunta: {question}

Resposta (com citações):"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_k=settings.k_retrieval
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        logger.info("QA chain created")

    def query(self, question: str) -> Dict[str, Any]]:
        """Fazer pergunta"""
        if not self.qa_chain:
            raise ValueError("QA chain not created")

        logger.info(f"Querying: {question}")

        start_time = time.time()
        result = self.qa_chain({"query": question})
        end_time = time.time()

        response = {
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ],
            "latency_seconds": end_time - start_time
        }

        logger.info(f"Query completed in {response['latency_seconds']:.2f}s")
        return response

    def run_pipeline(self, question: str):
        """Executar pipeline completo"""
        # Check if index exists
        if not self.load_index():
            # Build index
            documents = self.load_documents()
            self.build_index(documents)

        # Create QA chain
        self.create_qa_chain()

        # Query
        return self.query(question)

# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with documents")
    parser.add_argument("--persist-dir", default="vectorstore", help="Where to save index")
    parser.add_argument("--query", help="Question to ask")
    args = parser.parse_args()

    rag = RAGSystem(args.data_dir, args.persist_dir)

    if args.query:
        result = rag.run_pipeline(args.query)
        print(f"\nResposta: {result['answer']}")
        print(f"\nFontes:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"{i}. {doc['content']}")
            print(f"   Metadata: {doc['metadata']}")
    else:
        # Interactive mode
        while True:
            question = input("\nPergunta (ou 'quit'): ")
            if question.lower() == 'quit':
                break
            result = rag.run_pipeline(question)
            print(f"\n{result['answer']}")
```

### 4.3 config/settings.yaml

```yaml
# Configuração do sistema
embedding_model: "text-embedding-ada-002"
llm_model: "gpt-3.5-turbo"
temperature: 0.1
max_tokens: 1000

# Chunking
chunk_size: 1000
chunk_overlap: 200
separators: ["\n\n", "\n", ".", " "]

# Retrieval
k_retrieval: 4

# Performance
batch_size: 10
cache_size: 1000

# Paths
data_dir: "data/raw"
persist_dir: "vectorstore"
```

### 4.4 tests/test_qa.py

```python
import pytest
from main import RAGSystem

def test_rag_system():
    """Test RAG system"""
    # Setup
    rag = RAGSystem(
        data_dir="tests/data",
        persist_dir="tests/vectorstore"
    )

    # Build index
    documents = rag.load_documents()
    assert len(documents) > 0

    rag.build_index(documents)
    assert rag.vectorstore is not None

    # Create QA chain
    rag.create_qa_chain()
    assert rag.qa_chain is not None

    # Query
    result = rag.query("O que é RAG?")
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "source_documents" in result
    assert len(result["source_documents"]) > 0

    print(f"Test passed! Answer: {result['answer']}")

if __name__ == "__main__":
    test_rag_system()
```

---

## 5. Best Practices

### 5.1 Design

| ✅ DO | ❌ DON'T |
|-----|----------|
| Use chunk_size=1000, overlap=200 | Chunking muito pequeno ou muito grande |
| Preservar metadata | Perder informações de source |
| Incluir citations | Responder sem citar fontes |
| Usar k=2-5 documentos | k muito alto (ruim para LLM) |
| Limpar dados antes | Indexar dados ruidosos |

### 5.2 Performance

| ✅ DO | ❌ DON'T |
|-----|----------|
| Cache embeddings e queries | Gerar embedding toda vez |
| Processar em batch | Uma operação por vez |
| Usar async/await | Operações síncronas |
| Monitor com LangSmith | Fazer deploy sem monitorar |
| Otimizar com reranking | Sempre usar dense search |

### 5.3 Quality

| ✅ DO | ❌ DON'T |
|-----|----------|
| A/B test approaches | Escolher baseado em intuição |
| Human evaluation | Só usar métricas automáticas |
| Coletar feedback | Ignorar feedback dos usuários |
| Versionar modelos | Usar sempre mesma versão |
| Validar retrieval quality | Assumir que retrieval funciona |

### 5.4 Production

| ✅ DO | ❌ DON'T |
|-----|----------|
| Health checks | Deploy sem monitoring |
| Rate limiting | Ignorar rate limits |
| Error handling | Deixar erros sem tratamento |
| Logging estruturado | Usar print() para debug |
| Backup do índice | Indexar só em memória |

### 5.5 Security

| ✅ DO | ❌ DON'T |
|-------|-----------|
| API keys seguras | Hardcodar keys no código |
| Sanitizar inputs | Usar raw user input |
| Rate limiting | Permitir unlimited requests |
| Audit logs | Não logs de acesso |
| Data encryption | Dados em texto plano |

---

## 6. Code Examples

### Example 1: Minimal RAG (20 linhas)

```python
# Minimal RAG em 20 linhas
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Setup
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)
vectorstore = Chroma.from_texts(["RAG é uma técnica que..."], embeddings)
qa = RetrievalQA.from_chain_type(llm, vectorstore.as_retriever())

# Query
answer = qa.run("O que é RAG?")
print(answer)
```

### Example 2: RAG com PDF

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load PDF
loader = PyPDFLoader("documento.pdf")
pages = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)

# Index
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# Query
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever()
)
answer = qa.run("Qual o tema do documento?")
print(answer)
```

### Example 3: RAG com Conversational Memory

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5
)

# QA with memory
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    memory=memory,
    retriever=vectorstore.as_retriever()
)

# Chat
chat_history = []
result = qa({
    "question": "O que é RAG?",
    "chat_history": chat_history
})
print(result["answer"])

# Follow-up question
result2 = qa({
    "question": "Como funciona?",
    "chat_history": result["chat_history"]
})
print(result2["answer"])
```

### Example 4: RAG com Multi-Query

```python
from langchain.retrievers import MultiQueryRetriever
from langchain.llms import OpenAI

# Generate multiple queries
llm = OpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Query
docs = retriever.get_relevant_documents("O que é RAG?")
print(f"Found {len(docs)} relevant documents")

# Create QA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
answer = qa.run("Explique RAG")
print(answer)
```

### Example 5: RAG com Caching

```python
from functools import lru_cache
import hashlib

# Cache embeddings
@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    return embeddings.embed_query(text)

# Cache queries
@lru_cache(maxsize=1000)
def cached_query(question: str):
    return qa.run(question)

# Usage
answer1 = cached_query("O que é RAG?")  # Cached
answer2 = cached_query("Como funciona?")  # Cached
```

---

## 7. Decision Trees

### 7.1 Decision Tree: Qual Abordagem Usar?

```
START
  │
  ├─> Precisa de knowledge up-to-date?
  │   └─ NO ───> Fine-tuning ou Pure Generative
  │
  └─ YES ──> Domínio restrito e estático?
      ├─ YES ──> Fine-tuning
      │
      └─ NO ──> Volume de dados > 10GB?
          ├─ YES ──> RAG
          │
          └─ NO ──> Budget para fine-tuning?
              ├─ YES ──> Fine-tuning
              │
              └─ NO ──> RAG
```

### 7.2 Decision Tree: Qual Framework?

```
START
  │
  ├─> Experiência com LLMs?
  │   └─ Beginner ──> LlamaIndex (simpler)
  │
  └─ Intermediate+ ──> Precisa de flexibilidade?
      ├─ YES ──> LangChain
      │
      └─ NO ──> Need REST API?
          ├─ YES ──> Haystack
          │
          └─ NO ──> Simplicity?
              ├─ YES ──> txtai
              │
              └─ NO ──> LlamaIndex
```

### 7.3 Decision Tree: Vector Database?

```
START
  │
  ├─> Scale?
  │   ├─ < 1M vectors ──> Chroma ou FAISS
  │   │
  │   ├─ 1M-100M vectors ──> Qdrant ou Weaviate
  │   │
  │   └─ > 100M vectors ──> Pinecone ou Milvus
  │
  └─ Deployment?
      ├─ Self-hosted ──> Qdrant, Weaviate, Milvus
      │
      └─ Managed ──> Pinecone, Weaviate Cloud
```

---

## 8. Troubleshooting

### 8.1 Problema: Low Retrieval Quality

**Sintomas:**
- Respostas irrelevantes
- Baixo Recall@k
- Context não ajuda

**Causas Comuns:**
- Chunking inadequado (muito grande/pequeno)
- Overlap insuficiente
- Embedding model inadequado
- k muito baixo/alto
- Dados ruidosos

**Soluções:**

1. **Ajustar chunking:**
```python
# Testar diferentes tamanhos
for chunk_size in [500, 1000, 2000]:
    for overlap in [100, 200, 500]:
        # Evaluate retrieval quality
        recall = evaluate_recall(chunk_size, overlap)
        print(f"chunk_size={chunk_size}, overlap={overlap}, recall={recall}")
```

2. **Mudare embedding model:**
```python
# Experimentar diferentes models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# OpenAI (melhor qualidade, mais caro)
embeddings = OpenAIEmbeddings()

# BGE (open-source, bom quality)
embeddings = HuggingFaceEmbeddings('BAAI/bge-large-en-v1.5')
```

3. **Ajustar k:**
```python
# Testar diferentes k
for k in [2, 4, 6, 8]:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_k=k)
    )
    quality = evaluate(qa)
    print(f"k={k}, quality={quality}")
```

4. **Limpar dados:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Filter noise
def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special chars
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

# Load and clean
loader = TextLoader("file.txt")
docs = loader.load()
for doc in docs:
    doc.page_content = clean_text(doc.page_content)
```

### 8.2 Problema: Slow Performance

**Sintomas:**
- Query latency > 5s
- Indexing muito lento
- High memory usage

**Soluções:**

1. **Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    return embeddings.embed_query(text)

# Cache query results
@lru_cache(maxsize=1000)
def cached_query(question):
    return qa.run(question)
```

2. **Batch processing:**
```python
# Processar em batch
def batch_embed(texts):
    return embeddings.embed_documents(texts)

texts = ["text1", "text2", "text3"]
embeddings = batch_embed(texts)
```

3. **Async operations:**
```python
import asyncio

async def async_query(question):
    return await qa.ainvoke({"query": question})

# Multiple queries
tasks = [async_query(q) for q in questions]
results = await asyncio.gather(*tasks)
```

4. **Vector compression:**
```python
# Quantization (FAISS)
import faiss
import numpy as np

# 32-bit to 8-bit
quantizer = faiss.IndexScalarQuantizer(faiss.METRIC_L2)
index = quantizer.train(embeddings)
index.add(embeddings)
```

### 8.3 Problema: Hallucinations

**Sintomas:**
- Respostas com informações falsas
- Facts não confirmados no context
- Inconsistência com source

**Soluções:**

1. **Prompt engineering:**
```python
template = """
Use APENAS o contexto fornecido para responder.
Se a informação não estiver no contexto, diga "Não tenho informação suficiente para responder."

Contexto: {context}

Pergunta: {question}

Resposta (só com informações do contexto):"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
```

2. **Fact-checking:**
```python
from langchain.chains import LLMCheckerChain

# Add verification step
checker = LLMCheckerChain.from_llm(llm)
checked_result = checker.run(question, context, answer)
```

3. **Citation requirement:**
```python
# Prompt exige citações
template = """
Responda à pergunta usando APENAS as informações do contexto.
SEMPRE inclua a fonte (título do documento) para cada fato.

Contexto: {context}
Pergunta: {question}
Resposta com citações:"""

# Verify citations exist
if "Fonte:" not in answer:
    answer += "\n\n[Não foi possível verificar a fonte]"
```

### 8.4 Problema: Inconsistent Results

**Sintomas:**
- Mesma query retorna respostas diferentes
- Variação na quality
- Non-deterministic behavior

**Soluções:**

1. **Setar temperature baixo:**
```python
llm = OpenAI(temperature=0.0)  # Deterministic
```

2. **固定 prompt:**
```python
# Sempre usar mesmo prompt template
prompt = PromptTemplate(
    template="Question: {question}\nContext: {context}\nAnswer:",
    input_variables=["question", "context"]
)
```

3. **Cache determinístico:**
```python
import hashlib

def deterministic_cache_key(question, context):
    content = f"{question}|{context}"
    return hashlib.md5(content.encode()).hexdigest()

# Verificar se contexto mudou
```

### 8.5 Problema: High Costs

**Sintomas:**
- API costs muito altos
- Many API calls
- Expensive LLM usage

**Soluções:**

1. **Route queries:**
```python
def route_query(question):
    # Simple queries: cheap LLM
    if is_simple(question):
        return cheap_llm
    # Complex queries: expensive LLM
    return expensive_llm
```

2. **Reduce context:**
```python
# Menor k = menos tokens
retriever = vectorstore.as_retriever(search_k=2)  # vs 8

# Summarization
summarizer = load_summarizer()
context = summarizer.run(long_context)
```

3. **Cache hit rate:**
```python
# Maximizar cache hit
@lru_cache(maxsize=10000)
def cached_answer(question):
    return generate_answer(question)

# Cache policy
if question in cache:
    return cache[question]
```

---

## 9. Recursos Adicionais

### 9.1 Papers Importantes
- **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **Asai et al. (2023)**: "Self-RAG: Learning to Retrieve, Generate, and Critique"
- **Gao et al. (2023)**: "Retrieval-Augmented Generation for Large Language Models"

### 9.2 Ferramentas
- **LangChain**: RAG framework
- **LlamaIndex**: Index-centric RAG
- **Chroma**: Vector database
- **Pinecone**: Managed vector DB
- **RAGAS**: RAG evaluation
- **LangSmith**: Monitoring

### 9.3 Datasets
- **MS MARCO**: Question answering
- **BEIR**: Information retrieval
- **Natural Questions**: Real user queries
- **FiQA**: Financial QA
- **HotpotQA**: Multi-hop reasoning

### 9.4 Tutoriais
- LangChain RAG tutorial
- LlamaIndex quickstart
- Haystack RAG pipeline
- OpenAI RAG guide

### 9.5 Comunidades
- LangChain Discord
- LlamaIndex Discord
- r/LangChain (Reddit)
- r/MachineLearning (Reddit)

---

**Última atualização**: 09/11/2025
**Versão**: 1.0
**Autor**: RAG Knowledge Base Project

---

**Próximo**: [Guia 01 - Document Processing](./01-Document-Processing/README.md)

# Troubleshooting - RAG Fundamentals

## Problemas Comuns e Soluções

### 1. Low Retrieval Quality

**Sintomas:**
- Respostas irrelevantes
- Baixo Recall@k
- Context não ajuda na resposta

**Causas Comuns:**
- Chunking inadequado (muito grande/pequeno)
- Overlap insuficiente
- Embedding model inadequado
- k muito baixo/alto
- Dados ruidosos

**Soluções:**

#### Ajustar Chunking
```python
# Testar diferentes tamanhos
for chunk_size in [500, 1000, 2000]:
    for overlap in [100, 200, 500]:
        # Evaluate retrieval quality
        recall = evaluate_recall(chunk_size, overlap)
        print(f"chunk_size={chunk_size}, overlap={overlap}, recall={recall}")
```

**Recomendação:**
- `chunk_size=1000` - padrão para maioria dos casos
- `chunk_overlap=200` - preserva contexto entre chunks

#### Mudar Embedding Model
```python
# Experimentar diferentes models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# OpenAI (melhor qualidade, mais caro)
embeddings = OpenAIEmbeddings()

# BGE (open-source, bom quality)
embeddings = HuggingFaceEmbeddings('BAAI/bge-large-en-v1.5')

# MiniLM (rápido, qualidade média)
embeddings = HuggingFaceEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
```

#### Ajustar k
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

**Recomendação:**
- `k=2-3` - mínimo contexto
- `k=4-5` - balance contexto/precisão
- `k=6-8` - muito contexto, pode confundir LLM

#### Limpar Dados
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Filter noise
def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special chars
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    return text

# Load and clean
loader = TextLoader("file.txt")
docs = loader.load()
for doc in docs:
    doc.page_content = clean_text(doc.page_content)
```

---

### 2. Slow Performance

**Sintomas:**
- Query latency > 5s
- Indexing muito lento
- High memory usage

**Soluções:**

#### Caching
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

#### Batch Processing
```python
# Processar em batch
def batch_embed(texts):
    return embeddings.embed_documents(texts)

texts = ["text1", "text2", "text3"]
embeddings = batch_embed(texts)
```

#### Async Operations
```python
import asyncio

async def async_query(question):
    return await qa.ainvoke({"query": question})

# Multiple queries
tasks = [async_query(q) for q in questions]
results = await asyncio.gather(*tasks)
```

#### Vector Compression
```python
# Quantization (FAISS)
import faiss
import numpy as np

# 32-bit to 8-bit
quantizer = faiss.IndexScalarQuantizer(faiss.METRIC_L2)
index = quantizer.train(embeddings)
index.add(embeddings)
```

---

### 3. Hallucinations

**Sintomas:**
- Respostas com informações falsas
- Facts não confirmados no context
- Inconsistência com source

**Soluções:**

#### Prompt Engineering
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

#### Fact-Checking
```python
from langchain.chains import LLMCheckerChain

# Add verification step
checker = LLMCheckerChain.from_llm(llm)
checked_result = checker.run(question, context, answer)
```

#### Citation Requirement
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

---

### 4. Inconsistent Results

**Sintomas:**
- Mesma query retorna respostas diferentes
- Variação na quality
- Non-deterministic behavior

**Soluções:**

#### Setar Temperature Baixo
```python
llm = OpenAI(temperature=0.0)  # Deterministic
```

#### Fixed Prompt
```python
# Sempre usar mesmo prompt template
prompt = PromptTemplate(
    template="Question: {question}\nContext: {context}\nAnswer:",
    input_variables=["question", "context"]
)
```

#### Cache Determinístico
```python
import hashlib

def deterministic_cache_key(question, context):
    content = f"{question}|{context}"
    return hashlib.md5(content.encode()).hexdigest()

# Verificar se contexto mudou
```

---

### 5. High Costs

**Sintomas:**
- API costs muito altos
- Many API calls
- Expensive LLM usage

**Soluções:**

#### Route Queries
```python
def route_query(question):
    # Simple queries: cheap LLM
    if is_simple(question):
        return cheap_llm
    # Complex queries: expensive LLM
    return expensive_llm
```

#### Reduce Context
```python
# Menor k = menos tokens
retriever = vectorstore.as_retriever(search_k=2)  # vs 8

# Summarization
summarizer = load_summarizer()
context = summarizer.run(long_context)
```

#### Cache Hit Rate
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

### 6. Out of Memory Errors

**Sintomas:**
- Process killed
- Memory allocation failed
- Slow processing

**Soluções:**

#### Batch Processing
```python
# Processar em batches menores
batch_size = 100
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    process_batch(batch)
```

#### Vector Database External
```python
# Usar Pinecone (cloud) ao invés de Chroma (local)
from langchain.vectorstores import Pinecone

vectorstore = Pinecone.from_texts(
    texts,
    embeddings,
    index_name=index_name
)
```

#### Streaming
```python
# Processar documentos grandes em chunks
for doc in large_documents:
    process_document_stream(doc)
```

---

### 7. Import Errors

**Sintomas:**
- ModuleNotFoundError
- No module named 'langchain'

**Soluções:**

#### Instalar Dependências
```bash
pip install --upgrade langchain
pip install --upgrade langchain-community
pip install --upgrade chromadb
```

#### Virtual Environment
```bash
# Criar venv
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# rag_env\Scripts\activate  # Windows

# Instalar
pip install langchain chromadb
```

---

### 8. API Key Issues

**Sintomas:**
- AuthenticationError
- Invalid API key
- RateLimitError

**Soluções:**

#### Configurar API Key
```bash
export OPENAI_API_KEY="sua-key-aqui"
```

```python
import os
os.environ["OPENAI_API_KEY"] = "sua-key-aqui"
```

#### Rate Limiting
```python
import time
from openai.error import RateLimitError

def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)
```

---

## Debug Checklist

- [ ] Verificar se API key está configurada
- [ ] Confirmar que documentos foram carregados
- [ ] Verificar se chunks têm conteúdo
- [ ] Testar embeddings separadamente
- [ ] Verificar k value
- [ ] Testar LLM com prompt simples
- [ ] Verificar logs de erro
- [ ] Validar formato dos documentos

## Ferramentas de Debug

```python
# Verificar vector store
print(f"Number of documents: {vectorstore._collection.count()}")

# Verificar embeddings
sample_text = "test"
embedding = embeddings.embed_query(sample_text)
print(f"Embedding dim: {len(embedding)}")

# Verificar similaridade
docs = vectorstore.similarity_search("test", k=1)
print(f"Similarity score: {docs[0].metadata.get('score', 'N/A')}")
```

## Performance Profiling

```python
import time

# Profile query
start = time.time()
result = qa.run("O que é RAG?")
latency = time.time() - start
print(f"Query latency: {latency:.2f}s")
```

## Prevention Best Practices

1. **Teste com small dataset** antes de production
2. **Monitore métricas** continuamente
3. **Implemente logging** detalhado
4. **Versione** seu código e configurações
5. **Backup** de vector store
6. **Health checks** no sistema
7. **Error handling** robusto
8. **Retry logic** para APIs
9. **Circuit breaker** para falhas
10. **Metrics dashboard** para observabilidade

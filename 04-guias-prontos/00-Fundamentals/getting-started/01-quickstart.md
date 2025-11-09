# Quick Start: RAG Fundamentals

**Tempo estimado:** 15-30 minutos
**Nível:** Iniciante
**Pré-requisitos:** Python 3.8+, OpenAI API key (opcional)

## Objetivo
Aprender os conceitos fundamentais de RAG e criar seu primeiro sistema básico em 15 minutos.

## O que é RAG?
RAG (Retrieval-Augmented Generation) combina:
- **Memória Paramétrica** (modelos pré-treinados) - conhecimento geral
- **Memória Não-Paramétrica** (índices vetoriais) - conhecimento externo

```
Usuário Pergunta → Busca Relevante → LLM Responde com Contexto
```

## Passo a Passo

### Passo 1: Instalar Dependências
```bash
pip install langchain openai chromadb
```

### Passo 2: Entender a Arquitetura
RAG tem 2 fases:

**FASE 1: Indexing (uma vez)**
1. Load documents
2. Split em chunks
3. Generate embeddings
4. Store no vector DB

**FASE 2: Query (sempre)**
1. Embed user query
2. Search similar chunks
3. Generate com contexto

### Passo 3: Primeiro Exemplo (5 min)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. Setup (configure sua API key)
# export OPENAI_API_KEY="sua-key-aqui"
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

# 2. Create vector store (indexing)
texts = [
    "RAG é uma técnica que combina busca e geração",
    "RAG usa memória paramétrica e não-paramétrica",
    "Lewis et al. (2020) introduziu RAG"
]
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

### Passo 4: Exemplo com PDF

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load PDF
loader = PyPDFLoader("documento.pdf")
pages = loader.load()

# 2. Split em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)

# 3. Index
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Query
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
answer = qa.run("Qual o tema principal?")
print(answer)
```

### Passo 5: Compreender os Parâmetros

**Chunking (divisão de texto):**
- `chunk_size=1000` - Tamanho ideal para equilíbrio
- `chunk_overlap=200` - Sobreposição preserva contexto

**Retrieval (busca):**
- `k=2-5` - Número de documentos a recuperar
- `temperature=0` - Determinístico para factualidade

## Exemplo Completo Testável

```python
#!/usr/bin/env python3
"""
RAG Quick Start - Exemplo básico funcional
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub

# 1. Setup (sem API key necessária)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.1}
)

# 2. Texto de exemplo
texts = [
    "RAG combina memória paramétrica e não-paramétrica",
    "Memória paramétrica é o conhecimento dos LLMs pré-treinados",
    "Memória não-paramétrica são os índices vetoriais externos",
    "RAG reduz hallucinations e melhora factualidade"
]

# 3. Index
vectorstore = FAISS.from_texts(texts, embeddings)

# 4. Query
retriever = vectorstore.as_retriever(search_k=2)
docs = retriever.get_relevant_documents("O que é RAG?")

# 5. Generate
context = "\n".join([doc.page_content for doc in docs])
prompt = f"Com base no contexto: {context}\n\nPergunta: O que é RAG?\nResposta:"
answer = llm(prompt)

print(f"Contexto encontrado:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

print(f"\nResposta: {answer}")
```

## Quando Usar RAG?

### ✅ USE RAG se:
- Precisa de knowledge up-to-date
- Dados mudam frequentemente
- Precisa de citations/explicabilidade
- Volume de dados é grande
- Custo de fine-tuning é alto

### ❌ NÃO USE se:
- Domínio restrito e estático
- Precisa de performance máxima
- Tem budget para fine-tuning
- Queries sempre similares

## Próximos Passos

- 📖 **Tutorial Intermediário:** [LangChain vs LlamaIndex](../tutorials/02-intermediate.md)
- 💻 **Tutoriais Práticos:** [Tutoriais](../tutorials/)
- 📚 **Exemplos Completos:** [Code Examples](../code-examples/)
- 🔧 **Troubleshooting:** [Problemas Comuns](../troubleshooting/common-issues.md)

## Recursos

- 📄 **Paper Original:** Lewis et al. (2020) - https://arxiv.org/abs/2005.11401
- 📖 **LangChain Docs:** https://docs.langchain.com/oss/python/langchain/rag
- 🦙 **LlamaIndex:** https://developers.llamaindex.ai/
- 🎯 **Comparação Frameworks:** [Guia 10](../10-Frameworks-Tools/README.md)

## Problemas Comuns

### Erro: API Key não configurada
**Solução:** Configure a variável de ambiente
```bash
export OPENAI_API_KEY="sua-key-aqui"
```

### Erro: ImportError
**Solução:** Instalar dependências
```bash
pip install --upgrade langchain chromadb
```

### Resposta sem sentido
**Soluções:**
1. Verificar `chunk_size=1000` e `overlap=200`
2. Ajustar `k=2-5` (mais contexto)
3. Usar embeddings melhores
4. Prompts mais específicos

---

**Próximo:** [Tutorial Intermediário: Comparando Abordagens](../tutorials/02-intermediate.md)

# Quick Start: Chunking Strategies

**Tempo estimado:** 15-30 minutos
**Nível:** Iniciante
**Pré-requisitos:** Documentos para processar

## Objetivo
Aprender estratégias de chunking para otimizar retrieval e geração

## O que é Chunking?
Dividir documentos grandes em chunks menores para melhor retrieval:
```
Documento Grande → Chunks (1000 chars) → Embeddings → Vector DB
```

## Por que Chunking?

1. **Context Window Limit** - LLMs têm limite de tokens
2. **Melhor Precision** - Chunks menores = mais relevantes
3. **Custo Eficiente** - Menos tokens = menor custo
4. **Precisão Semântica** - Chunks coesos melhoram similarity

## RecursiveCharacterTextSplitter (Padrão)

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Tamanho ideal
    chunk_overlap=200,    # Preserva contexto
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)
```

## Parâmetros Importantes

### chunk_size
- **Padrão:** 1000 caracteres (~250 tokens)
- **Menor (500):** Mais chunks, maior precisão
- **Maior (2000):** Menos chunks, mais contexto

### chunk_overlap
- **Padrão:** 200 caracteres
- **Função:** Preservar contexto entre chunks
- **Resultado:** Evitar informação cortada

### separators
Ordem de prioridade para split:
1. `\n\n` (parágrafos)
2. `\n` (linhas)
3. `.` (frases)
4. ` ` (palavras)

## Estratégias de Chunking

### 1. Fixed Size
Mais simples, baseado apenas em tamanho:
```python
from langchain.text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### 2. Semantic
Baseado em significado:
```python
from langchain.text_splitters import SentenceTransformersTokenizer

# Divide por sentenças
splitter = SentenceTransformersTokenizer(
    chunk_size=1000,
    chunk_overlap=200
)
```

### 3. Hierarchical
Múltiplos níveis:
```python
from langchain.text_splitters import (
    TitleElementSplitter,
    HeaderElementSplitter
)

# Primeiro por headers
header_splitter = HeaderElementSplitter()

# Depois por párágrafos
paragraph_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

## Exemplo Completo

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Load document
loader = TextLoader("documento.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# 3. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

print(f"Documento dividido em {len(chunks)} chunks")
```

## Parâmetros por Caso de Uso

### Q&A System
- **chunk_size:** 1000
- **chunk_overlap:** 200
- **Separação:** `\n\n`, `\n`, `.`

### Summarization
- **chunk_size:** 2000-3000
- **chunk_overlap:** 500
- **Separação:** `\n\n`

### Code Analysis
- **chunk_size:** 500
- **chunk_overlap:** 100
- **Separação:** `\n\n`, função, classe

### Conversational AI
- **chunk_size:** 800
- **chunk_overlap:** 150
- **Separação:** `\n\n`, `\n`

## Comparação Estratégias

| Estratégia | Prós | Contras | Quando Usar |
|------------|------|---------|-------------|
| **Recursive** | Flexible, default | Pode quebrar estruturas | Geral |
| **Fixed** | Simples, rápido | Perde contexto | Textos simples |
| **Semantic** | Preserva significado | Lento | Textos complexos |
| **Hierarchical** | Estrutura preservada | Complexo | Documentos técnicos |

## Custom Splitter

```python
from langchain.text_splitters import TextSplitter

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # Sua lógica customizada
        return custom_split_logic(text)

# Uso
splitter = CustomTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

## Troubleshooting

### Chunks muito pequenos
**Problema:** Perdendo contexto
**Solução:** Aumentar chunk_size
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Aumentar
    chunk_overlap=300
)
```

### Chunks muito grandes
**Problema:** Retrieval impreciso
**Solução:** Diminuir chunk_size
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Diminuir
    chunk_overlap=100
)
```

### Contexto perdido entre chunks
**Problema:** Respostas incompletas
**Solução:** Aumentar overlap
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300  # Aumentar
)
```

## Próximos Passos

- 📖 **Tutorial Avançado:** [Hierarchical Chunking](../tutorials/)
- 💻 **Code Examples:** [Vários Exemplos](../code-examples/)
- 🔧 **Troubleshooting:** [Problemas Comuns](../troubleshooting/common-issues.md)

## Recursos

- 📄 **LangChain Splitters:** https://python.langchain.com/docs/modules/text_splitters/
- 📊 **Comparison Matrix:** [Best Practices](../best-practices/dos-donts.md)
- 🎯 **Document Processing:** [Guia 01](../01-Document-Processing/README.md)

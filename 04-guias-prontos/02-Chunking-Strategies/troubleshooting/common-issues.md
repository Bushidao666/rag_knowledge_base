# Troubleshooting - Chunking Strategies

## Problemas Comuns e Soluções

### 1. Chunks Muito Grandes

**Sintomas:**
- Retrieval impreciso
- Chunks não relacionados na busca
- Alto custo de embedding
- LLM confusion

**Soluções:**

```python
# Reduzir chunk_size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Reduzir de 1000 para 500
    chunk_overlap=100
)
```

**Quando usar chunk_size menor:**
- Textos muito diversos
- Queries específicas
- Alto precision necessário
- Contexto limitado

### 2. Chunks Muito Pequenos

**Sintomas:**
- Contexto perdido
- Respostas fragmentadas
- Information gaps
- Alto overlap (redundância)

**Soluções:**

```python
# Aumentar chunk_size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Aumentar
    chunk_overlap=300
)
```

**Quando usar chunk_size maior:**
- Textos coesos
- Queries amplas
- Contexto rico necessário
- Summarization tasks

### 3. Contexto Perdido Entre Chunks

**Sintomas:**
- Respostas incompletas
- Referências quebradas
- Perda de continuidade

**Soluções:**

```python
# Aumentar overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300  # Aumentar de 200 para 300
)
```

**Cálculo overlap recomendado:**
- 20-30% do chunk_size
- Para chunk_size=1000 → overlap=200-300

### 4. Separação Inadequada

**Sintomas:**
- Chunks quebrados em locais ruins
- Parágrafos cortados
- Sentenças incompletas

**Soluções:**

```python
# Ajustar separadores
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n\n",  # Parágrafos grandes
        "\n\n",    # Parágrafos normais
        "\n",      # Linhas
        ". ",      # Frases
        " ",       # Palavras
        ""         # Caracteres
    ]
)
```

**Ordem dos separadores:**
- Mais específico → Menos específico
- Preservar estrutura natural

### 5. Semantic Boundaries Ignorados

**Sintomas:**
- Chunks mixing topics
- Coherence loss
- Poor retrieval

**Soluções:**

```python
# Usar semantic splitter
from langchain.text_splitters import SentenceTransformersTokenizer

splitter = SentenceTransformersTokenizer(
    chunk_size=1000,
    chunk_overlap=200
)
```

**Ou custom separator:**
```python
# Separadores baseados no conteúdo
custom_separators = [
    "\n\n## ",  # Headers H2
    "\n### ",   # Headers H3
    "\n\n",     # Parágrafos
    ".\n",      # Fim de seção
    ". ",       # Frases
]
```

### 6. Documents Estruturados (Headers, Tabelas)

**Sintomas:**
- Headers separados do conteúdo
- Tabelas quebradas
- Estrutura perdida

**Soluções:**

```python
# Hierarchical splitting
from langchain.text_splitters import (
    HeaderElementSplitter,
    RecursiveCharacterTextSplitter
)

# 1. Separar por headers
header_splitter = HeaderElementSplitter()
sections = header_splitter.split_text(text)

# 2. Chunking dentro das seções
for section in sections:
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". "]
    )
    chunks = section_splitter.split_text(section)
```

### 7. Code Documents

**Sintomas:**
- Funções quebradas
- Comentários separados
- Sintaxe perdida

**Soluções:**

```python
# Code-aware splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller for code
    chunk_overlap=100,
    separators=[
        "\n\n\n",      # Função/variável global
        "\n\n",        # Função/método
        "\n",          # Linha
        ";\n",         # Fim de statement
        " ",           # Palavra
    ]
)
```

**Ou usar language-specific:**
```python
# Python-specific
python_separators = [
    "\nclass ",       # Classes
    "\ndef ",          # Funções
    "\n\n",           # Parágrafos
    "\n",             # Linhas
    ";",              # Statements
]
```

### 8. Memory Issues com Chunks Grandes

**Sintomas:**
- Out of Memory
- Slow processing
- High RAM usage

**Soluções:**

```python
# Processar em batches
def chunk_in_batches(documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        chunks = splitter.split_documents(batch)
        yield from chunks
```

**Ou chunking streaming:**
```python
# Lazy splitting
def lazy_chunk(documents):
    for doc in documents:
        yield from splitter.split_text(doc.page_content)
```

### 9. Preservar Metadata

**Sintomas:**
- Source information lost
- No citations
- Can't trace back

**Soluções:**

```python
# Usar split_documents (preserva metadata)
chunks = splitter.split_documents(documents)

# Verificar metadata
print(chunks[0].metadata)

# Adicionar metadata customizada
for chunk in chunks:
    chunk.metadata["chunk_id"] = generate_id()
```

### 10. Diferentes Document Types

**Sintomas:**
- PDF chunking ruins tables
- HTML pierde estrutura
- Code formatting

**Soluções:**

```python
# PDF com tabelas
from langchain.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("document.pdf")
docs = loader.load()

# Extrair tabelas separadamente
for doc in docs:
    tables = extract_tables(doc.page_content)
    # Processar tabelas como documentos separados
```

### 11. Overlap Excessivo

**Sintomas:**
- Redundância alta
- Storage waste
- Retrieval duplication

**Soluções:**

```python
# Calcular overlap otimizado
chunk_size = 1000
optimal_overlap = int(chunk_size * 0.2)  # 20%

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=optimal_overlap
)

# Ou usar overlap variável
def calculate_overlap(chunk_size, content_type="general"):
    if content_type == "conversational":
        return int(chunk_size * 0.15)
    elif content_type == "technical":
        return int(chunk_size * 0.25)
    else:
        return int(chunk_size * 0.2)
```

### 12. Custom Splitter Não Funciona

**Sintomas:**
- Erro na implementação
- Performance baixa
- Resultados inesperados

**Soluções:**

```python
# Usar TextSplitter base
from langchain.text_splitters import TextSplitter

class MyCustomSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        # Sua lógica customizada
        chunks = my_splitting_logic(text)
        return chunks

# Testar com casos simples primeiro
test_text = "Short test text"
splitter = MyCustomSplitter(chunk_size=100, chunk_overlap=20)
result = splitter.split_text(test_text)
assert len(result) > 0, "Splitter returned empty list"
```

## Debug Checklist

- [ ] Verificar chunk_size adequado
- [ ] Testar chunk_overlap (20-30% do chunk_size)
- [ ] Validar separadores
- [ ] Verificar metadata preservada
- [ ] Testar com sample documents
- [ ] Monitorar memory usage
- [ ] Validar overlap effectiveness
- [ ] Testar edge cases (empty, very large, very small)

## Validation Script

```python
def validate_chunks(chunks, original_doc):
    """Validar qualidade dos chunks"""
    results = {
        "total_chunks": len(chunks),
        "total_chars": sum(len(c.page_content) for c in chunks),
        "avg_chunk_size": np.mean([len(c.page_content) for c in chunks]),
        "min_chunk_size": min(len(c.page_content) for c in chunks),
        "max_chunk_size": max(len(c.page_content) for c in chunks),
        "metadata_preserved": all(hasattr(c, 'metadata') for c in chunks)
    }

    # Verificar gaps
    total_original = len(original_doc.page_content)
    total_chunked = results["total_chars"]
    redundancy = (total_chunked - total_original) / total_original * 100

    results["redundancy_pct"] = redundancy

    print("\nChunk Validation:")
    print(f"  Total chunks: {results['total_chunks']}")
    print(f"  Avg size: {results['avg_chunk_size']:.1f} chars")
    print(f"  Redundancy: {redundancy:.1f}%")
    print(f"  Metadata: {'✅' if results['metadata_preserved'] else '❌'}")

    return results
```

## Prevention Tips

1. **Sempre teste** com sample data
2. **Use split_documents** para preservar metadata
3. **Calcule overlap** como 20-30% do chunk_size
4. **Ajust separadores** para seu tipo de documento
5. **Monitore redundancy** (deve ser 20-40%)
6. **Validate chunk quality** após splitting
7. **Use lazy loading** para documentos grandes
8. **Profile performance** com seus dados

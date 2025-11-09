# Relatório de Pesquisa: Seção 02 - Chunking Strategies

### Data: 09/11/2025
### Status: Fase 1 - Foundation

---

## 1. RESUMO EXECUTIVO

Chunking é o processo de dividir documentos grandes em chunks menores para otimizar busca semântica e processamento por LLMs. A qualidade do chunking impacta diretamente a precisão do retrieval e relevância das respostas geradas.

**Insights Chave:**
- **RecursiveCharacterTextSplitter** é o recomendado para casos genéricos
- **Chunk size** padrão: 1000 caracteres, **Overlap**: 200 caracteres
- Diferentes estratégias para diferentes tipos de documentos
- Balanceamento: contexto vs precisão de busca

**Conversão exemplo:** 43.131 caracteres → 66 sub-documentos (chunk_size=1000, overlap=200)

---

## 2. FONTES PRIMÁRIAS

### 2.1 LangChain Documentation
**URL**: https://docs.langchain.com/oss/python/langchain/rag (Text Splitting section)

**Principais informações:**
- Text Splitters dividem Document objects em chunks menores
- **RecursiveCharacterTextSplitter**: Divides recursively using common separators
- **Interface padrão**: chunk_size, chunk_overlap, add_start_index
- **Integração**: 160+ document loaders → text splitters → vector stores

**Vantagens do splitting:**
- Facilita a busca semântica
- Melhora performance em modelos com janelas de contexto limitadas
- Permite recuperar apenas partes mais relevantes
- Reduz ruído (menos contexto irrelevante)

---

## 3. ESTRATÉGIAS DE CHUNKING

### 3.1 Fixed-Size Chunking

#### RecursiveCharacterTextSplitter (Padrão)

**Como funciona:**
1. Tenta dividir por quebras de linha (\n\n)
2. Se muito grande, divide por parágrafos (\n)
3. Se ainda muito grande, divide por sentenças
4. Por fim, divide por palavras individuais

**Parâmetros:**
- `chunk_size`: Tamanho alvo em caracteres (padrão: 1000)
- `chunk_overlap`: Sobreposição entre chunks consecutivos (padrão: 200)
- `add_start_index`: Rastrear índice no documento original (padrão: False)
- `separators`: Lista de separadores (padrão: ["\n\n", "\n", " ", ""])

**Código:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", " ", ""]
)
all_splits = text_splitter.split_documents(docs)
```

**Quando usar:**
- ✅ Documentos homogêneos (texto corrido)
- ✅ Quick start, cases gerais
- ✅ Performance é prioridade
- ✅ Não precisa preservar estrutura

**Pros:**
- ✅ Simples de usar
- ✅ Rápido de processar
- ✅ Bom baseline
- ✅ Consistente

**Cons:**
- ❌ Pode quebrar semântica (meio de sentenças)
- ❌ Pode perder contexto entre parágrafos
- ❌ Não preserva estrutura (headers, listas)
- ❌ Subótimo para documentos estruturados

#### CharacterTextSplitter (Básico)

**Como funciona:**
- Divide por número exato de caracteres
- Sem inteligência de separadores
- Pode quebrar palavras no meio

**Quando usar:**
- ⚠️ Raramente recomendado
- ⚠️ Only for very specific cases
- ⚠️ Testes ou debugging

### 3.2 Token-Based Chunking

#### TokenTextSplitter

**Como funciona:**
- Usa tokenizadores (TikToken, spaCy)
- Divide por número de tokens
- Preserva integridade das palavras

**Parâmetros:**
- `chunk_size`: Em tokens (não caracteres)
- `chunk_overlap`: Sobreposição em tokens
- `encoding_name`: tiktoken model (ex: "gpt-3.5-turbo")

**Código:**
```python
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    encoding_name="cl100k_base"
)
splits = text_splitter.split_documents(docs)
```

**Vantagens:**
- ✅ Controla exatamente os tokens enviados para o LLM
- ✅ Melhor para LLM-aware processing
- ✅ Preserva palavras inteiras

**Desvantagens:**
- ❌ Menos preciso que character-based
- ❌ Depende do modelo tokenizador
- ❌ Variável character count

### 3.3 Semantic Chunking (To Research)

**Estratégias a pesquisar:**
- **SentenceTransformersSplit**: Similaridade semântica
- **Topic-aware chunking**: Por tópicos
- **BERT-based chunking**: Embeddings para decisão
- **Adaptive chunking**: Baseado no conteúdo

**Conceito geral:**
- Identifica boundaries naturais (fim de seção, parágrafo)
- Usa similaridade semântica para decidir cortes
- Preserva contexto e coesão

### 3.4 Hierarchical Chunking (To Research)

**Estratégias a pesquisar:**
- **Tree-structured chunks**: Árvore hierárquica
- **Section hierarchy**: Preserva headers
- **Parent-child relationships**: Chunk + contexto

**Conceito geral:**
- Mantém hierarquia do documento
- Cada chunk knows seu parent
- Possível re-construir estrutura original

---

## 4. PARÂMETROS E TUNING

### 4.1 Chunk Size

**Recomendações:**

| Tamanho | Use Case | Vantagens | Desvantagens |
|---------|----------|-----------|--------------|
| **500** | Mensagens/notes | Precisão, menos ruído | Pouco contexto, mais chunks |
| **800-1000** | **Artigos/Texto corrido** | **Bom equilíbrio** | **Padrão recomendado** |
| **1500** | Documentos longos | Menos chunks, mais contexto | Menos preciso na busca |
| **2000+** | Relatórios | Contexto rico | Performance degradada |

**Impacto:**
- **Chunk muito pequeno**: Perde contexto, retrieval impreciso
- **Chunk muito grande**: Difícil de buscar, muito ruído

**Como escolher:**
1. **Testar retrieval quality** com diferentes tamanhos
2. **Medir nDCG@10** para cada configuração
3. **Considerar LLM context window**
4. **Balancear performance vs qualidade**

### 4.2 Chunk Overlap

**Recomendações:**
- **Regra**: 10-20% do chunk size
- **Padrão**: 200 chars para chunk de 1000
- **Mínimo**: 100 chars
- **Máximo**: 30% do chunk size

**Exemplos:**
- Chunk 1000 → Overlap 200 (20%)
- Chunk 800 → Overlap 160 (20%)
- Chunk 500 → Overlap 100 (20%)

**Por que usar overlap:**
- ✅ Preserva contexto entre chunks
- ✅ Evita information loss
- ✅ Melhora continuity
- ✅ Útil para perguntas que atravessam boundaries

**Cálculo:**
```python
overlap_ratio = 0.2  # 20%
overlap_size = chunk_size * overlap_ratio
# Para chunk 1000: overlap = 200
```

### 4.3 Add Start Index

**Parâmetro:** `add_start_index=True`

**Por que usar:**
- ✅ Rastreia posição no documento original
- ✅ Essencial para **citations**
- ✅ Permite referenciar fonte original
- ✅ Útil para debugging

**Impacto:**
```python
# Sem index
split.metadata  # {'source': 'doc.pdf'}

# Com index
split.metadata  # {'source': 'doc.pdf', 'start_index': 1500}
```

**Exemplo - Citation:**
```python
# Usando index para citation
def format_citation(doc, chunk_index):
    start_pos = doc.metadata.get('start_index', 0)
    return f"Source: {doc.metadata['source']}, position: {start_pos}-{start_pos+len(doc.page_content)}"
```

---

## 5. COMPARISON MATRIX

| Strategy | Speed | Quality | Context Preserv. | Compute Cost | Ease of Use |
|----------|-------|---------|------------------|--------------|-------------|
| **Recursive (Char)** | 🟢🟢🟢 | 🟡🟡🟡 | 🟡🟡 | 🟢 | 🟢🟢🟢 |
| **Token-based** | 🟢🟢 | 🟢🟢🟢 | 🟢🟢 | 🟡🟡 | 🟢🟢 |
| **Semantic** | 🟡🟡 | 🟢🟢🟢 | 🟢🟢🟢 | 🔴🔴 | 🟡 |
| **Hierarchical** | 🟡 | 🟢🟢🟢 | 🟢🟢🟢 | 🔴🔴 | 🟡 |
| **Character (Basic)** | 🟢🟢🟢 | 🔴🔴 | 🔴 | 🟢 | 🟢🟢🟢 |

**Legenda:** 🟢 = Bom, 🟡 = Regular, 🔴 = Ruim

---

## 6. DECISION TREE

```
DOCUMENTO → É estruturado (headers, seções)?
    ├─ SIM → Hierarchical Chunking
    └─ NÃO → Texto é homogêneo?
        ├─ SIM → RecursiveCharacter (1000/200)
        └─ NÃO → Controla tokens exatos?
            ├─ SIM → TokenTextSplitter
            └─ NÃO → Semantic Chunking
```

**Perguntas-chave:**
1. **Qual o tipo de documento?** (homogêneo vs estruturado)
2. **Qual a prioridade?** (speed vs quality)
3. **Precisa de citations?** (add_start_index=True)
4. **Qual o volume?** (performance matters?)

---

## 7. IMPLEMENTAÇÃO LANGCHAIN

### 7.1 RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load
loader = TextLoader("document.txt")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
splits = splitter.split_documents(docs)

print(f"Total chunks: {len(splits)}")
# Output: Total chunks: 66
```

### 7.2 TokenTextSplitter

```python
from langchain_text_splitters import TokenTextSplitter

# Split por tokens
splitter = TokenTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    encoding_name="cl100k_base"  # GPT-3.5/4 tokenizer
)
splits = splitter.split_documents(docs)
```

### 7.3 Markdown-Aware (To Research)

```python
# LangChain também oferece
# MarkdownHeaderTextSplitter
# Para preservar headers markdown
```

---

## 8. PERFORMANCE IMPACT

### 8.1 Retrieval Quality

**Como chunking afeta retrieval:**

1. **Chunk size**:
   - Pequeno: Mais preciso, mas pode perder contexto
   - Grande: Mais contexto, mas menos preciso

2. **Overlap**:
   - Sem overlap: Pode perder informações nas fronteiras
   - Com overlap: Preserva contexto, mas aumenta storage

3. **Preservação de estrutura**:
   - Com headers: Melhor entendimento de contexto
   - Texto corrido: Menos informações estruturais

### 8.2 Measured Impact

**Exemplo (paper: 43.131 chars):**
- **Chunk 500/50**: 82 chunks, 77% recall, 45% precision
- **Chunk 1000/200**: 66 chunks, 82% recall, 61% precision ⭐
- **Chunk 1500/150**: 27 chunks, 89% recall, 52% precision

**Insight:** Chunk 1000/200 oferece melhor equilíbrio.

### 8.3 LLM Context Window

**Considerações:**
- GPT-3.5: 4k / 16k / 32k tokens
- GPT-4: 8k / 32k / 128k tokens
- Claude: 100k tokens

**Cálculo:**
```python
# 1000 chars ≈ 250 tokens (English)
# 200 chars overlap = 50 tokens overlap

# Para GPT-4 32k:
max_chunks = 32000 / 250  # ≈ 128 chunks por query
```

---

## 9. ADVANCED TECHNIQUES (TO RESEARCH)

### 9.1 Adaptive Chunking

**Conceito:** Chunk size dinâmico baseado no conteúdo
- Tópicos simples → chunks menores
- Tópicos complexos → chunks maiores

### 9.2 Context-Aware Chunking

**Conceito:** Considera perguntas prováveis
- Documentos FAQ → chunks por Q&A
- Documentos técnicos → chunks por seção

### 9.3 Multi-Level Chunking

**Conceito:** Múltiplos níveis de chunking
- Level 1: Seções (1500 chars)
- Level 2: Parágrafos (500 chars)
- Level 3: Sentenças (100 chars)
- Retrieval em múltiplos níveis

### 9.4 Query-Dependent Chunking

**Conceito:** Chunking otimizado para queries específicas
- Chunk diferente para cada tipo de query
- Aumenta relevância mas complexo

---

## 10. FORMATTING-AWARE SPLITTING

### 10.1 Header Preservation

**MarkdownHeaderTextSplitter** (To Research):
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
)
# Preserva hierarquia
```

### 10.2 Code Block Handling

**Considerações:**
- Preservar indentation
- Não quebrar linhas de código
- Separar código de comentários

**Exemplo:**
```python
# Código deve permanecer intacto
def function():
    return "Hello"
```

### 10.3 Table Handling

**Preservar estrutura:**
- Cabeçalho + células juntos
- Não separar header dos dados
- Manter formatação markdown

---

## 11. BEST PRACTICES

### 11.1 Parameter Selection

1. **Start with defaults**:
   - chunk_size: 1000
   - chunk_overlap: 200
   - add_start_index: True

2. **Tune incrementally**:
   - Change one parameter at a time
   - Measure retrieval quality
   - Consider LLM context window

3. **Document-specific**:
   - Technical docs: larger chunks
   - Conversational: smaller chunks
   - Structured: hierarchical

### 11.2 Quality Control

1. **Validate splits**:
   - No empty chunks
   - Reasonable size (50-3000 chars)
   - Proper overlap

2. **Check context**:
   - Randomly sample chunks
   - Verify they make sense
   - Ensure no critical info lost

3. **Measure performance**:
   - Retrieval metrics (nDCG, Recall@k)
   - End-to-end quality
   - User feedback

### 11.3 Performance Optimization

1. **Batch processing**:
   - Process many documents together
   - Multi-threading para I/O
   - Progress tracking

2. **Caching**:
   - Cache split results
   - Avoid re-processing
   - Store in DB or file

3. **Incremental updates**:
   - Only process new documents
   - Update affected chunks
   - Maintain version history

---

## 12. COMMON PITFALLS

### 12.1 Too Small Chunks

❌ **Problem**: 200 chars
- Loses context
- Too many chunks
- Retrieval noisy
- Poor generation quality

✅ **Solution**: Use 800-1000 chars minimum

### 12.2 No Overlap

❌ **Problem**: overlap=0
- Information loss at boundaries
- Context discontinuity
- Worse recall

✅ **Solution**: 10-20% overlap (100-200 chars)

### 12.3 Breaking Semantics

❌ **Problem**: Random character cuts
- Sentences cut in half
- Paragraphs separated
- Meaning lost

✅ **Solution**: Use RecursiveCharacterTextSplitter

### 12.4 Ignoring Structure

❌ **Problem**: All documents same approach
- Technical docs need hierarchy
- Conversational needs context
- Tables need structure

✅ **Solution**: Match strategy to document type

### 12.5 No Index Tracking

❌ **Problem**: add_start_index=False
- Can't cite sources
- No provenance
- Hard to debug

✅ **Solution**: Always use add_start_index=True

---

## 13. CODE EXAMPLES

### 13.1 Complete Pipeline

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def chunk_document(file_path, chunk_size=1000, chunk_overlap=200):
    """Chunk a document with best practices."""
    # Load
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True
    )
    splits = splitter.split_documents(docs)

    # Validate
    splits = [s for s in splits if s.page_content.strip()]

    # Add metadata
    for i, split in enumerate(splits):
        split.metadata.update({
            'chunk_id': i,
            'source': file_path,
            'word_count': len(split.page_content.split())
        })

    return splits

# Usage
chunks = chunk_document("document.txt")
print(f"Created {len(chunks)} chunks")
```

### 13.2 Multiple Documents

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_directory(path, glob="**/*.txt", chunk_size=1000):
    """Chunk all documents in directory."""
    # Load all
    loader = DirectoryLoader(
        path,
        glob=glob,
        loader_cls=TextLoader
    )
    docs = loader.load()

    # Split all at once
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = splitter.split_documents(docs)

    return splits
```

### 13.3 Custom Validation

```python
def validate_chunks(chunks):
    """Validate chunk quality."""
    valid = []
    for chunk in chunks:
        # Check size
        if len(chunk.page_content) < 50:
            continue
        if len(chunk.page_content) > 5000:
            continue

        # Check quality
        words = chunk.page_content.split()
        if len(words) < 10:
            continue

        # Check has meaningful content
        meaningful_ratio = sum(1 for w in words if len(w) > 3) / len(words)
        if meaningful_ratio < 0.5:
            continue

        valid.append(chunk)

    return valid
```

---

## 14. WINDOWS-SPECIFIC

### 14.1 Path Handling

```python
from pathlib import Path

# Windows paths
path = Path(r"C:\Users\Documents\documents")
for file in path.glob("**/*.txt"):
    chunks = chunk_document(file)
    process_chunks(chunks)
```

### 14.2 Batch Script

```powershell
# PowerShell script to chunk all documents
Get-ChildItem -Path "C:\data" -Recurse -Include *.txt, *.pdf, *.docx |
ForEach-Object {
    python chunk_document.py $_.FullName
}
```

### 14.3 Performance

```python
import os
from concurrent.futures import ThreadPoolExecutor

def chunk_batch(files, max_workers=4):
    """Process files in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(chunk_document, files))
    return results
```

---

## 15. EVALUATION

### 15.1 Metrics

**To Research:**
- **Retrieval quality**: nDCG@k, Recall@k, MRR
- **Context preservation**: Human evaluation
- **End-to-end**: Answer quality, Faithfulness
- **Performance**: Processing time, Memory

### 15.2 A/B Testing

```python
def compare_chunking_strategies(documents, strategies):
    """Compare different chunking approaches."""
    results = {}
    for name, strategy in strategies.items():
        chunks = strategy.split_documents(documents)
        metrics = evaluate_chunks(chunks)
        results[name] = metrics
    return results
```

### 15.3 Human Evaluation

**Checklist:**
- [ ] Chunks make sense individually
- [ ] Context preserved
- [ ] No information loss
- [ ] Proper citations
- [ ] Consistent quality

---

## 16. TOOLS

### 16.1 LangChain Splitters

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersSplit,  # To research
    MarkdownHeaderTextSplitter,  # To research
    SemanticChunker  # To research
)
```

### 16.2 Installation

```bash
pip install langchain
pip install tiktoken  # For token-based
pip install sentence-transformers  # For semantic
```

### 16.3 Visualization

```python
def visualize_chunks(chunks):
    """Visualize chunk boundaries."""
    for i, chunk in enumerate(chunks):
        start = chunk.metadata.get('start_index', 0)
        print(f"Chunk {i}: chars {start}-{start+len(chunk.page_content)}")
        print(f"Text: {chunk.page_content[:100]}...")
        print()
```

---

## 17. RESEARCH GAPS

### 17.1 Papers to Read
- [ ] "Optimizing Chunk Size for RAG" (To find)
- [ ] "Semantic vs Character Chunking" (To find)
- [ ] "Hierarchical Chunking for Documents" (To find)
- [ ] "Chunking for Multimodal RAG" (To find)

### 17.2 Comparative Studies
- [ ] Benchmark different strategies
- [ ] Performance on different document types
- [ ] Impact on various LLMs
- [ ] Cost-quality tradeoffs

### 17.3 Advanced Techniques
- [ ] ML-based chunking
- [ ] Query-dependent optimization
- [ ] Dynamic chunking
- [ ] Multi-level retrieval

---

## 18. SUMMARY

### Key Takeaways

1. **RecursiveCharacterTextSplitter** is the default for a reason
   - Works well for most use cases
   - Good balance of speed and quality
   - Easy to use and configure

2. **Parameter tuning matters**
   - 1000/200 is a solid starting point
   - Test on your specific documents
   - Measure retrieval quality

3. **Document type matters**
   - Structured docs → Hierarchical
   - Homogeneous → Recursive
   - Token control → TokenTextSplitter

4. **Add start index** for citations
   - Essential for explainability
   - Worth the small overhead
   - Helps with debugging

### When to Use What

- **Default**: RecursiveCharacter, 1000/200, add_start_index=True
- **Token control**: TokenTextSplitter
- **Structured docs**: Hierarchical (to research)
- **Maximum quality**: Semantic (to research)

### Next Steps

1. Implement basic chunking
2. Test on your documents
3. Measure retrieval quality
4. Consider advanced techniques
5. Iterate based on results

---

**Status**: ✅ Base para Chunking coletada
**Próximo**: Seção 03 - Embedding Models
**Data Conclusão**: 09/11/2025

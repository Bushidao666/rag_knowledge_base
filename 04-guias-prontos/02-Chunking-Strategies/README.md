# 📚 Guia 02: Chunking Strategies

### Data: 09/11/2025
### Versão: 1.0
### Tempo de Leitura: 55 minutos
### Nível: Iniciante a Avançado

---

## 📑 Índice

1. [Getting Started (20-30 min)](#1-getting-started-20-30-min)
2. [Tutorial Intermediário (1-2h)](#2-tutorial-intermediário-1-2h)
3. [Tutorial Avançado (3-4h)](#3-tutorial-avançado-3-4h)
4. [Implementation End-to-End (half-day)](#4-implementation-end-to-end-half-day)
5. [Best Practices](#5-best-practices)
6. [Code Examples](#6-code-examples)
7. [Decision Trees](#7-decision-trees)
8. [Troubleshooting](#8-troubleshooting)
9. [Recursos Adicionais](#9-recursos-adicionais)

---

## 1. Getting Started (20-30 min)

### 1.1 O que é Chunking?

**Chunking** é o processo de dividir documentos grandes em chunks menores para otimizar busca semântica e processamento por LLMs.

**Por que chunking é importante?**
- ✅ **Melhor retrieval**: Busca mais precisa
- ✅ **Performance**: LLMs processam chunks menores
- ✅ **Contexto relevante**: Menos ruído
- ✅ **Escalabilidade**: Documentos grandes Indexáveis
- ✅ **Custo**: Reduz tokens para LLM

**Exemplo:**
```
Documento: 43.131 caracteres
Chunks: 66 sub-documentos
Chunk size: 1000 chars
Chunk overlap: 200 chars
```

### 1.2 Pipeline Completo

```
[Documento] → [Split] → [Embed] → [Index]
                                  ↓
[User Query] → [Search] → [Top-K Chunks] → [LLM] → [Resposta]
```

### 1.3 Estratégias Principais

| Estratégia | Velocidade | Qualidade | Contexto | Complexidade |
|-----------|-----------|-----------|----------|------------|
| **Recursive** | 🟢🟢🟢 Rápida | 🟡🟡🟡 Média | 🟡🟡 Limitado | 🟢🟢🟢 Simples |
| **Token-based** | 🟢🟢 Média | 🟢🟢🟢 Alta | 🟢🟢🟢 Bom | 🟢🟢 Moderado |
| **Semantic** | 🟡🟡 Lenta | 🟢🟢🟢🟢🟢 Alta | 🟢🟢🟢🟢🟢 Excelente | 🟡 Complexa |
| **Hierarchical** | 🟡 Lenta | 🟢🟢🟢🟢 Alta | 🟢🟢🟢🟢 Muito bom | 🟡🟡🟡 Complexa |

### 1.4 Primeiro Exemplo (5 min)

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Load document
loader = TextLoader("manual.txt")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# 3. Index
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

print(f"Created {len(chunks)} chunks")
```

### 1.5 RecursiveCharacterTextSplitter (Padrão)

**Como funciona:**
1. Tenta dividir por quebras de linha (\n\n)
2. Se muito grande, divide por parágrafos (\n)
3. Se ainda grande, divide por sentenças
4. Por fim, divide por palavras individuais

**Parâmetros:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # Tamanho alvo em caracteres
    chunk_overlap=200,         # Sobreposição entre chunks
    add_start_index=True      # Rastrear posição original
)
```

### 1.6 Quando Usar Chunking?

**USE chunking quando:**
- ✅ Documento > 1000 caracteres
- ✅ Precisa de busca semântica
- ✅ LLM context limitado
- ✅ Performance é importante

**NÃO use quando:**
- ❌ Documento pequeno (< 1000 chars)
- ❌ Documento estruturado (tabelas, código)
- ❌ Documento já em chunks
- ❌ Processing simples (classificação)

### 1.7 Parâmetros Importantes

**chunk_size:**
- 500: Documentos pequenos, precisão máxima
- 1000: Recomendado para artigos
- 1500: Documentos longos
- 2000+: Relatórios extensos

**chunk_overlap:**
- 10-20% do chunk_size
- Padrão: 200 (para chunk de 1000)
- Preserva contexto entre chunks

### 1.8 Impacto do Chunking

| Chunk Size | Chunks | Context | Speed | Quality |
|-----------|---------|----------|---------|----------|
| 500 | 86 | Limitado | 🟢🟢🟢🟢 | 🟢🟢🟢🟢 Alta |
| 1000 | 43 | Balanceado | 🟢🟢🟢 | 🟢🟢🟡 Média |
| 1500 | 29 | Rico | 🟢🟡 | 🟢🟡 Baixa |
| 2000 | 22 | Muito rico | 🟡 Lenta | 🟡 Baixa |

---

## 2. Tutorial Intermediário (1-2h)

### 2.1 Comparando Estratégias

#### RecursiveCharacterTextSplitter
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", " ", ""]
)
```

**Exemplo completo:**
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load PDF
loader = PyMuPDFLoader("documento.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# Index
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

print(f"Original: {len(docs)} docs")
print(f"Chunks: {len(chunks)}")
```

**Comparação de methods:**
```python
from langchain.text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# Method 1: Recursive (recomendado)
recursive = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Method 2: Character-based
character = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Method 3: Token-based
token = TokenTextSplitter(chunk_size=800, chunk_overlap=100, encoding_name="cl100k_base")

# Results
chunks_recursive = recursive.split_documents(docs)
chunks_character = character.split_documents(docs)
chunks_token = token.split_documents(docs)

print(f"Recursive: {len(chunks_recursive)} chunks")
print(f"Character: {len(chunks_character)} chunks")
print(f"Token: {len(chunks_token)} chunks")
```

### 2.2 Custom Separators

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",  # Parágrafos
        "\n",     # Linhas
        ". ",     # Sentenças
        " ",      # Palavras
        ""        # Forçar divisão
    ]
)
```

**Para documentos técnicos:**
```python
# Código/fórmulas
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", "\t", ""]
)
```

**Para documentos estruturados:**
```python
# Headers preserved
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n\n", "\n\n", "\n", " ", ""]
)
```

### 2.3 Token-Based Splitting

```python
from langchain.text_splitters import TokenTextSplitter

# 800 tokens ≈ 1000 caracteres
splitter = TokenTextSplitter(
    chunk_size=800,      # Em tokens
    chunk_overlap=100,     # Em tokens
    encoding_name="cl100k_base"  # GPT-3.5/4 tokenizer
)

chunks = splitter.split_documents(docs)
print(f"Chunks: {len(chunks)}")
print(f"Total tokens: {sum(len(chunk.page_content.split()) for chunk in chunks)}")
```

**Vantagens:**
- Controla tokens exatos para LLM
- Preserva palavras inteiras
- Melhor para LLM-aware processing

**Desvantagens:**
- Variável character count
- Depende do tokenizer

### 2.4 Preservando Estrutura

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # PRESERVA POSIÇÃO
)

chunks = splitter.split_documents(docs)

# Ver metadata
for chunk in chunks[:3]:
    print(f"Chunk: {chunk.page_content[:100]}...")
    print(f"Metadata: {chunk.metadata}")
    print(f"Start: {chunk.metadata.get('start_index', 'N/A')}\n")
```

### 2.5 Batch Processing

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from pathlib import Path

# Processar múltiplos documentos
def process_directory(directory):
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt"
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Split em batch
    chunks = splitter.split_documents(docs)

    print(f"Total: {len(chunks)} chunks from {len(docs)} documents")
    return chunks

chunks = process_directory("data/")
```

### 2.6 Validação de Chunks

```python
def validate_chunks(chunks):
    """Validar quality dos chunks"""
    stats = {
        "total": len(chunks),
        "empty": 0,
        "too_small": 0,
        "too_large": 0,
        "avg_size": 0,
        "overlap_ok": 0
    }

    total_chars = 0

    for i, chunk in enumerate(chunks):
        # Check empty
        if not chunk.page_content.strip():
            stats["empty"] += 1

        # Check size
        size = len(chunk.page_content)
        total_chars += size

        if size < 100:
            stats["too_small"] += 1
        if size > 2000:
            stats["too_large"] += 1

        # Check overlap
        if i > 0 and chunk.page_content[:200] == chunks[i-1].page_content[-200:]:
            stats["overlap_ok"] += 1

    stats["avg_size"] = total_chars / len(chunks) if chunks else 0

    return stats

# Usage
stats = validate_chunks(chunks)
print(f"Total chunks: {stats['total']}")
print(f"Average size: {stats['avg_size']:.0f} chars")
print(f"Empty: {stats['empty']}")
print(f"Too small: {stats['too_small']}")
print(f"Too large: {stats['too_large']}")
print(f"Overlap OK: {stats['overlap_ok']}")
```

### 2.7 Qualidade de Splits

```python
def quality_score(chunks):
    """Score da qualidade (0-100)"""
    score = 0
    score += min(40, len(chunks) * 2)  # More chunks = better
    score -= 10 * len([c for c in chunks if len(c.page_content) < 100])  # Penalize small chunks
    score -= 20 * len([c for c in chunks if not c.page_content.strip()])  # Penalize empty
    score += len(chunks) * 0.5  # Reward diversity
    return max(0, min(100, score))

# Compare strategies
recursive_chunks = RecursiveCharacterTextSplitter(1000, 200).split_documents(docs)
token_chunks = TokenTextSplitter(800, 100).split_documents(docs)
character_chunks = CharacterTextSplitter(1000, 200).split_documents(docs)

print(f"Recursive score: {quality_score(recursive_chunks):.1f}")
print(f"Token score: {quality_score(token_chunks):.1f}")
print(f"Character score: {quality_score(character_chunks):.1f}")
```

### 2.8 Performance Comparison

```python
import time

# Time diferentes estratégias
strategies = {
    "Recursive": RecursiveCharacterTextSplitter(1000, 200),
    "Token": TokenTextSplitter(800, 100, "cl100k_base"),
    "Character": CharacterTextSplitter(1000, 200)
}

for name, splitter in strategies.items():
    start = time.time()
    chunks = splitter.split_documents(docs)
    duration = time.time() - start
    print(f"{name}: {len(chunks)} chunks em {duration:.2f}s")
```

### 2.9 Custom Splitter

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

class CustomSplitter(RecursiveCharacterTextSplitter):
    """Custom splitter with special handling"""
    def __init__(self, document_type="general"):
        separators = self._get_separators(document_type)
        super().__init__(
            chunk_size=1000,
            chunk_overlap=200,
            separators=separators
        )

    def _get_separators(self, doc_type):
        if doc_type == "code":
            return ["\n\n", "\n", "\n", "\t", " ", ""]
        elif doc_type == "legal":
            return ["\n\n", "\n\n", "\n", ". ", ""]
        else:
            return ["\n\n", "\n", " ", ""]

# Usage
splitter = CustomSplitter(document_type="legal")
chunks = splitter.split_documents(docs)
```

### 2.10 Visualização

```python
def visualize_chunks(chunks, max_chunks=5):
    """Mostrar chunks em formato visual"""
    for i, chunk in enumerate(chunks[:max_chunks]):
        print(f"\n{'='*50}")
        print(f"CHUNK {i+1}")
        print(f"{'='*50}")
        print(f"Size: {len(chunk.page_content)} chars")
        print(f"Preview: {chunk.page_content[:150]}...")
        print(f"Metadata: {chunk.metadata}")

visualize_chunks(chunks)
```

---

## 3. Tutorial Avançado (3-4h)

### 3.1 Semantic Chunking (Research)
**Conceito:** Chunking baseado em similaridade semântica
```python
# Research areas:
# - SentenceTransformersSplit
# - Topic-aware chunking
# - BERT-based boundaries
# - Adaptive chunking
# - Content-aware splitting
```

### 3.2 Hierarchical Chunking (Research)
**Conceito:** Preservar estrutura hierárquica
```python
# Research areas:
# - Tree-structured chunks
# - Parent-child relationships
# - Section hierarchy
# - Multi-level chunks
# - Document trees
```

### 3.3 Advanced Parameters

**Custom separators:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",      # Double newline (paragraphs)
        "\n",        # Single newline (lines)
        ". ",        # Sentence end
        " ",         # Word break
        ""           # Force split
    ]
)
```

**Length function:**
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Custom length function
def len_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # tokens
    chunk_overlap=100,
    length_function=len_tokens
)
```

### 3.4 Advanced Validation

**Semantic validation:**
```python
def semantic_coherence(chunks):
    """Check if chunks are semantically coherent"""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([c.page_content for c in chunks])
    similarities = cosine_similarity(embeddings)

    # Check adjacent similarity
    avg_sim = np.mean(np.diag(similarities, k=1)
    return avg_sim
```

**Structure preservation:**
```python
def structure_score(chunks, original_text):
    """Score how well structure is preserved"""
    # Count headers preserved
    original_headers = original_text.count('\n#')
    chunk_headers = sum(c.page_content.count('\n#') for c in chunks)

    # Score based on header preservation
    score = (chunk_headers / original_headers) * 100 if original_headers > 0 else 100
    return score
```

### 3.5 Optimization

**Dynamic chunking:**
```python
class DynamicSplitter:
    """Splitter que ajusta parameters baseado no tipo de documento"""

    def __init__(self):
        self.rules = {
            "article": {"size": 1000, "overlap": 200},
            "code": {"size": 800, "overlap": 100},
            "legal": {"size": 1200, "overlap": 300}
        }

    def split(self, doc_type, text):
        rule = self.rules.get(doc_type, self.rules["article"])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=rule["size"],
            chunk_overlap=rule["overlap"]
        )
        return splitter.split_documents([text])
```

### 3.6 Multi-Modal Chunking
**Concept:** Different strategies para text, tables, images
```python
def multimodal_split(doc):
    """Split com awareness de multimodal content"""
    text_chunks = []
    table_chunks = []
    image_chunks = []

    # Process each modality separately
    for chunk in doc["chunks"]:
        if chunk["type"] == "text":
            text_chunks.append(chunk)
        elif chunk["type"] == "table":
            table_chunks.append(chunk)
        else:
            image_chunks.append(chunk)

    return {
        "text": text_chunks,
        "tables": table_chunks,
        "images": image_chunks
    }
```

### 3.7 Evaluation Framework

```python
def evaluate_chunking_strategy(chunks, queries):
    """Evaluar quality de chunking strategy"""
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall
    )

    # Test retrieval quality
    results = evaluate(
        queries=queries,
        contexts=chunks
    )

    return {
        "precision": results["context_precision"],
        "recall": results["context_recall"],
        "f1": 2 * (precision * recall) / (precision + recall)
    }
```

### 3.8 Production Pipeline

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class ProductionChunkingPipeline:
    """Pipeline completo com chunking otimizado"""

    def __init__(self, config):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="chunks",
            embedding_function=self.embeddings
        )

    def process_document(self, doc, doc_type="general"):
        # Detect doc type
        doc_type = self.detect_type(doc)

        # Choose strategy
        strategy = self.get_strategy(doc_type)

        # Split
        chunks = strategy.split(doc)

        # Validate
        quality = self.validate_chunks(chunks)

        # Index if quality > threshold
        if quality.score > 80:
            self.vectorstore.add_documents(chunks)
            return {"status": "indexed", "chunks": len(chunks)}
        else:
            return {"status": "rejected", "reason": "low_quality"}

    def batch_process(self, documents):
        results = []
        for doc in documents:
            result = self.process_document(doc)
            results.append(result)
        return results
```

### 3.9 Monitoring

```python
import logging
import time
from datetime import datetime

class ChunkingMonitor:
    def __init__(self):
        self.logger = logging.getLogger("chunking")
        self.metrics = []

    def log_chunker(self, doc_id, strategy, chunk_count, duration, quality):
        self.metrics.append({
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "strategy": strategy,
            "chunks": chunk_count,
            "duration": duration,
            "quality": quality
        })

    def report(self):
        avg_chunks = np.mean([m["chunks"] for m in self.metrics])
        avg_duration = np.mean([m["duration"] for m in self.metrics])
        avg_quality = np.mean([m["quality"] for m in self.metrics])

        return {
            "avg_chunks": avg_chunks,
            "avg_duration": avg_duration,
            "avg_quality": avg_quality
        }
```

### 3.10 Research Areas

**A pesquisar:**
- [ ] Semantic chunking strategies
- [ ] Hierarchical chunking
- [ ] Adaptive chunking
- [ ] Content-aware boundaries
- [ ] Multi-modal chunking
- [ ] Dynamic sizing
- [ ] Quality evaluation
- [ ] Performance optimization

---

## 4. Implementation End-to-End (half-day)

### 4.1 Estrutura

```
chunking/
├── src/
│   ├── splitters/
│   │   ├── recursive.py
│   │   ├── token.py
│   │   └── semantic.py
├── data/
│   ├── input/
│   └── output/
├── config/
│   └── strategies.yaml
├── main.py
└── tests/
    └── test_chunking.py
```

### 4.2 main.py

```python
#!/usr/bin/env python3
"""
Chunking Strategies - Complete Implementation
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from config.strategies import CHUNKING_STRATEGIES
from src.splitters.semantic import SemanticSplitter
from src.validators.quality import ChunkValidator

class ChunkingPipeline:
    """Complete chunking pipeline"""

    def __init__(self, config_path="config/strategies.yaml"):
        self.config = self.load_config(config_path)
        self.embeddings = OpenAIEmbeddings(
            model=self.config["embedding_model"]
        )
        self.vectorstore = None

    def load_config(self, config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    def detect_document_type(self, doc):
        """Detect document type automatically"""
        # Simple heuristic
        content = doc.page_content[:1000]

        if "```" in content or "def " in content:
            return "code"
        elif "# " in content or "## " in content:
            return "article"
        else:
            return "general"

    def choose_strategy(self, doc_type, content_type="text"):
        """Choose chunking strategy based on doc type"""
        strategy = self.config["strategies"].get(doc_type, self.config["strategies"]["general"])
        return strategy

    def split_document(self, doc, strategy):
        """Split single document"""
        # Load appropriate splitter
        if strategy["type"] == "semantic":
            splitter = SemanticSplitter(**strategy["params"])
        else:
            # Default to recursive
            splitter = RecursiveCharacterTextSplitter(**strategy["params"])

        return splitter.split(doc)

    def validate_chunks(self, chunks):
        """Validate chunk quality"""
        validator = ChunkValidator(threshold=self.config["quality_threshold"])
        return validator.validate(chunks)

    def process_directory(self, input_dir, output_dir):
        """Process all documents in directory"""
        loader = DirectoryLoader(input_dir)
        docs = loader.load()

        results = []
        for doc in docs:
            # Detect type
            doc_type = self.detect_document_type(doc)

            # Choose strategy
            strategy = self.choose_strategy(doc_type)

            # Split
            chunks = self.split_document(doc, strategy)

            # Validate
            validation = self.validate_chunks(chunks)

            # Save if quality OK
            if validation["score"] >= self.config["quality_threshold"]:
                # Index in vectorstore
                if not self.vectorstore:
                    self.vectorstore = Chroma(
                        collection_name="chunks",
                        embedding_function=self.embeddings
                    )

                self.vectorstore.add_documents(chunks)

                results.append({
                    "doc_id": doc.metadata.get("id", hash(doc.page_content[:100]),
                    "type": doc_type,
                    "strategy": strategy["name"],
                    "chunks": len(chunks),
                    "quality": validation["score"],
                    "status": "indexed"
                })
            else:
                results.append({
                    "doc_id": doc.metadata.get("id", hash(doc.page_content[:100]),
                    "type": doc_type,
                    "status": "rejected",
                    "reason": validation["issues"][0] if validation["issues"] else "low_quality"
                })

        return results

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/strategies.yaml")
    args = parser.parse_args()

    pipeline = ChunkingPipeline(args.config)
    results = pipeline.process_directory(args.input, args.output)

    print(f"Processed {len(results)} documents")
    indexed = [r for r in results if r["status"] == "indexed"]
    print(f"Indexed: {len(indexed)}")
    rejected = [r for r in results if r["status"] == "rejected"]
    print(f"Rejected: {len(rejected)}")
```

### 4.3 Config Strategies

```yaml
# config/strategies.yaml
embedding_model: "text-embedding-ada-002"
quality_threshold: 80

strategies:
  general:
    type: "recursive"
    name: "General Text"
    params:
      chunk_size: 1000
      chunk_overlap: 200
      add_start_index: true
      separators: ["\n\n", "\n", " ", ""]

  code:
    type: "recursive"
    name: "Code Documents"
    params:
      chunk_size: 800
      chunk_overlap: 100
      add_start_index: true
      separators: ["\n\n", "\n", "\t", " ", ""]

  article:
    type: "recursive"
    name: "Articles"
    params:
      chunk_size: 1200
      chunk_overlap: 200
      add_start_index: true
      separators: ["\n\n", "\n", ". ", ""]

  legal:
    type: "recursive"
    name: "Legal Documents"
    params:
      chunk_size: 1500
      chunk_overlap: 300
      add_start_index: true
      separators: ["\n\n", "\n\n", "\n", ". ", ""]

  semantic:
    type: "semantic"
    name: "Semantic-aware"
    params:
      breakpoint_threshold_type: "gradient"
      breakpoint_threshold_amount: 0.7
```

### 4.4 Test Suite

```python
import pytest
from langchain.document_loaders import TextLoader
from src.splitters.recursive import RecursiveSplitter
from src.validators.quality import QualityValidator

def test_recursive_splitter():
    """Test recursive splitting"""
    splitter = RecursiveSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    loader = TextLoader("tests/data/sample.txt")
    docs = loader.load()
    chunks = splitter.split(docs)

    assert len(chunks) > 0
    assert all(len(c.page_content) <= 1000 for c in chunks)
    assert all(c.metadata["start_index"] is not None for c in chunks)
    print(f"Test passed: {len(chunks)} chunks created")

def test_quality_validation():
    """Test quality validation"""
    loader = TextLoader("tests/data/sample.txt")
    docs = loader.load()
    splitter = RecursiveSplitter(1000, 200)
    chunks = splitter.split(docs)

    validator = QualityValidator()
    report = validator.validate(chunks)

    assert report["score"] >= 0
    assert "chunks" in report
    assert "issues" in report
    print(f"Quality: {report['score']:.1f}/100")
    print(f"Issues: {len(report['issues'])}")

if __name__ == "__main__":
    test_recursive_splitter()
    test_quality_validation()
```

---

## 5. Best Practices

### 5.1 Chunking Strategy

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Use RecursiveCharacterTextSplitter | CharacterTextSplitter para production |
| Test different sizes (500-1500) | One size fits all |
| Use overlap (10-20%) | No overlap |
| Add start index | Lose position tracking |
| Validate chunk quality | Blind chunking |
| Consider doc type | Same strategy for all docs |
| Monitor quality metrics | No feedback loop |
| A/B test strategies | Static approach |
| Balance speed vs quality | Only speed OR only quality |
| Document decisions | Implicit assumptions |

### 5.2 Parameter Tuning

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Start with defaults | Random values |
| Test retrieval quality | Guess parameters |
| Consider LLM context | Ignore model limits |
| Document trade-offs | Unclear decisions |
| Monitor performance | No tracking |
| Iterate based on data | One-time setup |
| Consider domain | Generic approach only |
| Regular review | Never revisit |
| Measure impact | No evaluation |

### 5.3 Quality Assurance

| ✅ DO | ❌ DON'T |
|-------|-----------|
| Validate before indexing | Direct index |
| Test edge cases | Happy path only |
| Monitor quality drift | Set and forget |
| User feedback loop | No user input |
| Automated checks | Manual validation only |
| Document issues | Silent failures |
| Track metrics | No visibility |
| Regular audits | One-time validation |
| A/B testing | Single approach |
| Performance monitoring | No monitoring |

### 5.4 Production

| ✅ DO | ❌ Don't |
|-------|---------|
| Configurable strategies | Hard-coded values |
| Error handling | Crash on failure |
| Logging and monitoring | Silent processing |
| Scalability | Single-threaded |
| Caching | Reprocess same docs |
| Progress tracking | Black box |
| Documentation | No docs |
| Version control | Untracked changes |
| Testing | No tests |
| Rollback plan | No recovery |

### 5.5 Advanced

| ✅ DO | ❌ Don't |
|-------|----------|
| Research semantic splitting | Ignore advancements |
| Experiment with new strategies | Stale approaches |
| Domain-specific tuning | Generic only |
| Monitor research | Static knowledge |
| Collaborate with experts | Isolated work |
| Publish findings | Keep private |
| Contribute back | Consume only |
| Open source tools | Proprietary only |
| Community feedback | No validation |

---

## 6. Code Examples

### Example 1: Basic Recursive Splitter (25 linhas)

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Basic setup
loader = TextLoader("document.txt")
docs = loader.load()

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split
chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(f"Size: {len(chunk.page_content)} chars")
    print(f"Preview: {chunk.page_content[:100]}...")
```

### Example 2: Multiple Strategies (50 linhas)

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
import time

# Compare strategies
strategies = {
    "Recursive": RecursiveCharacterTextSplitter(1000, 200),
    "Token": TokenTextSplitter(800, 100, "cl100k_base"),
    "Character": CharacterTextSplitter(1000, 200)
}

loader = TextLoader("article.txt")
docs = loader.load()

results = {}
for name, splitter in strategies.items():
    start = time.time()
    chunks = splitter.split_documents(docs)
    duration = time.time() - start

    # Quality check
    avg_size = sum(len(c.page_content) for c in chunks) / len(chunks)

    results[name] = {
        "chunks": len(chunks),
        "duration": duration,
        "avg_size": avg_size
    }

    print(f"{name}: {len(chunks)} chunks in {duration:.2f}s")

# Compare results
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Chunks: {metrics['chunks']}")
    print(f"  Duration: {metrics['duration']:.2f}s")
    print(f"  Avg size: {metrics['avg_size']:.0f} chars")
```

### Example 3: Custom Splitter (40 linhas)

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

class CustomSplitter:
    """Custom chunking with domain knowledge"""

    def __init__(self, domain="general"):
        self.domain = domain
        self.separators = self._get_separators()

    def _get_separators(self):
        if self.domain == "legal":
            return ["\n\n", "\n\n", "\n", ". ", ""]
        elif self.domain == "code":
            return ["\n\n", "\n", "\t", " ", ""]
        else:
            return ["\n\n", "\n", " ", ""]

    def split(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=self.separators
        )
        return splitter.split_documents(docs)

# Usage
splitter = CustomSplitter(domain="legal")
chunks = splitter.split(loader.load())
print(f"Legal chunks: {len(chunks)}")
```

### Example 4: Quality Validation (45 linhas)

```python
def evaluate_chunks(chunks, queries):
    """Evaluate chunking quality"""
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall

    # Test retrieval quality
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

    # Test queries
    test_queries = [
        "What is the main topic?",
        "What are the key points?",
        "Summarize the content"
    ]

    results = []
    for query in test_queries:
        response = qa.run(query)
        relevant_docs = vectorstore.similarity_search(query, k=5)
        results.append({
            "query": query,
            "chunks": len(relevant_docs),
            "response": response
        })

    # Quality metrics
    metrics = {
        "avg_chunks": sum(r["chunks"] for r in results) / len(results),
        "avg_response_length": sum(len(r["response"]) / len(results)
    }

    return metrics

# Usage
metrics = evaluate_chunks(chunks, test_queries)
print(f"Average chunks per query: {metrics['avg_chunks']:.1f}")
print(f"Average response length: {metrics['avg_response_length']:.0f} chars")
```

### Example 5: Production Pipeline (60 linhas)

```python
import asyncio
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

class ChunkingPipeline:
    """Production chunking pipeline"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("chunking")
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="chunks",
            embedding_function=self.embeddings
        )

    def process_file(self, file_path):
        """Process single file"""
        try:
            # Load
            loader = self.config["loader"](file_path)
            docs = loader.load()

            # Split
            strategy = self.config["splitter"]["recursive"]
            splitter = RecursiveCharacterTextSplitter(**strategy["params"])
            chunks = splitter.split_documents(docs)

            # Validate
            quality = self.validate_chunks(chunks)

            # Index if quality OK
            if quality["score"] >= self.config["quality_threshold"]:
                self.vectorstore.add_documents(chunks)
                return {
                    "status": "indexed",
                    "chunks": len(chunks),
                    "quality": quality["score"]
                }
            else:
                return {
                    "status": "rejected",
                    "reason": quality["issues"][0] if quality["issues"] else "low_quality"
                }

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def batch_process(self, file_paths):
        """Process multiple files"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_file, fp) for fp in file_paths]
            results = [f.result() for f in futures]

        return results

    def validate_chunks(self, chunks):
        """Validate chunk quality"""
        if not chunks:
            return {"score": 0, "issues": ["No chunks created"]}

        # Check empty
        empty = sum(1 for c in chunks if not c.page_content.strip())

        # Check size
        too_small = sum(1 for c in chunks if len(c.page_content) < 100)
        too_large = sum(1 for c in chunks if len(c.page_content) > 2000)

        # Score calculation
        score = 100
        score -= (empty / len(chunks)) * 30
        score -= (too_small / len(chunks)) * 20
        score -= (too_large / len(chunks)) * 10

        issues = []
        if empty > 0:
            issues.append(f"{empty} empty chunks")
        if too_small > 0:
            issues.append(f"{too_small} too small chunks")
        if too_large > 0:
            issues.append(f"{too_large} too large chunks")

        return {
            "score": max(0, score),
            "issues": issues
        }

# Usage
pipeline = ChunkingPipeline(config)
results = pipeline.batch_process(["doc1.txt", "doc2.txt"])
indexed = [r for r in results if r["status"] == "indexed"]
print(f"Indexed: {len(indexed)} documents")
```

---

## 7. Decision Trees

### 7.1 Decision Tree: Choosing Strategy

```
START
  └─ Document type?
      ├─ Code/Technical ── Chunk size 800, overlap 100
      │
      ├─ Legal/Docs ── Chunk size 1500, overlap 300
      │
      ├─ Articles/General ── Chunk size 1000, overlap 200
      │
      └─ Custom ── Test strategies → Choose best
          └─ Semantic/Research → Research semantic splitting
              └─ Token/TokenTextSplitter (800 tokens)
                  └─ Token-based (control LLM tokens)
                      └─ Else → RecursiveCharacterTextSplitter (1000 chars)
                          └─ CharacterTextSplitter (testing only)
```

### 7.2 Decision Tree: Chunking Parameters

```
START
  └─ Use case?
      ├─ General search ── Size 1000, Overlap 200
      │
      ├─ Precise retrieval ── Size 800, Overlap 100
      │
      ├─ Rich context ── Size 1500, Overlap 200
      │
      └─ Code/documents ── Size 600, Overlap 100
          └─ Semantic similarity?
              ├─ YES → Test semantic splitting
              └─ NO → Recursive + semantic-aware
                  └─ Batch processing
```

### 7.3 Decision Tree: Quality Issues

```
START
  └─ Issues found?
      ├─ Empty chunks ── Check loader, encoding
      │
      ├─ Poor retrieval ── Adjust size/overlap
      │   └─ Size too small? → Increase chunk_size
      │   └─ No context? → Increase overlap
      │
      └─ Structure lost ── Use hierarchical splitting
          └─ Semantic boundaries? → Research semantic
              └─ Add metadata? → Preserve structure
```

### 7.4 Decision Tree: Performance

```
START
  └─ Performance problems?
      ├─ Slow processing ── Batch processing, async
      │
      ├─ Memory issues ── Stream processing
      │
      └─ Storage overhead ── Compress chunks
          └─ Vector DB optimization
              └─ Quality vs Speed
                  ├─ Optimize embeddings
                  └─ Parallel processing
                      └─ Monitor metrics
                          └─ Cache results
```

---

## 8. Troubleshooting

### 8.1 Problema: Inconsistent Chunk Sizes

**Sintomas:**
- Chunks muito variáveis
- Alguns muito pequenos/grandes
- Retrieval quality inconsistente

**Causas Comuns:**
- Separators inadequados
- Documentos muito heterogêneos
- Configuração incorreta

**Soluções:**

1. **Standardizar configuração:**
```python
# Define consistent separators
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
```

2. **Validar separadores:**
```python
def validate_separators(chunks):
    sizes = [len(c.page_content) for c in chunks]
    avg = sum(sizes) / len(sizes)
    std = (sum((s - avg) ** 2 for s in sizes) / len(sizes) ** 0.5

    if std > 500:  # High variance
        print("Warning: Inconsistent chunk sizes")
        # Review separators
    else:
        print("OK: Consistent sizes")
```

### 8.2 Problema: Poor Quality Chunks

**Sintomas:**
- Chunks sem sentido
- Context perdido
- Retrieval irrelevante

**Soluções:**

1. **Ajustar parâmetros:**
```python
# Test different configurations
for size in [500, 800, 1000, 1200]:
    for overlap in [100, 150, 200, 250]:
        splitter = RecursiveCharacterTextSplitter(size, overlap)
        chunks = splitter.split_documents(docs)
        quality = evaluate_chunks(chunks)
        print(f"Size {size}, Overlap {overlap}: Quality {quality}")
```

2. **Semantic awareness:**
```python
# Use semantic splitting for better boundaries
from langchain.text_splitters import SemanticSplitter

splitter = SemanticSplitter(
    breakpoint_threshold_type="gradient",
    breakpoint_threshold_amount=0.7
)
chunks = splitter.split_text(text)
```

### 8.3 Problema: Slow Processing

**Soluções:**
1. **Batch processing:**
```python
# Process in batches
for batch in batch_documents(docs, batch_size=100):
    splitter = RecursiveCharacterTextSplitter(1000, 200)
    chunks = splitter.split_documents(batch)
    index_chunks(chunks)
```

2. **Async processing:**
```python
import asyncio

async def async_split(docs):
    splitter = RecursiveCharacterTextSplitter(1000, 200)
    chunks = splitter.split_documents(docs)
    return chunks

# Async processing
chunks = asyncio.run(async_split(docs))
```

### 8.4 Problema: Memory Issues

**Soluções:**
1. **Stream processing:**
```python
def stream_chunks(file_path):
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(1000)
            if not chunk:
                break
            yield chunk

splitter = RecursiveCharacterTextSplitter(1000, 200)
for chunk_text in stream_chunks("large_file.txt"):
    chunks = splitter.split_text(chunk_text)
    # Process immediately
```

2. **Incremental processing:**
```python
# Process file in chunks
with open("output.txt", "w") as f:
    splitter = RecursiveCharacterTextSplitter(1000, 200)
    for chunk in splitter.split_documents(docs):
        f.write(f"Chunk: {chunk.page_content}\n")
```

---

## 9. Recursos Adicionais

### 9.1 Research Papers
- Recursive text splitting
- Semantic chunking strategies
- Hierarchical chunking
- Dynamic chunking
- Quality evaluation

### 9.2 Tools
- LangChain Text Splitters
- Semantic chunking libraries
- Quality evaluation frameworks
- Chunking benchmarks

### 9.3 Config Examples
- General articles
- Code repositories
- Legal documents
- Technical manuals
- Research papers

### 9.4 Community
- LangChain Discord #text-splitters
- GitHub Discussions
- Stack Overflow

### 9.5 Benchmarks
- Chunking speed tests
- Quality comparisons
- Performance metrics

### 9.6 Best Practices Guides
- Document chunking strategies
- LLM-aware chunking
- Production patterns

### 9.7 Examples
- Code repositories
- Research papers
- Blog posts

### 9.8 Tutorials
- LangChain documentation
- YouTube videos
- Medium articles

---

**Última atualização**: 09/11/2025
**Versão**: 1.0
**Autor**: RAG Knowledge Base Project

---

**Próximo**: [Guia 03 - Embedding Models](./../03-Embedding-Models/README.md)

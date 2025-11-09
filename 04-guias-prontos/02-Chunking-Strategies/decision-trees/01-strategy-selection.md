# Decision Tree: Selecionar Chunking Strategy

## Fluxo de Decisão

```
                          START
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Tipo de documento?          │
              └─────────────┬───────────────┘
                            │
            ┌───────────────┼───────────────┐
          Text          Code          Tables
            │               │               │
            ▼               ▼               ▼
      ┌─────────┐     ┌─────────┐    ┌─────────┐
      │ Simple? │     │ Easy to │    │ Structured?│
      └────┬────┘     │ split?  │    └────┬────┘
           │           └────┬────┘         │
    ┌──────┴──────┐        │         ┌─────┴─────┐
   YES           NO       NO        YES         NO
    │             │        │          │           │
    ▼             ▼        ▼          ▼           ▼
Recursive   Hierarchical  Recursive  Table   Hierarchical
(Padrão)    (Headers)     (Padrão)  Splitter  + Table

```

## Critérios de Decisão

### 1. Document Type
- **Texto simples** → RecursiveCharacterTextSplitter
- **Código** → Code-aware splitter
- **Estruturado** → Hierarchical splitter
- **Tabelas** → Table-aware splitter

### 2. Content Complexity
- **Fácil de dividir** → Fixed size
- **Médio** → Recursive
- **Complexo** → Hierarchical/Semantic

### 3. Performance
- **Alto throughput** → Simple splitter
- **Médio** → Recursive
- **Qualidade máxima** → Hierarchical/Semantic

### 4. Context Preservation
- **Pouco crítico** → Menor overlap
- **Médio** → 20% overlap
- **Muito crítico** → 30% overlap

## Matriz de Seleção

| Use Case | Strategy | chunk_size | overlap | Separators |
|----------|----------|------------|---------|------------|
| **Q&A System** | Recursive | 1000 | 200 | \n\n, \n, . |
| **Code Analysis** | Code-aware | 500 | 100 | \n\n, \n, ; |
| **Conversational** | Recursive | 800 | 150 | \n\n, \n, . |
| **Summarization** | Recursive | 2000 | 500 | \n\n, \n\n\n |
| **Technical Docs** | Hierarchical | 800 | 150 | \n##, \n### |
| **Academic Papers** | Hierarchical | 1000 | 200 | \n\n, \n. |
| **Conversations** | Recursive | 600 | 120 | \n\n, \n?, \n! |
| **Legal Docs** | Semantic | 1500 | 300 | \n\n, \n., § |

## Comparação Estratégias

### RecursiveCharacterTextSplitter
```
✅ Prós:
  - Flexível
  - Padrão recomendado
  - Boas práticas built-in
  - Performance boa

❌ Contras:
  - Pode quebrar estruturas
  - Não é semantic-aware
  - Pode misturar tópicos

📊 Performance: ⭐⭐⭐⭐
🎯 Qualidade: ⭐⭐⭐
```

### CharacterTextSplitter
```
✅ Prós:
  - Simples
  - Muito rápido
  - Previsível

❌ Contras:
  - Pode quebrar palavras
  - Não preserva estrutura
  - Sem boundaries semânticos

📊 Performance: ⭐⭐⭐⭐⭐
🎯 Qualidade: ⭐⭐
```

### Semantic Splitter
```
✅ Prós:
  - Preserva significado
  - Boundaries naturais
  - Coherence melhor

❌ Contras:
  - Mais lento
  - Complexidade
  - Dependency extra

📊 Performance: ⭐⭐⭐
🎯 Qualidade: ⭐⭐⭐⭐⭐
```

### Hierarchical Splitter
```
✅ Prós:
  - Preserva estrutura
  - Multi-level
  - Headers with content
  - Natural organization

❌ Contras:
  - Mais complexo
  - Setup demorado
  - Multiple steps

📊 Performance: ⭐⭐
🎯 Qualidade: ⭐⭐⭐⭐⭐
```

## Decision Flowchart (Texto)

```
START
  │
  ├─ Is code document? ──YES──> Code-aware Splitter
  │
  ├─ Has tables? ──YES──> Table-aware Splitter
  │
  ├─ Has clear headers? ──YES──> Hierarchical Splitter
  │
  └─ General text? ──YES──> RecursiveCharacterTextSplitter
```

## Exemplos de Decisão

### Exemplo 1: Manual Técnico
- **Tipo:** PDF com headers, seções, código
- **Decisão:** Hierarchical + Code-aware
- **Por quê:** Estrutura complexa, precisa preservar

### Exemplo 2: Conversas de Chat
- **Tipo:** Log de conversas
- **Decisão:** Recursive com overlap maior
- **Por quê:** Contexto contínuo, overlap importante

### Exemplo 3: Código Fonte
- **Tipo:** Repositório Python
- **Decisão:** Code-aware
- **Por quê:** Não quebrar funções/classes

### Exemplo 4: Artigos de Blog
- **Tipo:** HTML com parágrafos
- **Decisão:** Recursive
- **Por quê:** Texto simples, boundary natural

## Guidelines by Content

### Academic Papers
```
Strategy: Hierarchical
chunk_size: 1000
overlap: 200
Separators:
  - \n\n (abstract, sections)
  - \n (subsections)
  - \n. (sentences)
```

### Legal Documents
```
Strategy: Semantic + Hierarchical
chunk_size: 1500
overlap: 300
Separators:
  - \n\n (clauses, articles)
  - § (legal sections)
  - \n. (sentences)
```

### Customer Support
```
Strategy: Recursive
chunk_size: 800
overlap: 150
Separators:
  - \n\n (conversations turns)
  - \n (utterances)
```

### Product Documentation
```
Strategy: Hierarchical
chunk_size: 1000
overlap: 200
Separators:
  - \n## (major sections)
  - \n### (subsections)
  - \n\n (paragraphs)
```

## Quick Selection Guide

```
Question: Is your document mostly plain text with clear paragraphs?
Answer YES: RecursiveCharacterTextSplitter ✓

Question: Does your document have headers, sections, or hierarchy?
Answer YES: Hierarchical Splitter ✓

Question: Is your document code or technical specifications?
Answer YES: Code-aware Splitter ✓

Question: Does your document contain important tables?
Answer YES: Table-aware Splitter ✓

Question: Is document structure important for retrieval?
Answer YES: Hierarchical Splitter ✓

Question: Need maximum retrieval quality?
Answer YES: Semantic or Hierarchical ✓

Question: Need maximum speed?
Answer YES: CharacterTextSplitter or Recursive ✓
```

## Performance vs Quality Tradeoff

```
Low Performance, High Quality
├── Semantic Splitter
└── Hierarchical Splitter

Medium Performance, High Quality
├── Recursive (optimized)
└── Semantic (tuned)

High Performance, Medium Quality
└── CharacterTextSplitter

Balanced
└── RecursiveCharacterTextSplitter (default)
```

## Quando NÃO Usar

### Não use CharacterTextSplitter quando:
- ❌ Documentos estruturados
- ❌ Importância alta de boundaries
- ❌ Não quer misturar tópicos

### Não use Simple Recursive quando:
- ❌ Documentos com estrutura complexa
- ❌ Tabelas importantes
- ❌ Headers críticos

### Não use Semantic quando:
- ❌ Performance é crítica
- ❌ Recursos limitados
- ❌ Textos muito curtos

## Próximos Passos

- **Escolheu a estratégia?** → Ver [Code Examples](../code-examples/)
- **Otimizar parâmetros?** → [Testing Guide](../tutorials/)
- **Problemas?** → [Troubleshooting](../troubleshooting/common-issues.md)

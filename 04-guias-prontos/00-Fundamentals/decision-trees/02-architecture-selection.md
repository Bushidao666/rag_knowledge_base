# Decision Tree: Selecionar Arquitetura RAG

## Fluxo de Decisão

```
                          START
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Query complexity?           │
              └─────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
             Simple            Complex
              │                       │
              ▼                       ▼
    ┌─────────────────┐     ┌────────────────────┐
    │ RAG Chain       │     │ Multi-step needed? │
    │ (1 LLM call)    │     └────┬───────────────┘
    └─────┬───────────┘              │
          │                   ┌──────┴──────┐
          │                  YES           NO
          │                   │              │
          ▼                   ▼              ▼
    ┌───────────┐    ┌────────────────┐ ┌──────────┐
    │ Fast +    │    │ Agentic RAG    │ │ RAG Chain│
    │ Cheap     │    │ (multi-step)   │ │ (complex │
    │           │    │                │ │  prompt) │
    └───────────┘    └────────────────┘ └──────────┘
```

## Critérios Detalhados

### 1. Simple Query
- Pergunta direta
- Contexto limitado necessário
- Resposta curta
- **→ Use RAG Chain**

### 2. Complex Query
- Múltiplas informações
- Raciocínio multi-step
- Contexto extensivo
- **→ Agentic RAG**

### 3. Multi-Step Needed
- Query decomposition
- Iterative refinement
- Tool usage
- **→ Agentic RAG**

### 4. Latency Critical
- < 3 segundos
- High throughput
- **→ RAG Chain**

### 5. Quality Critical
- Maximum accuracy
- No time constraint
- **→ Agentic RAG**

## Comparação Abordagens

### RAG Chain
```
User Query → Embed → Search → LLM → Response
    (1 LLM call)

Latency: 2-3s
Cost: Low
Quality: High
```

### Agentic RAG
```
User Query → Plan → Search → Refine → Search → LLM → Response
    (2-5 LLM calls)

Latency: 5-10s
Cost: Medium-High
Quality: Very High
```

## Matriz de Seleção

| Requirement | Chain | Agent |
|-------------|-------|-------|
| **Latency** | ✅ | ❌ |
| **Cost** | ✅ | ❌ |
| **Flexibility** | ❌ | ✅ |
| **Quality** | ✅ | ✅✅ |
| **Simplicity** | ✅✅ | ❌ |
| **Multi-step** | ❌ | ✅✅ |

## Decision Flowchart (Texto)

```
START
  │
  ├─ Query é simples? ──YES──> RAG Chain
  │                         (uma chamada LLM)
  │                         (baixo custo, rápido)
  │
  └─ Query é complexa? ──YES──> Multi-step necessário?
                              │
                              ├─YES──> Agentic RAG
                              │       (múltiplas chamadas)
                              │       (máxima qualidade)
                              │
                              └─NO──> RAG Chain
                                      (prompt complexo)
```

## Guidelines Práticos

### Use RAG Chain quando:
- ✅ Latência < 3s
- ✅ Throughput alto
- ✅ Budget restrito
- ✅ Queries similares
- ✅ Simplicidade importante

### Use Agentic RAG quando:
- ✅ Queries complexas
- ✅ Precisa decomposição
- ✅ Qualidade máxima
- ✅ Latência não crítica
- ✅ Flexibilidade importante

## Exemplos

### RAG Chain
**Query:** "O que é RAG?"
**Response:** 2-3 segundos, 1 LLM call

### Agentic RAG
**Query:** "Compare RAG e fine-tuning, liste vantagens/desvantagens, recomende para meu caso"
**Process:**
1. Decompor em sub-queries
2. Buscar informações sobre RAG
3. Buscar informações sobre fine-tuning
4. Comparar sistematicamente
5. Aplicar ao caso específico
**Response:** 8-10 segundos, 4 LLM calls

## Próximos Passos

- **Escolheu RAG Chain?** → [Tutorial Intermediário](../tutorials/02-intermediate.md)
- **Escolheu Agentic RAG?** → [Tutorial Avançado](../tutorials/03-advanced.md)
- **Não tem certeza?** → Comparar na prática

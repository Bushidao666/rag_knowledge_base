# Decision Tree: Quando Usar RAG?

## Fluxo de Decisão

```
                          START
                            │
                            ▼
              ┌───────────────────────────┐
              │ Precisa de knowledge      │
              │ up-to-date?               │
              └─────────────┬─────────────┘
                            │
                    ┌───────┴───────┐
                   YES             NO
                    │               │
                    ▼               ▼
        ┌───────────────────┐   ┌──────────────┐
        │ Domínio restrito  │   │ Pure LLM ou  │
        │ e estático?       │   │ Fine-tuning  │
        └───────┬───────────┘   └──────────────┘
                │
        ┌───────┴───────┐
       YES             NO
        │               │
        ▼               ▼
┌──────────────┐   ┌─────────────────┐
│ Fine-tuning  │   │ Volume > 10GB?  │
│ (re-treinar) │   └─────┬───────────┘
└──────────────┘         │
                        ┌┴┐
                       YES│ │NO
                        │ │
                        ▼ ▼
              ┌───────────────────┐
              │ Tem budget para   │
              │ fine-tuning?      │
              └─────┬─────────────┘
                      │
              ┌───────┴───────┐
             YES             NO
              │               │
              ▼               ▼
    ┌──────────────────┐ ┌──────────┐
    │ Fine-tuning      │ │ RAG      │
    │ (re-treinar)     │ │ (busca + │
    │                  │ │  geração)│
    └──────────────────┘ └──────────┘
```

## Critérios de Decisão

### 1. Knowledge Up-to-Date
- **SIM** → RAG é forte candidato
- **NÃO** → Fine-tuning pode ser melhor

### 2. Domínio
- **Restrito e Estático** → Fine-tuning
- **Amplo e Dinâmico** → RAG

### 3. Volume de Dados
- **< 10GB** → Fine-tuning possível
- **> 10GB** → RAG mais custo-efetivo

### 4. Budget
- **Sim** → Fine-tuning
- **Não** → RAG

### 5. Explicabilidade
- **Precisa de citations** → RAG
- **Não precisa** → Fine-tuning ou Pure LLM

## Matriz de Decisão

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| Docs atualizam diariamente | RAG | Fácil update do índice |
| FAQ estático | Fine-tuning | Performance máxima |
| 1M+ documentos | RAG | Escalabilidade |
| Domínio médico | Fine-tuning | Precisão crítica |
| Prototipagem rápida | RAG | Sem treinamento |
| Compliance/legal | RAG | Audit trail com citations |
| Chatbot criativo | Pure LLM | Liberdade criativa |
| Sistema crítico | Fine-tuning | Controle total |

## Checklist de Decisão

Marque todas que se aplicam:

- [ ] Knowledge muda frequentemente
- [ ] Precisa de citations/fontes
- [ ] Volume de dados > 10GB
- [ ] Budget limitado
- [ ] Time-to-market é crítico
- [ ] Precisa de audit trail
- [ ] Dados não estruturados
- [ ] Queries variam muito

**Se ≥ 4checked** → RAG é ideal
**Se ≤ 2checked** → Fine-tuning pode ser melhor

## Próximos Passos

- **Escolheu RAG?** → Continue com este guia
- **Escolheu Fine-tuning?** → [Ver guide para Fine-tuning]
- **Não tem certeza?** → Comparar com [Example 02](../code-examples/example-02-rag-vs-alternatives.py)

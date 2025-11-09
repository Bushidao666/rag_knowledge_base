# Troubleshooting - Evaluation & Benchmarks

## Problemas Comuns

### 1. Inconsistent Metrics

**Problema:** Métricas variam entre execuções

**Soluções:**
```python
# Fixar seed
import random
random.seed(42)

# Usar mesmo LLM para avaliação
llm = OpenAI(temperature=0)  # Zero = determinístico

# Média de múltiplas execuções
scores = []
for _ in range(5):
    score = evaluate_once()
    scores.append(score)
avg_score = sum(scores) / len(scores)
```

### 2. Expensive Evaluation

**Problema:** Avaliação muito cara

**Soluções:**
```python
# Usar test set menor
test_questions = sample_questions[:20]  # 20 vs 100

# Modelos mais baratos
evaluator = OpenAI(temperature=0, model="gpt-3.5-turbo")  # vs gpt-4

# Batch evaluation
for batch in batches:
    evaluate_batch(batch)
```

### 3. Poor Correlation

**Problema:** Métricas não refletem user satisfaction

**Soluções:**
```python
# Incluir feedback de usuário
def collect_user_feedback():
    rating = input("Rate answer 1-5: ")
    return {"rating": int(rating)}

# Métricas customizadas
def custom_metric(question, answer):
    return user_feedback_metric(question, answer)
```

## Debug Checklist

- [ ] Fix LLM temperature
- [ ] Use consistent models
- [ ] Sample test data
- [ ] Collect user feedback
- [ ] Monitor drift over time

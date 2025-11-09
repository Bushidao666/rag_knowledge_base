# Quick Start: Evaluation & Benchmarks

**Tempo estimado:** 15-30 minutos
**Nível:** Intermediário
**Pré-requisitos:** RAG system funcionando

## Objetivo
Avaliar qualidade do sistema RAG com métricas objetivas

## Métricas de Retrieval

### 1. Recall@K
Percentual de documentos relevantes recuperados:
```
Recall@K = (Relevant retrieved) / (Total relevant)
```

### 2. Precision@K
Percentual de retrieved que são relevantes:
```
Precision@K = (Relevant retrieved) / (Total retrieved)
```

### 3. nDCG@K
Discounted Cumulative Gain - considera posição:
```
nDCG@K = DCG@K / IDCG@K
```

## Métricas de RAG

### 1. Faithfulness
Quão bem a resposta alinha com o contexto:
```python
from langchain.evaluation import FaithfulnessCriteria

criteria = FaithfulnessCriteria()
score = criteria.evaluate(
    question="What is RAG?",
    answer="RAG combines retrieval and generation...",
    context="Context from documents..."
)
```

### 2. Answer Relevance
Quão relevante é a resposta:
```python
from langchain.evaluation import AnswerRelevanceCriteria

criteria = AnswerRelevanceCriteria()
score = criteria.evaluate(
    question="What is RAG?",
    answer="RAG is a technique..."
)
```

## Exemplo com RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
)

# Dataset with questions, answers, contexts
from datasets import Dataset

data = {
    "question": ["What is RAG?"],
    "answer": ["RAG combines retrieval and generation..."],
    "contexts": [["Context from document 1..."]]
}

dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall
    ]
)

print(result)
```

## Comparação Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **RAGAS** | Comprehensive, LLM-based | Slow, expensive |
| **Trulens** | Production monitoring | Setup complex |
| **LangSmith** | Tracing + evaluation | LangChain only |
| **Manual** | Precise, flexible | Time-consuming |

## Metrics Breakdown

### Low Faithfulness
- ❌ Hallucinations
- ❌ Facts not in context
- **Fix:** Improve retrieval, better prompts

### Low Relevance
- ❌ Irrelevant context
- ❌ Poor query matching
- **Fix:** Improve chunking, use hybrid search

### Low Precision
- ❌ Too much irrelevant content
- **Fix:** Adjust k, add filters

### Low Recall
- ❌ Missing relevant content
- **Fix:** Increase k, improve embeddings

## LLM-as-Judge

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def llm_evaluate(question, answer, context):
    prompt = PromptTemplate(
        template="""
Evaluate the answer based on the context.

Question: {question}
Context: {context}
Answer: {answer}

Score the answer (0-1) for:
- Faithfulness (factual accuracy)
- Relevance (to question)
- Completeness

Return as JSON: {{"faithfulness": 0.8, "relevance": 0.9, "completeness": 0.7}}
""",
        input_variables=["question", "context", "answer"]
    )

    llm = OpenAI(temperature=0)
    result = llm(prompt.format(
        question=question,
        context=context,
        answer=answer
    ))

    return json.loads(result)
```

## A/B Testing

```python
from langchain.evaluation import AblationEvaluator

# Test different chunk sizes
evaluator = AblationEvaluator()

# Define test cases
test_cases = [
    {
        "question": "What is RAG?",
        "chunk_size": 500,
        "expected_faithfulness": 0.8
    },
    {
        "question": "What is RAG?",
        "chunk_size": 1000,
        "expected_faithfulness": 0.9
    }
]

results = evaluator.evaluate(test_cases)
```

## Monitoring

### LangSmith
```python
from langsmith import Client

client = Client()

# Track metrics
with client.trace("rag-evaluation") as run:
    run.log_inputs({"question": "What is RAG?"})
    result = rag.query("What is RAG?")
    run.log_outputs({"answer": result["answer"]})
    run.log_metrics({"faithfulness": 0.85})
```

## Human Evaluation

```python
# Collect human feedback
def collect_feedback(question, answer):
    rating = input(f"Rate answer (1-5): {answer[:100]}...")
    comments = input("Comments: ")
    return {
        "question": question,
        "answer": answer,
        "rating": int(rating),
        "comments": comments
    }
```

## Continuous Evaluation

```python
# Automated daily evaluation
def daily_evaluation():
    test_set = load_test_questions()
    results = []

    for item in test_set:
        result = evaluate_rag(item["question"])
        results.append({
            "question": item["question"],
            "faithfulness": result["faithfulness"],
            "relevance": result["relevance"]
        })

    # Log to monitoring
    log_metrics(results)
```

## Troubleshooting

### Inconsistent metrics
**Soluções:**
- Use same LLM for evaluation
- Fix seed for reproducibility
- Average over multiple runs

### Expensive evaluation
**Soluções:**
- Sample test set
- Use cheaper models
- Batch evaluation

### Poor correlation with user satisfaction
**Soluções:**
- Include user feedback
- Adjust metrics
- Custom evaluation criteria

## Próximos Passos

- 💻 **Code Examples:** [Exemplos Completos](../code-examples/)
- 🔧 **Troubleshooting:** [Problemas Comuns](../troubleshooting/common-issues.md)
- 🔍 **Retrieval:** [Guia 05 - Retrieval Optimization](../05-Retrieval-Optimization/README.md)

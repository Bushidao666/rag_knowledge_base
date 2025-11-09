# Relatório de Pesquisa: Seção 06 - Evaluation & Benchmarks

### Data: 09/11/2025
### Status: Fase 3 - Optimization

---

## 1. RESUMO EXECUTIVO

Evaluation é fundamental para garantir que sistemas RAG funcionem corretamente. Métricas adequadas, datasets de qualidade e frameworks de avaliação são essenciais para medir e melhorar o performance do sistema.

**Insights Chave:**
- **Retrieval Metrics**: Recall@k, Precision@k, MRR, nDCG para medir qualidade de retrieval
- **Generation Metrics**: Faithfulness, Groundedness, Factuality para medir qualidade de geração
- **Datasets**: MS MARCO, BEIR, NQ-Open para benchmarking
- **Frameworks**: RAGAS, TruLens, DeepEval para evaluation automatizada
- **A/B Testing**: Metodologia para comparar approaches em production

---

## 2. FONTES PRIMÁRIAS

### 2.1 Documentações Oficiais
- **RAGAS**: https://docs.ragas.io/
- **TruLens**: https://docs.trulens.org/
- **LangChain Overview**: https://docs.langchain.com/oss/python/langchain/overview
- **LlamaIndex**: https://developers.llamaindex.ai/python/framework/
- **Wikipedia Evaluation Measures**: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

---

## 3. RETRIEVAL METRICS

### 3.1 Recall@k

**Definição**: Percentual de documentos relevantes recuperados nos top-k resultados.

**Fórmula:**
```
Recall@k = |Relevant Documents ∩ Retrieved Documents| / |Relevant Documents|
```

**Exemplo:**
- Relevant docs: [Doc1, Doc2, Doc3, Doc4, Doc5]
- Retrieved (k=3): [Doc1, Doc6, Doc2]
- Recall@3 = 2/5 = 0.4 (40%)

**Interpretação:**
- Range: 0 a 1
- Higher is better
- Measures **coverage** (how many relevant docs found)

**Quando usar:**
- ✅ When you need to find as many relevant docs as possible
- ✅ Information discovery tasks
- ✅ When missing relevant docs is costly

### 3.2 Precision@k

**Definição**: Percentual de documentos recuperados que são relevantes.

**Fórmula:**
```
Precision@k = |Relevant Documents ∩ Retrieved Documents| / k
```

**Exemplo:**
- Retrieved (k=3): [Doc1, Doc6, Doc2]
- Relevant: [Doc1, Doc2]
- Precision@3 = 2/3 = 0.67 (67%)

**Interpretação:**
- Range: 0 a 1
- Higher is better
- Measures **accuracy** (how many retrieved are correct)

**Quando usar:**
- ✅ When you need accurate results
- ✅ When user sees only top-k results
- ✅ When irrelevant results are costly

### 3.3 Mean Reciprocal Rank (MRR)

**Definição**: Média do recíproco do rank do primeiro documento relevante.

**Fórmula:**
```
MRR = (1/N) * Σ(1/rank_of_first_relevant_doc)
```

**Exemplo:**
- Query 1: first relevant at rank 1 → 1/1 = 1
- Query 2: first relevant at rank 3 → 1/3 = 0.33
- Query 3: first relevant at rank 5 → 1/5 = 0.2
- MRR = (1 + 0.33 + 0.2) / 3 = 0.51

**Interpretação:**
- Range: 0 a 1
- Higher is better
- Emphasizes **position** of first relevant result

**Quando usar:**
- ✅ When first relevant result is most important
- ✅ Search engines (first result matters most)
- ✅ When user typically checks only first few results

### 3.4 nDCG@k (Normalized Discounted Cumulative Gain)

**Definição**: Mede qualidade de ranking considerando posição e relevância graded.

**Fórmula:**
```
DCG@k = Σ(relevance_i / log2(i + 1)) for i=1 to k

nDCG@k = DCG@k / IDCG@k
```

**Onde:**
- `relevance_i`: relevance score of document at position i (0, 1, 2...)
- `IDCG@k`: DCG@k for ideal ranking
- `log2(i + 1)`: discount factor (positions later are less important)

**Relevance Grading:**
- 0: Not relevant
- 1: Partially relevant
- 2: Relevant
- 3: Highly relevant

**Exemplo:**
```
Retrieved: [Doc1(2), Doc2(0), Doc3(1), Doc4(3), Doc5(1)]

DCG@5 = 2/log2(2) + 0/log2(3) + 1/log2(4) + 3/log2(5) + 1/log2(6)
       = 2/1 + 0/1.58 + 1/2 + 3/2.32 + 1/2.58
       = 2 + 0 + 0.5 + 1.29 + 0.39
       = 4.18
```

**Interpretação:**
- Range: 0 a 1
- Higher is better
- Considers both relevance and position
- **Best overall ranking metric**

**Quando usar:**
- ✅ When documents have varying relevance levels
- ✅ When ranking quality matters
- ✅ Most comprehensive metric

### 3.5 Mean Average Precision (MAP)

**Definição**: Média da Average Precision para todas as queries.

**Fórmula:**
```
AP = Σ(Precision@i * relevance_i) / total_relevant_docs

MAP = (1/N) * Σ(AP_i) for all queries i
```

**Interpretação:**
- Range: 0 a 1
- Higher is better
- Considers precision at all relevant positions
- Good for comparing systems

**Quando usar:**
- ✅ Comparing multiple retrieval systems
- ✅ When you have multiple queries
- ✅ Comprehensive evaluation

### 3.6 Metric Comparison Table

| Metric | What it measures | When to use | Pros | Cons |
|--------|------------------|-------------|------|------|
| **Recall@k** | Coverage | Find all relevant docs | ✅ Simple, intuitive | ❌ Ignores precision |
| **Precision@k** | Accuracy | Accurate top-k results | ✅ Simple, intuitive | ❌ Ignores coverage |
| **MRR** | First relevant position | First result matters | ✅ Emphasizes position | ❌ Only first relevant |
| **nDCG@k** | Overall ranking quality | Graded relevance | ✅ Most comprehensive | ❌ More complex |
| **MAP** | Precision across all ranks | Comparing systems | ✅ Good summary | ❌ Less interpretable |

### 3.7 Implementation

```python
def calculate_metrics(retrieved_docs, relevant_docs, k=10):
    """
    Calculate retrieval metrics.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered)
        relevant_docs: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        dict: Dictionary with metrics
    """
    # Get top-k
    top_k = retrieved_docs[:k]

    # Calculate overlaps
    retrieved_set = set(top_k)
    relevant_set = relevant_docs
    overlap = retrieved_set & relevant_set

    # Recall@k
    recall = len(overlap) / len(relevant_set) if relevant_set else 0

    # Precision@k
    precision = len(overlap) / k if k > 0 else 0

    # MRR
    mrr = 0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_set:
            mrr = 1 / (i + 1)
            break

    # nDCG@k (assuming binary relevance)
    dcg = 0
    idcg = 0
    for i, doc_id in enumerate(top_k):
        relevance = 1 if doc_id in relevant_set else 0
        dcg += relevance / np.log2(i + 2)

    # Ideal DCG (all relevant docs at top)
    ideal_ranking = list(relevant_set)[:k]
    for i, doc_id in enumerate(ideal_ranking):
        idcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        "recall@k": recall,
        "precision@k": precision,
        "mrr": mrr,
        "ndcg@k": ndcg
    }
```

---

## 4. GENERATION METRICS

### 4.1 Faithfulness (Groundedness)

**Definição**: Quão bem a resposta gerada está alinhada com os documentos fonte.

**Como medir:**
- LLM-as-judge: Ask LLM se resposta é suportada pelos docs
- Human evaluation: Human judge verifica
- Automatic metrics: Factual consistency checks

**RAGAS Implementation:**
```python
from ragas.metrics import faithfulness

result = faithfulness(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
    contexts=["Paris is the capital of France and its largest city."]
)

# Result: 1.0 (fully faithful) or 0.0 (not faithful)
```

**Interpretação:**
- Range: 0 a 1
- 1.0: Fully supported by sources
- 0.0: Not supported by sources
- Critical for preventing hallucinations

### 4.2 Context Precision

**Definição**: Quão precisos são os contextos (documents) selecionados para responder a pergunta.

**RAGAS Implementation:**
```python
from ragas.metrics import context_precision

result = context_precision(
    question="What is the capital of France?",
    contexts=[
        "Paris is the capital of France and its largest city.",
        "France is a country in Europe."
    ],
    answer="The capital of France is Paris."
)

# Measures how many selected contexts are actually useful
```

### 4.3 Context Recall

**Definição**: Quão completos são os contextos selecionados (não deixam informações importantes de fora).

**RAGAS Implementation:**
```python
from ragas.metrics import context_recall

result = context_recall(
    question="What is the capital of France?",
    contexts=["Paris is the capital of France and its largest city."],
    answer="The capital of France is Paris.",
    # Ground truth: necessary contexts
    contexts_needed=["Paris is the capital of France"]
)

# Measures if all necessary contexts were retrieved
```

### 4.4 Answer Relevance

**Definição**: Quão relevante é a resposta para a pergunta.

**Como medir:**
- LLM-as-judge: Ask LLM if answer addresses question
- Semantic similarity: Similarity between question and answer
- Human evaluation

### 4.5 Factual Correctness

**Definição**: Se os fatos na resposta estão corretos.

**RAGAS:**
```python
from ragas.metrics import factuality

result = factuality(
    question="When was the US Constitution written?",
    answer="The US Constitution was written in 1787.",
    contexts=["The US Constitution was written and signed in 1787."]
)
```

### 4.6 Traditional NLP Metrics

**BLEU** (Bilingual Evaluation Understudy):
- Measures n-gram overlap
- Originally for machine translation
- Can be used for text generation

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation):
- Measures recall of n-grams
- Originally for summarization
- Good for extractive summaries

**BERTScore**:
- Uses BERT embeddings to compute similarity
- More semantic than BLEU/ROUGE
- Good for paraphrase detection

### 4.7 Generation Metrics Comparison

| Metric | Measures | Range | Use Case | Pros | Cons |
|--------|----------|-------|----------|------|------|
| **Faithfulness** | Alignment with sources | 0-1 | RAG quality | ✅ RAG-specific | ❌ LLM-dependent |
| **Context Precision** | Context usefulness | 0-1 | Retrieval quality | ✅ Precise | ❌ Requires LLM |
| **Context Recall** | Context completeness | 0-1 | Retrieval quality | ✅ Comprehensive | ❌ Requires ground truth |
| **Factual Correctness** | Factual accuracy | 0-1 | Fact checking | ✅ Direct | ❌ Hard to automate |
| **BLEU/ROUGE** | N-gram overlap | 0-1+ | Summarization | ✅ Simple | ❌ Not semantic |
| **BERTScore** | Semantic similarity | -1 to 1 | Paraphrase | ✅ Semantic | ❌ Less interpretable |

---

## 5. DATASETS

### 5.1 MS MARCO

**Description**: Microsoft Machine Reading COmprehension dataset

**Contents:**
- 1M+ query-document pairs
- 100k+ queries with answers
- Real web search queries
- Human generated answers

**Use Cases:**
- Learning to rank
- Document ranking
- QA training
- RAG evaluation

**Download:**
```bash
# Via datasets library
from datasets import load_dataset

dataset = load_dataset("ms_marco", "v1.1")
```

**Evaluation Protocol:**
- Query-document pairs
- Binary relevance judgments
- Standard for information retrieval

### 5.2 BEIR (Benchmarking Information Retrieval)

**Description**: A heterogeneous benchmark for information retrieval

**Contents:**
- 17+ datasets
- Various domains (MS MARCO, NFCorpus, TREC-COVID, etc.)
- Different task types
- Standardized evaluation

**Datasets Included:**
1. **MS MARCO**: Web search
2. **NFCorpus**: Medical documents
3. **TREC-COVID**: COVID research
4. **FiQA**: Financial QA
5. **Signal-1M**: News retrieval
6. **TopiMCQA**: Topic-based QA
7. **ArguAna**: Argument retrieval
8. **T-Rex**: Wikidata facts
9. **FEVER**: Fact verification
10. **Climate-FEVER**: Climate fact checking
11. **SciFact**: Scientific claims
12. **BioASQ**: Biomedical QA
13. **NQ**: Natural Questions
14. **HotpotQA**: Multi-hop QA
15. **FiD**: Fuse-in-Decoder
16. **CAMeo**: Complex questions
17. **Quora**: Duplicate questions

**Usage:**
```python
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import DenseRetrievalDenoser
from beir.retrieval.evaluation import EvaluateRetrieval

# Download and load dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
out_dir = "datasets/"
util.download_and_unzip(url, out_dir)

# Load data
corpus, queries, qrels = GenericDataLoader(out_dir + "msmarco/").load(split="dev")
```

### 5.3 NQ-Open (Natural Questions)

**Description**: Open-domain question answering from real user queries

**Contents:**
- 100k+ questions
- Wikipedia documents
- Human verified answers
- Natural language queries

**Use Cases:**
- Open-domain QA
- RAG evaluation
- Passage retrieval

**Format:**
```json
{
  "question": "What is the capital of France?",
  "answer": ["Paris"],
    "title": "France",
    "ctxs": [/* relevant contexts */]
}
```

### 5.4 SQuAD (Stanford QA)

**Description**: Reading comprehension dataset

**Contents:**
- 100k+ questions
- Wikipedia passages
- Extractive answers
- Human annotated

**Versions:**
- SQuAD 1.1: Selectable answers
- SQuAD 2.0: Includes unanswerable questions

### 5.5 Custom Dataset Creation

**Steps:**
1. Collect queries
2. Gather documents
3. Get relevance judgments
4. Create answer annotations (if QA)

**Tools:**
- Amazon Mechanical Turk
- Prolific
- Custom annotation tools

**Considerations:**
- Domain specificity
- Query diversity
- Answer quality
- Inter-annotator agreement

### 5.6 Dataset Comparison

| Dataset | Size | Domain | Task Type | Download |
|---------|------|--------|-----------|----------|
| **MS MARCO** | 1M+ pairs | Web | Ranking | Easy |
| **BEIR** | 17 datasets | Mixed | Mixed | Easy |
| **NQ-Open** | 100k+ | General | QA | Easy |
| **SQuAD** | 100k+ | Wikipedia | Reading Comp | Easy |
| **Custom** | Variable | Your domain | Variable | Hard |

---

## 6. EVALUATION FRAMEWORKS

### 6.1 RAGAS

**Overview**: RAG-specific evaluation framework

**Installation:**
```bash
pip install ragas
```

**Key Features:**
- RAG-specific metrics
- LLM-based evaluation
- Support for various LLM providers
- Easy integration

**Available Metrics:**

**1. Faithfulness**
```python
from ragas.metrics import faithfulness

score = faithfulness(
    question="When was the US Constitution written?",
    answer="The US Constitution was written in 1787.",
    contexts=["The US Constitution was written in 1787."]
)

print(score)  # 1.0 or 0.0
```

**2. Context Precision & Recall**
```python
from ragas.metrics import context_precision, context_recall

precision = context_precision(
    question="What is the capital of France?",
    contexts=["Paris is the capital.", "France is a country."],
    answer="The capital of France is Paris."
)

recall = context_recall(
    question="What is the capital of France?",
    contexts=["Paris is the capital."],
    answer="The capital of France is Paris.",
    contexts_needed=["Paris is the capital of France"]
)
```

**3. Answer Relevance**
```python
from ragas.metrics import answer_relevance

score = answer_relevance(
    question="What is the capital of France?",
    answer="The capital of France is Paris."
)
```

**Complete Evaluation:**
```python
from ragas import evaluate
from datasets import Dataset

# Prepare data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["The capital of France is Paris."],
    "contexts": [["Paris is the capital of France."]]
}

dataset = Dataset.from_dict(data)

# Run evaluation
results = evaluate(dataset, metrics=[faithfulness, context_precision, answer_relevance])

print(results)
```

**Advantages:**
- RAG-specific metrics
- Easy to use
- Active development
- Good documentation

**Limitations:**
- Requires LLM API (cost)
- LLM-as-judge can be subjective
- Limited metrics (focused on RAG)

### 6.2 TruLens

**Overview**: Evaluation and tracking for LLM applications

**Installation:**
```bash
pip install trulens
```

**Key Features:**
- Comprehensive evaluation
- Tracing and debugging
- Feedback functions
- Custom metrics

**Basic Usage:**
```python
from trulens_eval import Feedback, Tru, TruApp
from trulens_eval.feedback.provider import OpenAI
import openai

# Setup
provider = OpenAI()
Tru().reset_database()

# Define feedback functions
f_groundedness = Feedback(
    provider.groundedness,
    name="Groundedness"
)

f_answer_relevance = Feedback(
    provider.relevance,
    name="Answer Relevance"
)

# Create app
from trulens_eval import RAG

rag = RAG(
    prompt_prefix="Answer the question: ",
    app_id="My RAG App"
)

# Add feedback
rag = rag.on_conversation_default().with_feedback([f_groundedness, f_answer_relevance])

# Query
with rag:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )

# Get results
records = Tru().get_records()
print(records)
```

**Feedback Functions:**
- **Groundedness**: If answer is supported by sources
- **Answer Relevance**: If answer addresses question
- **Context Relevance**: If retrieved contexts are relevant
- **Summarization Quality**: For summaries
- **Custom**: Write your own

**Advantages:**
- Comprehensive tracking
- Multiple metrics
- Good for debugging
- Custom feedback functions

**Limitations:**
- More complex setup
- Requires LLM API
- Heavyweight for simple use cases

### 6.3 DeepEval

**Overview**: Unit testing for LLM/RAG applications

**Installation:**
```bash
pip install deepeval
```

**Setup:**
```python
# In your terminal
deepeval set-openai-api-key YOUR_KEY
```

**Basic Usage:**
```python
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define test case
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    retrieval_context=["Paris is the capital of France."]
)

# Define metrics
faithfulness_metric = FaithfulnessMetric(threshold=0.5)
relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

# Run test
assert_test(test_case, [faithfulness_metric, relevancy_metric])
```

**Advantages:**
- Unit testing framework
- CI/CD integration
- Clear pass/fail
- Simple syntax

**Limitations:**
- Testing focused
- Less comprehensive than TruLens
- Limited to simple use cases

### 6.4 LangSmith

**Overview**: LangChain's evaluation and tracing platform

**Setup:**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
os.environ["LANGCHAIN_PROJECT"] = "my_rag_project"
```

**Usage:**
```python
from langsmith import Client
from langchain.smith import RunEvalConfig

client = Client()

# Define evaluation
eval_config = RunEvalConfig(
    evaluators=["qa", "context_precision"],
    custom_evaluators=[]
)

# Run evaluation
results = client.run_on_dataset(
    dataset_name="my_dataset",
    llm_or_chain_factory=your_rag_chain,
    evaluation=eval_config,
)

print(results)
```

**Advantages:**
- LangChain integration
- Good UI for visualization
- Dataset management
- Tracing

**Limitations:**
- LangChain-specific
- Cloud-based (needs account)
- Less flexible than other tools

### 6.5 Framework Comparison

| Framework | Focus | Metrics | Ease of Use | Cost | Best For |
|-----------|-------|---------|-------------|------|----------|
| **RAGAS** | RAG-specific | RAG metrics | ✅ Easy | API cost | RAG evaluation |
| **TruLens** | General LLM | Comprehensive | 🟡 Medium | API cost | Full-stack evaluation |
| **DeepEval** | Testing | Core metrics | ✅ Easy | API cost | Unit testing |
| **LangSmith** | LangChain | Core metrics | 🟡 Medium | Platform | LangChain apps |

### 6.6 Which Framework to Choose?

**For RAG-specific evaluation:**
→ **RAGAS** or **TruLens**

**For unit testing:**
→ **DeepEval**

**For LangChain apps:**
→ **LangSmith**

**For quick evaluation:**
→ **RAGAS** (simplest)

**For comprehensive evaluation:**
→ **TruLens** (most features)

---

## 7. OFFLINE VS ONLINE EVALUATION

### 7.1 Offline Evaluation

**Definition**: Evaluation on fixed datasets with known ground truth.

**Process:**
1. Choose dataset (MS MARCO, BEIR, etc.)
2. Run your system
3. Compare results to ground truth
4. Calculate metrics

**Advantages:**
- ✅ Reproducible
- ✅ Easy to iterate
- ✅ No user impact
- ✅ Can test extreme cases
- ✅ Standardized benchmarks

**Disadvantages:**
- ❌ May not reflect real usage
- ❌ Limited to available datasets
- ❌ No actual user feedback
- ❌ Can overfit to benchmark

**When to use:**
- Development and tuning
- Comparing different approaches
- Regression testing
- Academic research

### 7.2 Online Evaluation

**Definition**: Evaluation with real users in production.

**Methods:**

**1. A/B Testing**
```python
# A/B test implementation
def ab_test(control_rag, treatment_rag, traffic_split=0.5):
    user_query = get_user_query()

    if random.random() < traffic_split:
        answer = treatment_rag.query(user_query)
        group = "treatment"
    else:
        answer = control_rag.query(user_query)
        group = "control"

    # Log for analysis
    log_result(user_query, answer, group)
```

**2. Shadow Mode**
- Run both systems side-by-side
- Only show control to user
- Log treatment for analysis
- Compare performance

**3. User Feedback**
```python
# Collect explicit feedback
def collect_feedback(answer):
    print("Was this answer helpful? (1-5)")
    rating = int(input())
    log_feedback(answer, rating)

    print("Any additional comments?")
    comment = input()
    log_comment(answer, comment)
```

**Advantages:**
- ✅ Real user data
- ✅ Actual user satisfaction
- ✅ Real-world performance
- ✅ Can measure business metrics

**Disadvantages:**
- ❌ User impact (bad answers)
- ❌ Hard to control variables
- ❌ More complex setup
- ❌ Requires traffic

**When to use:**
- Pre-production testing
- Validating offline results
- Measuring business impact
- Continuous improvement

### 7.3 A/B Testing Best Practices

**1. Statistical Significance**
```python
import scipy.stats as stats

def check_significance(control_results, treatment_results):
    t_stat, p_value = stats.ttest_ind(treatment_results, control_results)

    if p_value < 0.05:
        print("Results are statistically significant")
        return True
    else:
        print("Results are not statistically significant")
        return False
```

**2. Sample Size**
```python
# Calculate required sample size
from statsmodels.stats.power import ttest_power

# Effect size (Cohen's d)
effect_size = 0.5
alpha = 0.05
power = 0.8

sample_size = ttest_power(effect_size, alpha=alpha, nobs=None, alternative='two-sided')
print(f"Required sample size: {sample_size}")
```

**3. Multiple Comparisons**
```python
from statsmodels.stats.multitest import multipletests

# If testing multiple metrics
p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
```

**4. Guardrails**
- Gradually increase treatment traffic
- Set up monitoring for degradation
- Have rollback plan
- Monitor user satisfaction

---

## 8. HUMAN EVALUATION

### 8.1 When to Use

**Use human evaluation when:**
- ✅ Building a critical system
- ✅ Automated metrics are insufficient
- ✅ Quality is paramount
- ✅ Need to validate automated metrics

**Don't use when:**
- ❌ Budget/time constraints
- ❌ Iterating quickly
- ❌ Automated metrics are sufficient
- ❌ Early prototyping

### 8.2 Evaluation Protocol

**1. Define Evaluation Criteria**
```markdown
Evaluation Criteria:
- Relevance (1-5): How relevant is the answer to the question?
- Faithfulness (1-5): Is the answer supported by the sources?
- Completeness (1-5): Does the answer cover all aspects?
- Correctness (1-5): Are the facts correct?
```

**2. Create Evaluation Interface**
```html
<!DOCTYPE html>
<html>
<body>
    <h2>Evaluation Task</h2>

    <div>
        <strong>Question:</strong>
        <p id="question"></p>
    </div>

    <div>
        <strong>Answer:</strong>
        <p id="answer"></p>
    </div>

    <div>
        <strong>Context:</strong>
        <ul id="context"></ul>
    </div>

    <div>
        <label>Relevance (1-5):</label>
        <input type="range" id="relevance" min="1" max="5" value="3">
        <span id="relevance_val">3</span>
    </div>

    <div>
        <label>Faithfulness (1-5):</label>
        <input type="range" id="faithfulness" min="1" max="5" value="3">
        <span id="faithfulness_val">3</span>
    </div>

    <div>
        <label>Comments:</label>
        <textarea id="comments" rows="4" cols="50"></textarea>
    </div>

    <button onclick="submit()">Submit</button>

    <script>
        // Load question, answer, context
        // Handle form submission
    </script>
</body>
</html>
```

**3. Quality Control**
- Inter-annotator agreement (Cohen's kappa)
- Double annotation (2 annotators per item)
- Regular calibration sessions
- Clear annotation guidelines

**4. Statistical Analysis**
```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Calculate inter-annotator agreement
annotator1_scores = [3, 4, 2, 5, 3]
annotator2_scores = [3, 4, 3, 5, 4]

kappa = cohen_kappa_score(annotator1_scores, annotator2_scores)
print(f"Cohen's Kappa: {kappa}")

# Interpretation:
# < 0.20: Slight agreement
# 0.21-0.40: Fair agreement
# 0.41-0.60: Moderate agreement
# 0.61-0.80: Substantial agreement
# 0.81-1.00: Almost perfect agreement
```

### 8.3 Crowdsourcing

**Platforms:**
- Amazon Mechanical Turk
- Prolific
- Appen
- Scale

**Guidelines:**
- Clear instructions
- Example evaluations
- Qualification tests
- Payment per task
- Quality bonuses

**Example Task:**
```markdown
Task: Evaluate RAG System Responses

You will be shown a question, answer, and source documents.

Rate the answer on a scale of 1-5 for:
1. Relevance: How well does the answer address the question?
2. Faithfulness: Is the answer supported by the source documents?
3. Correctness: Are the facts in the answer correct?

Payment: $0.10 per evaluation
Estimated time: 2 minutes per task

Qualification: Must pass test with 80% accuracy
```

### 8.4 Cost Estimation

**Human Evaluation Costs:**
- Crowd workers: $0.10-$0.50 per evaluation
- Expert annotators: $20-$100 per hour
- Platform fees: 10-30% of payment

**Example:**
- 1000 evaluations
- $0.20 per evaluation
- Platform fee (20%): $0.04
- Total: $240

---

## 9. EVALUATION PIPELINE

### 9.1 Complete Evaluation Setup

```python
import json
from datetime import datetime
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from datasets import Dataset

class RAGEvaluator:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.test_data = []
        self.results = {}

    def add_test_case(self, question, expected_answer, contexts):
        """Add a test case."""
        self.test_data.append({
            "question": question,
            "expected_answer": expected_answer,
            "contexts": contexts
        })

    def run_evaluation(self):
        """Run evaluation on all test cases."""
        if not self.test_data:
            print("No test cases added")
            return

        # Get answers from RAG system
        questions = []
        contexts = []
        answers = []

        for test_case in self.test_data:
            question = test_case["question"]
            contexts_list = test_case["contexts"]

            # Get answer from RAG
            answer = self.rag_system.query(question)

            questions.append(question)
            contexts.append(contexts_list)
            answers.append(answer)

        # Create dataset
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }

        dataset = Dataset.from_dict(dataset_dict)

        # Run RAGAS evaluation
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                context_precision,
                context_recall
            ]
        )

        self.results = results
        return results

    def print_results(self):
        """Print evaluation results."""
        if not self.results:
            print("No results to print. Run evaluation first.")
            return

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        for metric_name, scores in self.results.items():
            print(f"\n{metric_name}:")
            print(f"  Mean: {np.mean(scores):.3f}")
            print(f"  Std:  {np.std(scores):.3f}")
            print(f"  Min:  {np.min(scores):.3f}")
            print(f"  Max:  {np.max(scores):.3f}")

    def save_results(self, filename):
        """Save results to file."""
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "num_test_cases": len(self.test_data),
            "results": {k: v for k, v in self.results.items()}
        }

        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {filename}")

    def compare_versions(self, old_results, new_results):
        """Compare two versions of results."""
        print("\n" + "="*60)
        print("VERSION COMPARISON")
        print("="*60)

        for metric in old_results.keys():
            if metric in new_results:
                old_mean = np.mean(old_results[metric])
                new_mean = np.mean(new_results[metric])
                diff = new_mean - old_mean
                pct_change = (diff / old_mean) * 100

                print(f"\n{metric}:")
                print(f"  Old: {old_mean:.3f}")
                print(f"  New: {new_mean:.3f}")
                print(f"  Change: {diff:+.3f} ({pct_change:+.1f}%)")

# Usage
evaluator = RAGEvaluator(my_rag_system)

# Add test cases
evaluator.add_test_case(
    question="What is the capital of France?",
    expected_answer="Paris is the capital of France.",
    contexts=["Paris is the capital of France and its largest city."]
)

evaluator.add_test_case(
    question="When was the US Constitution written?",
    expected_answer="The US Constitution was written in 1787.",
    contexts=["The US Constitution was written in 1787."]
)

# Run evaluation
results = evaluator.run_evaluation()

# Print results
evaluator.print_results()

# Save results
evaluator.save_results("evaluation_results.json")
```

### 9.2 Continuous Evaluation

```python
import schedule
import time

def daily_evaluation():
    """Run evaluation on test set daily."""
    evaluator = RAGEvaluator(rag_system)

    # Load test set
    with open("test_set.json") as f:
        test_set = json.load(f)

    for test_case in test_set:
        evaluator.add_test_case(**test_case)

    results = evaluator.run_evaluation()
    evaluator.print_results()
    evaluator.save_results(f"evaluation_{datetime.now().strftime('%Y%m%d')}.json")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(daily_evaluation)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 9.3 Production Monitoring

```python
class ProductionMonitor:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.metrics = {
            "query_count": 0,
            "avg_latency": 0,
            "error_rate": 0,
            "user_satisfaction": 0
        }

    def log_query(self, question, answer, latency, user_rating=None):
        """Log a query for monitoring."""
        self.metrics["query_count"] += 1

        # Update average latency
        self.metrics["avg_latency"] = (
            (self.metrics["avg_latency"] * (self.metrics["query_count"] - 1) + latency)
            / self.metrics["query_count"]
        )

        # Log user rating if provided
        if user_rating:
            self.metrics["user_satisfaction"] = (
                (self.metrics["user_satisfaction"] * (self.metrics["query_count"] - 1) + user_rating)
                / self.metrics["query_count"]
            )

    def get_metrics(self):
        """Get current metrics."""
        return self.metrics

    def check_alerts(self):
        """Check if any metrics trigger alerts."""
        alerts = []

        if self.metrics["avg_latency"] > 2.0:  # seconds
            alerts.append("High latency detected")

        if self.metrics["user_satisfaction"] < 3.0:  # 1-5 scale
            alerts.append("Low user satisfaction")

        return alerts

# Usage in production
monitor = ProductionMonitor(rag_system)

def handle_user_query(question):
    start_time = time.time()

    try:
        answer = rag_system.query(question)
        latency = time.time() - start_time

        # Log query
        monitor.log_query(question, answer, latency)

        # Check alerts
        alerts = monitor.check_alerts()
        if alerts:
            send_alert(alerts)

        return answer

    except Exception as e:
        monitor.log_query(question, None, time.time() - start_time)
        raise e
```

---

## 10. AUTOMATED TESTING

### 10.1 Unit Tests

```python
import unittest
from rag_evaluator import RAGEvaluator

class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RAGEvaluator(mock_rag_system)

        # Add test cases
        self.evaluator.add_test_case(
            question="What is the capital of France?",
            expected_answer="The capital of France is Paris.",
            contexts=["Paris is the capital of France."]
        )

    def test_faithfulness(self):
        """Test faithfulness score."""
        results = self.evaluator.run_evaluation()

        # Check that faithfulness is above threshold
        self.assertGreater(
            results["faithfulness"].mean(),
            0.8,
            "Faithfulness score too low"
        )

    def test_context_precision(self):
        """Test context precision."""
        results = self.evaluator.run_evaluation()

        # Check that context precision is reasonable
        self.assertGreater(
            results["context_precision"].mean(),
            0.5,
            "Context precision too low"
        )

    def test_no_hallucinations(self):
        """Test that system doesn't hallucinate."""
        # Add test case with limited context
        self.evaluator.add_test_case(
            question="What is the population of Mars?",
            expected_answer="I don't have information about Mars' population.",
            contexts=["Mars is a planet in our solar system."]
        )

        results = self.evaluator.run_evaluation()

        # Should have low faithfulness for hallucinated answer
        self.assertLess(
            results["faithfulness"].mean(),
            0.3,
            "System is hallucinating"
        )

if __name__ == "__main__":
    unittest.main()
```

### 10.2 CI/CD Integration

```yaml
# .github/workflows/rag-evaluation.yml
name: RAG Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install ragas

    - name: Run evaluation
      run: |
        python eval_script.py --test-set test_data.json --output results.json

    - name: Check results
      run: |
        python check_evaluation_results.py --threshold 0.8

    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: evaluation-results
        path: results.json
```

### 10.3 Regression Testing

```python
class RegressionTester:
    def __init__(self, baseline_results_file):
        """Initialize with baseline results."""
        with open(baseline_results_file) as f:
            self.baseline = json.load(f)

    def test_regression(self, current_results_file, tolerance=0.05):
        """Test for regression compared to baseline."""
        with open(current_results_file) as f:
            current = json.load(f)

        regression_found = False

        for metric in self.baseline["results"].keys():
            if metric in current["results"]:
                baseline_mean = np.mean(self.baseline["results"][metric])
                current_mean = np.mean(current["results"][metric])

                # Check if current is significantly worse
                if current_mean < baseline_mean * (1 - tolerance):
                    print(f"❌ REGRESSION in {metric}:")
                    print(f"   Baseline: {baseline_mean:.3f}")
                    print(f"   Current:  {current_mean:.3f}")
                    regression_found = True
                else:
                    print(f"✅ {metric}: {current_mean:.3f} (baseline: {baseline_mean:.3f})")

        if regression_found:
            print("\n❌ REGRESSION DETECTED")
            return False
        else:
            print("\n✅ NO REGRESSIONS")
            return True

# Usage
tester = RegressionTester("baseline_results.json")
is_passing = tester.test_regression("current_results.json")

if not is_passing:
    exit(1)  # Fail CI/CD
```

---

## 11. WINDOWS-SPECIFIC CONSIDERATIONS

### 11.1 PowerShell Script for Evaluation

```powershell
# Run evaluation on Windows
param(
    [string]$TestSet = "test_data.json",
    [string]$Output = "results.json"
)

Write-Host "Starting RAG Evaluation..." -ForegroundColor Green

# Set environment variables
$env:OPENAI_API_KEY = "your-key"
$env:EVALUATION_DATA_PATH = $TestSet
$env:EVALUATION_OUTPUT_PATH = $Output

# Run evaluation
python evaluate_rag.py --test-set $TestSet --output $Output

if ($LASTEXITCODE -eq 0) {
    Write-Host "Evaluation completed successfully!" -ForegroundColor Green

    # Display results
    $results = Get-Content $Output | ConvertFrom-Json
    Write-Host "`nResults:" -ForegroundColor Cyan
    $results.results | Format-Table
} else {
    Write-Host "Evaluation failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}
```

### 11.2 Batch Testing

```powershell
# Test multiple configurations
$configs = @(
    @{Name="Config1"; ChunkSize=500; Overlap=100},
    @{Name="Config2"; ChunkSize=1000; Overlap=200},
    @{Name="Config3"; ChunkSize=1500; Overlap=300}
)

foreach ($config in $configs) {
    Write-Host "Testing $($config.Name)..." -ForegroundColor Yellow

    # Set config
    $env:CHUNK_SIZE = $config.ChunkSize
    $env:CHUNK_OVERLAP = $config.Overlap

    # Run evaluation
    $output = "results_$($config.Name).json"
    python evaluate_rag.py --test-set test_data.json --output $output

    # Store results
    $results = Get-Content $output | ConvertFrom-Json

    # Print summary
    Write-Host "Results for $($config.Name):" -ForegroundColor Cyan
    Write-Host "  Faithfulness: $($results.results.faithfulness)" -ForegroundColor White
    Write-Host "  Context Precision: $($results.results.context_precision)" -ForegroundColor White
}
```

---

## 12. BEST PRACTICES

### 12.1 Evaluation Strategy

1. **Start with offline evaluation**
   - Use standard datasets
   - Establish baseline metrics
   - Iterate and improve

2. **Add human evaluation for critical systems**
   - Validate automated metrics
   - Get qualitative feedback
   - Calibrate thresholds

3. **Monitor in production**
   - Track key metrics
   - Collect user feedback
   - A/B test improvements

4. **Automate evaluation**
   - CI/CD integration
   - Regression tests
   - Scheduled evaluations

### 12.2 Metric Selection

**For retrieval systems:**
- Recall@k, nDCG@k, MRR
- Context precision and recall

**For RAG systems:**
- Faithfulness (critical)
- Context precision/recall
- Answer relevance

**For production systems:**
- User satisfaction
- Business metrics
- Latency

### 12.3 Common Mistakes

❌ **Only using one metric**
- Solution: Use multiple complementary metrics

❌ **Not validating metrics**
- Solution: Human evaluation, correlation analysis

❌ **Testing only on perfect data**
- Solution: Include messy, real-world data

❌ **No baseline**
- Solution: Establish baseline before making changes

❌ **Infrequent evaluation**
- Solution: Automate and run regularly

### 12.4 Reporting Results

```markdown
# RAG System Evaluation Report

## Summary
- Test date: 2025-11-09
- Test set: 100 query-answer pairs
- System version: v1.2.3

## Results

### Retrieval Metrics
- **Recall@10**: 0.85 (0.82 baseline)
- **nDCG@10**: 0.78 (0.75 baseline)
- **MRR**: 0.71 (0.68 baseline)

### RAG Metrics
- **Faithfulness**: 0.89 (0.87 baseline)
- **Context Precision**: 0.82 (0.80 baseline)
- **Context Recall**: 0.76 (0.74 baseline)

## Improvements
- ✅ 3% improvement in Recall@10
- ✅ 4% improvement in Faithfulness
- ✅ All metrics above thresholds

## Issues
- ⚠️ Some answers lack detail
- ⚠️ Context recall could be higher

## Next Steps
1. Improve chunking strategy
2. Add query expansion
3. Test on larger dataset
```

---

## 13. RESEARCH GAPS

### 13.1 To Research
- [ ] Automated evaluation metric validation
- [ ] Metric correlation analysis
- [ ] Domain-specific evaluation approaches
- [ ] Multi-turn conversation evaluation
- [ ] Non-English evaluation
- [ ] Cost-quality tradeoffs in evaluation

### 13.2 Emerging Techniques
- [ ] LLM-as-judge improvements
- [ ] Synthetic data for evaluation
- [ ] Active learning for evaluation
- [ ] Metric learning for retrieval
- [ ] Human-AI hybrid evaluation
- [ ] Real-time evaluation systems

---

## 14. RECOMMENDATIONS

### 14.1 For Beginners
**Start here**: RAGAS with small test set
- Simple to use
- RAG-specific metrics
- Good starting point

### 14.2 For Production
**Recommended**: TruLens + human evaluation
- Comprehensive metrics
- Production monitoring
- User feedback integration

### 14.3 For Research
**Recommended**: RAGAS + custom metrics
- Flexible
- Good for experimentation
- Can add domain-specific metrics

### 14.4 Evaluation Checklist

- [ ] Clear evaluation criteria
- [ ] Appropriate metrics selected
- [ ] Test set is representative
- [ ] Baseline established
- [ ] Human validation (if critical)
- [ ] Automated evaluation in CI/CD
- [ ] Production monitoring setup
- [ ] Regular evaluation schedule
- [ ] Results tracking over time
- [ ] Clear reporting format

---

**Status**: ✅ Base para Evaluation & Benchmarks coletada
**Próximo**: Consolidar Fase 3
**Data Conclusão**: 09/11/2025

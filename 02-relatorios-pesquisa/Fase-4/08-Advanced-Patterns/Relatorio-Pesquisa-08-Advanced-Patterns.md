# Relatório de Pesquisa: Seção 08 - Advanced Patterns

### Data: 09/11/2025
### Status: Fase 4 - Advanced Topics

---

## 1. RESUMO EXECUTIVO

Advanced RAG Patterns extends beyond traditional text-based retrieval to handle complex, multi-modal, and intelligent retrieval scenarios. These patterns enable more sophisticated AI applications with enhanced reasoning and contextual understanding.

**Insights Chave:**
- **Multimodal RAG**: Processes text, images, tables, and code together
- **Agentic RAG**: Multi-step reasoning with tool usage
- **Graph RAG**: Structured knowledge representation
- **Self-RAG**: Self-reflection and correction
- **Fusion RAG**: Multi-query and result fusion
- **Corrective RAG**: Iterative improvement of responses

---

## 2. MULTIMODAL RAG

### 2.1 Overview

**Multimodal RAG** processes and retrieves from multiple data types: text, images, tables, code, audio, video.

**Architecture:**
```
[Text] → [Text Encoder] → [Text Embeddings]
[Images] → [Vision Encoder] → [Image Embeddings]
[Tables] → [Table Encoder] → [Table Embeddings]
                     ↓
              [Unified Vector Space]
                     ↓
              [Multimodal Retrieval]
```

### 2.2 CLIP (Contrastive Language-Image Pre-training)

**CLIP** encodes images e text no mesmo embedding space.

**Como Funciona:**
- Image encoder: Vision Transformer (ViT) ou ResNet
- Text encoder: Transformer
- Training: Contrastive learning (maximize similarity of correct pairs, minimize incorrect)

**Vantagens:**
- ✅ Unified text-image embedding
- ✅ Zero-shot classification
- ✅ Good generalization
- ✅ Fast inference

**Implementação:**
```python
import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Encode text
text = clip.tokenize(["a cat", "a dog", "a car"]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)

# Encode image
image = Image.open("image.jpg")
image_input = preprocess(image).unsqueeze(0).to(device)
with torch.no_grad():
    image_features = model.encode_image(image_input)

# Compute similarity
logits_per_image, logits_per_text = model(image_input, text)
probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```

### 2.3 LLaVA (Large Language and Vision Assistant)

**LLaVA** combines LLM com vision model para conversational AI.

**Arquitetura:**
```
[Image] → [Vision Encoder (CLIP ViT)] → [Vision Features]
[VLM] → [Projection Layer] → [LLM Features]
[Text Query] → [LLM] → [Response]
```

**Vantagens:**
- ✅ Visual question answering
- ✅ Image understanding
- ✅ Conversational interface
- ✅ Grounded responses

**Implementação:**
```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run import eval_model

# Load model
model_path = "liuhaotian/llava-v1.5-13b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_8bit=False,
    load_4bit=False,
    device="cuda"
)

# Generate response
prompt = "Describe this image in detail."
image_file = "image.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": model_name,
    "query": prompt,
    "image_file": image_file,
    "conv_mode": "llava_v1",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

result = eval_model(args)
print(result)
```

### 2.4 BLIP (Bootstrapped Language-Image Pre-training)

**BLIP** para image captioning e VQA.

**Capabilities:**
- Image captioning
- Visual question answering
- Image-text retrieval
- Visual grounding

### 2.5 Table RAG

**Table RAG** handles structured tabular data.

**Approaches:**

**1. Direct Table Embedding:**
```python
import pandas as pd
from sentence_transformers import SentenceTransformer

# Embed each row
df = pd.read_csv("data.csv")
model = SentenceTransformer("all-mpnet-base-v2")

# Combine row into text
df['text'] = df.apply(lambda row: " ".join(row.astype(str)), axis=1)
embeddings = model.encode(df['text'].tolist())
```

**2. Schema-Aware Embedding:**
```python
def create_table_embedding(row, schema):
    """Embed table row with schema context."""
    parts = []
    for col, value in zip(schema.columns, row):
        parts.append(f"{col}: {value}")
    return " | ".join(parts)

schema = df.columns
df['text'] = df.apply(lambda row: create_table_embedding(row, df), axis=1)
```

**3. Cell-Level Retrieval:**
```python
def search_tables(query, tables, top_k=5):
    """Search across multiple tables."""
    all_results = []

    for table in tables:
        for idx, row in table.iterrows():
            # Create query for each cell
            cell_text = " ".join([f"{col}: {val}" for col, val in row.items()])
            similarity = compute_similarity(query, cell_text)
            all_results.append({
                'table': table.name,
                'row_idx': idx,
                'text': cell_text,
                'similarity': similarity
            })

    # Sort by similarity
    return sorted(all_results, key=lambda x: x['similarity'], reverse=True)[:top_k]
```

### 2.6 Code RAG

**Code RAG** para programming assistance.

**Approaches:**

**1. AST-Based Chunking:**
```python
import ast
from tree_sitter import Parser

def extract_code_functions(code, language="python"):
    """Extract functions from code using AST."""
    tree = ast.parse(code)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(code, node)
            functions.append({
                'name': node.name,
                'code': func_code,
                'line_start': node.lineno,
                'line_end': node.end_lineno
            })

    return functions
```

**2. Code Embedding:**
```python
from sentence_transformers import SentenceTransformer

# Use code-specific model
model = SentenceTransformer("microsoft/codebert-base")

def embed_code(code_snippet):
    """Embed code snippet."""
    return model.encode(code_snippet)
```

**3. Context-Aware Retrieval:**
```python
def retrieve_code_context(query, code_chunks, max_context=3):
    """Retrieve relevant code with context."""
    # Get similar chunks
    similar_chunks = vectorstore.similarity_search(query, k=10)

    # Expand with context
    results = []
    for chunk in similar_chunks:
        # Get function/class
        function = get_enclosing_function(chunk)
        if function:
            results.append(function)

    # Remove duplicates
    results = list({r['name']: r for r in results}.values())

    return results[:max_context]
```

### 2.7 Multimodal Fusion Strategies

**Early Fusion:**
- Combine multimodal inputs before encoding
- Joint embedding space
- Example: CLIP, BLIP

**Late Fusion:**
- Encode each modality separately
- Fuse at retrieval level
- Example: Separate indexes, then combine

**Cross-Modal Attention:**
- Attention mechanism across modalities
- Example: LLaVA, Flamingo

---

## 3. AGENTIC RAG

### 3.1 Overview

**Agentic RAG** uses LLM agents para multi-step reasoning, tool usage, e iterative retrieval.

**Arquitetura:**
```
[Query] → [Plan] → [Tool Calls] → [Retriever] → [Synthesize] → [Response]
    ↑                                                                    ↓
    └───────────── [Reflection] ←────────────────────────────────────────┘
```

### 3.2 ReAct Pattern

**ReAct** (Reasoning + Acting) combines reasoning e acting para problem solving.

**Process:**
1. **Thought**: What should I do?
2. **Action**: Execute a tool/API call
3. **Observation**: What happened?
4. Repeat until done

**Implementação (LangChain):**
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Define tools
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search, wikipedia]

# Define agent prompt
prompt = PromptTemplate.from_template("""
You are an agent that can use tools to answer questions.

Question: {input}
Thought: {agent_scratchpad}
""")

# Create agent
llm = OpenAI(temperature=0)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = agent_executor.invoke({
    "input": "What is the capital of France? When was it founded?"
})
```

### 3.3 Multi-Hop Reasoning

**Multi-hop** retrieval performs sequential retrieval steps.

**Implementação:**
```python
class MultiHopRetriever:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query(self, question, max_hops=3):
        """Perform multi-hop retrieval."""
        context = ""
        current_question = question

        for hop in range(max_hops):
            # Retrieve
            docs = self.retriever.get_relevant_documents(current_question, k=5)

            # Extract information
            info = "\n".join([doc.page_content for doc in docs])
            context += f"\nHop {hop + 1}:\n{info}\n"

            # Check if we have enough info
            prompt = f"""
            Given the context: {info}
            Question: {question}

            Do you have enough information to answer the question? (yes/no)
            """
            response = self.llm.generate(prompt)

            if "yes" in response.lower():
                break

            # Generate next question
            prompt = f"""
            Given the question: {question}
            And the context: {info}

            What is the next question to ask to find more information?
            """
            current_question = self.llm.generate(prompt)

        return context
```

### 3.4 Self-Reflection

**Self-reflection** allows agent to critique e improve its own responses.

```python
class SelfReflectiveAgent:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query(self, question):
        # Initial retrieval
        docs = self.retriever.get_relevant_documents(question, k=5)
        context = "\n".join([doc.page_content for doc in docs])

        # Generate initial response
        prompt = f"""
        Question: {question}
        Context: {context}

        Provide a comprehensive answer.
        """
        initial_response = self.llm.generate(prompt)

        # Self-reflection
        reflection_prompt = f"""
        Original question: {question}
        Initial response: {initial_response}

        Rate the response (1-10) and explain any issues:
        """
        reflection = self.llm.generate(reflection_prompt)

        # If response is poor, try again
        if "rating: 1-" in reflection or "rating: 2-" in reflection:
            # Try different retrieval or approach
            docs = self.retriever.get_relevant_documents(question, k=10)
            context = "\n".join([doc.page_content for doc in docs])

            # Regenerate with more context
            prompt = f"""
            Question: {question}
            Context: {context}

            Provide a comprehensive, detailed answer.
            """
            final_response = self.llm.generate(prompt)
        else:
            final_response = initial_response

        return final_response
```

### 3.5 Tool-Augmented RAG

**Use external tools** para enhance retrieval.

**Available Tools:**
- Web search (Google, Bing, DuckDuckGo)
- Code execution
- API calls (database, CRM, etc.)
- Document parsers
- Calculators

**Implementação:**
```python
from langchain.agents import create_openai_functions_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain_openai import OpenAI

# Define tools
search = DuckDuckGoSearchRun()

def get_current_date():
    """Get current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

tools = [
    Tool(
        name="web_search",
        description="Search the web for information",
        func=search.run
    ),
    Tool(
        name="get_date",
        description="Get the current date",
        func=get_current_date
    )
]

# Create agent
llm = OpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

# Run
result = agent_executor.invoke({
    "input": "What is the weather in Paris today?"
})
```

---

## 4. GRAPH RAG

### 4.1 Overview

**Graph RAG** uses knowledge graphs para structured retrieval e reasoning.

**Advantages:**
- ✅ Explicit relationships
- ✅ Multi-hop reasoning
- ✅ Structure preservation
- ✅ Contextual understanding

### 4.2 Knowledge Graph Construction

**Steps:**
1. Extract entities e relationships
2. Create graph structure
3. Index nodes e edges
4. Query with graph traversal

**Entity Extraction:**
```python
import spacy
from collections import defaultdict

def extract_entities(text):
    """Extract entities using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })

    return entities

# Relationship extraction
def extract_relationships(text):
    """Extract subject-verb-object triples."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    relationships = []
    for sent in doc.sents:
        sent_ents = {e.text: e for e in sent.ents}
        for token in sent:
            if token.dep_ in ['nsubj', 'dobj']:
                subj = token.text
                verb = token.lemma_
                # Find object
                obj = None
                for child in token.children:
                    if child.dep_ in ['dobj', 'attr', 'prep']:
                        obj = child.text

                if obj and subj in sent_ents and obj in sent_ents:
                    relationships.append({
                        'subject': subj,
                        'relation': verb,
                        'object': obj
                    })

    return relationships
```

### 4.3 Graph Storage

**Neo4j:**
```python
from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_entity(self, name, entity_type):
        """Create entity node."""
        with self.driver.session() as session:
            session.run(
                "CREATE (e:Entity {name: $name, type: $type})",
                name=name, type=entity_type
            )

    def create_relationship(self, subject, relation, obj):
        """Create relationship."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Entity {name: $subject})
                MATCH (b:Entity {name: $object})
                CREATE (a)-[:RELATION {type: $relation}]->(b)
                """,
                subject=subject, relation=relation, object=obj
            )
```

### 4.4 Graph Retrieval

**Cypher Query:**
```python
def retrieve_graph_context(query, max_hops=2):
    """Retrieve context using graph traversal."""
    with self.driver.session() as session:
        result = session.run(
            """
            MATCH (start:Entity)-[:RELATION*1..{max_hops}]-(related:Entity)
            WHERE start.name CONTAINS $query
            RETURN start, related, length(path) as hops
            ORDER BY hops
            LIMIT 10
            """,
            query=query, max_hops=max_hops
        )

        context = []
        for record in result:
            context.append({
                'entity': record['start']['name'],
                'related': record['related']['name'],
                'hops': record['hops']
            })

        return context
```

### 4.5 Vector + Graph Hybrid

**Combine** vector similarity com graph structure.

```python
class HybridGraphRAG:
    def __init__(self, vectorstore, graph_db):
        self.vectorstore = vectorstore
        self.graph_db = graph_db

    def query(self, question):
        # Vector retrieval
        vector_docs = self.vectorstore.similarity_search(question, k=5)

        # Graph traversal
        entities = extract_entities(question)
        graph_context = []
        for entity in entities:
            context = self.graph_db.retrieve_related(entity['text'])
            graph_context.extend(context)

        # Combine
        combined_context = "\n".join([
            doc.page_content for doc in vector_docs
        ]) + "\n" + "\n".join([
            f"{c['entity']} - {c['related']} (hops: {c['hops']})"
            for c in graph_context
        ])

        return combined_context
```

---

## 5. SELF-RAG

### 5.1 Overview

**Self-RAG** (Self-Reflective Retrieval-Augmented Generation) allows the model to learn when e how to retrieve.

**Key Features:**
- Self-reflection tokens
- Critic generation
- Adaptive retrieval
- Learning from feedback

### 5.2 Architecture

**Self-RAG Training:**
1. Generate response
2. Critic generate reflection
3. Select best retrieval
4. Update model

**Inference:**
```
[Query] → [Generate] → [Critic] → [Retrieve] → [Refine] → [Response]
```

### 5.3 Implementation

```python
class SelfRAG:
    def __init__(self, retriever, generator, critic):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic

    def query(self, question):
        # Initial generation without retrieval
        response = self.generator.generate(question)

        # Critic evaluates response
        reflection = self.critic.evaluate(question, response)

        # If critic says retrieval needed
        if reflection.needs_retrieval:
            # Retrieve
            docs = self.retriever.get_relevant_documents(question, k=3)

            # Generate with context
            context = "\n".join([doc.page_content for doc in docs])
            response = self.generator.generate_with_context(
                question, context
            )

        return response
```

---

## 6. CORRECTIVE RAG

### 6.1 Overview

**Corrective RAG** iteratively improves responses through feedback e correction.

**Process:**
1. Initial response
2. Evaluate quality
3. Identify issues
4. Correct and re-generate

### 6.2 Implementation

```python
class CorrectiveRAG:
    def __init__(self, retriever, generator, evaluator):
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator

    def query(self, question, max_iterations=3):
        """Iteratively improve response."""
        for iteration in range(max_iterations):
            # Generate response
            if iteration == 0:
                docs = self.retriever.get_relevant_documents(question, k=3)
            else:
                docs = self.retriever.get_relevant_documents(question, k=5)

            context = "\n".join([doc.page_content for doc in docs])
            response = self.generator.generate_with_context(question, context)

            # Evaluate
            feedback = self.evaluator.evaluate(question, response)

            # Check if good enough
            if feedback.score > 0.8:
                return response

            # If not, continue to next iteration

        return response
```

---

## 7. FUSION RAG

### 7.1 Multi-Query Fusion

**Combine** results de multiple queries.

```python
class MultiQueryRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query(self, question):
        # Generate multiple query variations
        variations = self.llm.generate_variations(question)
        all_results = []

        # Retrieve for each variation
        for variation in variations:
            docs = self.retriever.get_relevant_documents(variation, k=3)
            all_results.extend(docs)

        # Remove duplicates
        unique_docs = list({doc.page_content: doc for doc in all_results}.values())

        # Score and rank
        scored_docs = self.llm.score_documents(question, unique_docs)
        top_docs = sorted(scored_docs, key=lambda x: x['score'], reverse=True)[:5]

        return [doc['document'] for doc in top_docs]
```

### 7.2 Result Fusion

**Combine** multiple retrieval results.

```python
from collections import Counter

def fuse_results(results_list, method="rrf"):
    """Fuse results from multiple retrievers."""
    if method == "rrf":  # Reciprocal Rank Fusion
        scores = Counter()
        for results in results_list:
            for rank, doc in enumerate(results, 1):
                scores[doc] += 1 / (rank + 60)  # 60 is typical

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in fused]

    elif method == "max":  # Max score
        scores = {}
        for results in results_list:
            for doc in results:
                if doc not in scores or doc['score'] > scores[doc]['score']:
                    scores[doc] = doc

        sorted_docs = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return sorted_docs
```

---

## 8. COMPARISON MATRIX

| Pattern | Use Case | Complexity | Quality | Speed | When to Use |
|---------|----------|------------|---------|-------|-------------|
| **Multimodal RAG** | Text + Images/Tables | High | High | Medium | Multi-media data |
| **Agentic RAG** | Complex reasoning | Very High | Very High | Low | Multi-step problems |
| **Graph RAG** | Structured knowledge | High | High | Medium | Relationships matter |
| **Self-RAG** | Self-improvement | Medium | High | Medium | Continuous learning |
| **Corrective RAG** | Quality critical | Medium | Very High | Low | High-stakes |
| **Fusion RAG** | Robust retrieval | Medium | High | Medium | Noisy queries |

---

## 9. DECISION TREE

```
USE CASE
├─ Multiple data types (text + images)?
│   └─ SIM → Multimodal RAG
│
├─ Complex reasoning (multi-step)?
│   └─ SIM → Agentic RAG
│
├─ Structured relationships?
│   └─ SIM → Graph RAG
│
├─ Quality is critical?
│   └─ SIM → Corrective RAG
│
├─ Queries are noisy/ambiguous?
│   └─ SIM → Fusion RAG
│
└─ Standard text RAG?
    └─ SIM → Traditional RAG
```

---

## 10. IMPLEMENTATION GUIDELINES

### 10.1 Multimodal RAG
- Start with CLIP para simple image-text tasks
- Use LLaVA para complex VQA
- Consider computational cost (large models)
- Preprocess images (resize, normalize)

### 10.2 Agentic RAG
- Use for complex, multi-step problems
- Implement proper tool interfaces
- Add error handling
- Monitor agent actions
- Consider cost (multiple LLM calls)

### 10.3 Graph RAG
- Build knowledge graph para structured data
- Use Neo4j ou similar
- Implement graph traversal
- Consider maintenance cost
- Hybrid com vector search

### 10.4 Self/Corrective RAG
- Use para high-quality requirements
- Implement feedback loops
- Add evaluation metrics
- Consider iteration cost
- Monitor improvement

---

## 11. BEST PRACTICES

### 11.1 General
- Start simple, add complexity gradually
- Evaluate each pattern on your data
- Consider computational cost
- Monitor performance
- Have fallback options

### 11.2 Pattern Selection
- Match pattern to use case
- Don't over-engineer
- Test in production
- A/B test different approaches
- Monitor user satisfaction

### 11.3 Implementation
- Use established libraries
- Implement caching
- Add monitoring
- Version control
- Document decisions

---

## 12. RESEARCH GAPS

### 12.1 To Research
- [ ] Multimodal RAG benchmarks
- [ ] Agentic RAG evaluation methods
- [ ] Graph RAG at scale
- [ ] Self-RAG training strategies
- [ ] Corrective RAG effectiveness
- [ ] Fusion RAG optimization

### 12.2 Future Directions
- [ ] Unified multimodal frameworks
- [ ] Autonomous agent systems
- [ ] Knowledge graph construction automation
- [ ] Self-improving RAG systems
- [ ] Real-time corrective systems
- [ ] Advanced fusion techniques

---

**Status**: ✅ Base para Advanced Patterns coletada
**Próximo**: Seção 09 - Architecture Patterns
**Data Conclusão**: 09/11/2025

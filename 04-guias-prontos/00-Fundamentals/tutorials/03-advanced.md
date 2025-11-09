# Tutorial Avançado: RAG Avançado (2024-2025)

**Tempo estimado:** 3-4 horas
**Nível:** Avançado
**Pré-requisitos:** RAG intermediário, Python avançado, experiência com LLMs

## Objetivo
Implementar técnicas avançadas de RAG: Self-RAG, Agentic RAG, Multimodal RAG, e Reranking.

## 1. Self-RAG (2024)

### 1.1 Conceito
Sistema que se auto-avalia e itera para melhorar a qualidade.

```python
class SelfRAG:
    def __init__(self, retriever, generator, critic):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic

    def query(self, question, max_iterations=3, quality_threshold=0.8):
        for i in range(max_iterations):
            # 1. Retrieve
            context = self.retriever.get_relevant_docs(question)

            # 2. Generate
            answer = self.generator.generate(question, context)

            # 3. Self-critique
            feedback = self.critic.evaluate(question, answer, context)
            quality_score = feedback["quality_score"]

            # 4. Check if quality is good enough
            if quality_score >= quality_threshold:
                return {
                    "answer": answer,
                    "quality_score": quality_score,
                    "iterations": i + 1,
                    "feedback": feedback
                }

            # 5. Improve context/answer for next iteration
            if i < max_iterations - 1:
                question = self._improve_question(question, feedback)
                context = self._improve_context(context, feedback)

        return {
            "answer": answer,
            "quality_score": quality_score,
            "iterations": max_iterations,
            "feedback": feedback
        }

    def _improve_question(self, question, feedback):
        """Melhorar pergunta baseado no feedback"""
        improvement_prompt = f"""
Pergunta original: {question}

Feedback sobre a resposta:
- Pontos fortes: {feedback['strengths']}
- Pontos fracos: {feedback['weaknesses']}
- Informações em falta: {feedback['missing_info']}

Reescreva a pergunta para ser mais específica e clara:
"""
        return self.generator.improve(improvement_prompt)

    def _improve_context(self, context, feedback):
        """Melhorar contexto baseado no feedback"""
        if feedback.get('need_more_context'):
            # Retrieve additional docs
            additional_docs = self.retriever.search(
                feedback['missing_topics'],
                k=3
            )
            context += "\n\n" + "\n".join(additional_docs)

        return context
```

### 1.2 Implementação Completa

```python
class LLMCritic:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, question, answer, context):
        """Evalua a qualidade da resposta"""
        evaluation_prompt = f"""
Avalie a qualidade da resposta com base no contexto fornecido.

Pergunta: {question}

Contexto: {context}

Resposta: {answer}

Forneça uma avaliação detalhada em formato JSON:

{{
  "quality_score": 0.0-1.0,
  "faithfulness": 0.0-1.0,
  "completeness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "strengths": ["ponto1", "ponto2"],
  "weaknesses": ["ponto1", "ponto2"],
  "missing_info": ["info1", "info2"],
  "need_more_context": true/false,
  "suggestions": ["sugestão1", "sugestão2"]
}}
"""
        response = self.llm(evaluation_prompt)
        return json.loads(response)

# Usage
self_rag = SelfRAG(
    retriever=retriever,
    generator=generator,
    critic=LLMCritic(llm)
)

result = self_rag.query("Explique RAG em detalhes")
print(f"Answer: {result['answer']}")
print(f"Quality: {result['quality_score']}")
print(f"Iterations: {result['iterations']}")
```

## 2. Agentic RAG

### 2.1 Multi-Step Reasoning

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.schema import BaseMessage

class AgenticRAG:
    def __init__(self, llm, retriever, tools):
        self.llm = llm
        self.retriever = retriever
        self.tools = tools
        self.agent = None

    def create_agent(self):
        """Criar agent com tools customizadas"""
        system_prompt = """
Você é um assistente especializado em RAG.

Use as ferramentas para:
1. Buscar informações relevantes
2. Avaliar a qualidade da resposta
3. Expandir queries quando necessário
4. Responder perguntas com precisão

Sempre cite suas fontes e seja preciso.
"""

        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=system_prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )

    def query(self, question):
        """Query com multi-step reasoning"""
        if not self.agent:
            self.create_agent()

        return self.agent_executor.run(question)

# Definir tools
def search_docs(query: str):
    """Buscar documentos relevantes"""
    docs = retriever.similarity_search(query, k=3)
    context = "\n\n".join([
        f"Source {i}: {doc.page_content}"
        for i, doc in enumerate(docs, 1)
    ])
    return context

def expand_query(query: str):
    """Expandir query com sinônimos"""
    expansion_prompt = f"""
Gere 3 variações da query que capturam aspectos diferentes:

Query: {query}

Variações:
"""
    variations = llm(expansion_prompt).split('\n')
    return variations[:3]

def evaluate_answer(question: str, answer: str):
    """Evaluar qualidade da resposta"""
    eval_prompt = f"""
Avalie a resposta (0-1):

Pergunta: {question}
Resposta: {answer}

Score:"""
    return llm(eval_prompt)

# Criar tools
tools = [
    Tool(
        name="search",
        func=search_docs,
        description="Busca documentos relevantes"
    ),
    Tool(
        name="expand_query",
        func=expand_query,
        description="Expandir query com variações"
    ),
    Tool(
        name="evaluate",
        func=evaluate_answer,
        description="Avaliar qualidade da resposta"
    )
]

# Usage
agentic_rag = AgenticRAG(llm, retriever, tools)
result = agentic_rag.query("Como RAG reduz hallucinations?")
print(result)
```

## 3. Multimodal RAG

### 3.1 Text + Images

```python
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
import openai

class MultimodalRAG:
    def __init__(self):
        self.text_encoder = OpenAIEmbeddings()
        self.image_encoder = self._load_clip()
        self.vectorstore = Chroma()
        self.llm = OpenAI(temperature=0)

    def _load_clip(self):
        """Carregar CLIP para embeddings de imagem"""
        from sentence_transformers import CLIPModel
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, image_path: str):
        """Encode imagem para embedding"""
        from PIL import Image
        image = Image.open(image_path)
        return self.image_encoder.get_image_features(image)

    def index_multimodal_document(self, text: str, image_path: str = None):
        """Indexar documento multimodal"""
        # Encode text
        text_embedding = self.text_encoder.embed_documents([text])[0]

        # Encode image se presente
        embeddings = [text_embedding]
        metadatas = [{"type": "text", "content": text}]

        if image_path:
            image_embedding = self.encode_image(image_path).tolist()
            embeddings.append(image_embedding)
            metadatas.append({
                "type": "image",
                "content": image_path
            })

        # Store
        self.vectorstore.add_embeddings(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"doc_{len(embeddings)}"]
        )

    def query(self, text_query: str = None, image_query: str = None):
        """Query multimodal"""
        # Encode query
        if image_query:
            query_embedding = self.encode_image(image_query)
            query_type = "image"
        else:
            query_embedding = self.text_encoder.embed_query(text_query)
            query_type = "text"

        # Search
        results = self.vectorstore.similarity_search_by_vector(
            query_embedding,
            k=5
        )

        # Separate by type
        text_docs = [r for r in results if r.metadata.get("type") == "text"]
        image_docs = [r for r in results if r.metadata.get("type") == "image"]

        # Generate response
        context = self._build_context(text_docs, image_docs)

        prompt = f"""
Pergunta: {text_query or 'Descreva a imagem'}

Contexto multimodal:
{context}

Responda de forma completa, integrando informações textuais e visuais:
"""
        answer = self.llm(prompt)

        return {
            "answer": answer,
            "text_sources": [d.page_content for d in text_docs],
            "image_sources": [d.metadata["content"] for d in image_docs]
        }

    def _build_context(self, text_docs, image_docs):
        context = "Documentos textuais:\n"
        for i, doc in enumerate(text_docs, 1):
            context += f"{i}. {doc.page_content}\n"

        if image_docs:
            context += "\nImagens relacionadas:\n"
            for i, doc in enumerate(image_docs, 1):
                context += f"{i}. {doc.metadata['content']}\n"

        return context

# Usage
mm_rag = MultimodalRAG()

# Index
mm_rag.index_multimodal_document(
    "Este é um grafo mostrando arquitetura RAG",
    "images/rag_architecture.png"
)

# Query com texto
result = mm_rag.query(
    text_query="Explique a arquitetura RAG mostrada"
)

# Query com imagem
result2 = mm_rag.query(
    image_query="images/diagram.png"
)
```

## 4. Reranking com Cross-Encoders

### 4.1 BGE Reranker

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query: str, documents: list, top_k: int = 3):
        """Rerankar documentos com cross-encoder"""
        # Create pairs (query, document)
        pairs = [(query, doc.page_content) for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # Score
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze().cpu()

        # Sort by score
        scored_docs = [
            (doc, score.item())
            for doc, score in zip(documents, scores)
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [doc for doc, _ in scored_docs[:top_k]]

# Usage
reranker = CrossEncoderReranker()

# Initial retrieval (fast, less accurate)
initial_docs = vectorstore.similarity_search(query, k=20)

# Rerank (slower, more accurate)
final_docs = reranker.rerank(query, initial_docs, top_k=3)
```

### 4.2 ColBERT Reranking

```python
from colbert import Searcher

class ColBERTReranker:
    def __init__(self, index_path: str):
        self.searcher = Searcher(index=index_path)

    def rerank(self, query: str, documents: list, k: int = 100):
        """Rerankar com ColBERT"""
        # ColBERT é eficiente para reranking em larga escala
        # Funciona melhor com indices pré-criados
        passages = [doc.page_content for doc in documents]

        # Search
        results = self.searcher.search(
            query,
            k=k,
            # selection=Passage selection
        )

        # Map back to documents
        reranked_docs = [documents[i] for i, _ in results]

        return reranked_docs
```

## 5. Performance Optimization

### 5.1 Async Processing

```python
import asyncio
from typing import List

class OptimizedRAG:
    def __init__(self, vectorstore, llm, embeddings):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.cache = {}

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str):
        """Cache embeddings"""
        return self.embeddings.embed_query(text)

    async def async_query(self, question: str):
        """Query assíncrono"""
        # Embed
        query_embedding = self.get_embedding(question)

        # Search (potentially async if vectorstore supports it)
        docs = await self.vectorstore.asimilarity_search(
            question,
            k=4
        )

        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        answer = await self.llm.ainvoke(prompt)

        return {
            "answer": answer,
            "source_docs": docs
        }

    async def batch_query(self, questions: List[str]):
        """Processar múltiplas queries em paralelo"""
        tasks = [self.async_query(q) for q in questions]
        results = await asyncio.gather(*tasks)
        return results

# Usage
async def main():
    rag = OptimizedRAG(vectorstore, llm, embeddings)

    # Single query
    result = await rag.async_query("O que é RAG?")

    # Batch queries
    questions = [
        "O que é RAG?",
        "Como usar RAG?",
        "Por que RAG é importante?"
    ]
    results = await rag.batch_query(questions)

asyncio.run(main())
```

### 5.2 Caching Strategy

```python
import hashlib
from functools import lru_cache
import redis

class CachedRAG:
    def __init__(self, vectorstore, llm, embeddings):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def _get_cache_key(self, question: str, k: int = 4):
        """Gerar cache key determinístico"""
        content = f"{question}|k={k}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, question: str, k: int = 4):
        """Buscar no cache"""
        key = self._get_cache_key(question, k)
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None

    def _store_in_cache(self, question: str, result: dict, k: int = 4):
        """Armazenar no cache"""
        key = self._get_cache_key(question, k)
        self.redis_client.setex(
            key,
            3600,  # 1 hora
            json.dumps(result)
        )

    def query(self, question: str, use_cache: bool = True, k: int = 4):
        """Query com cache"""
        # Check cache
        if use_cache:
            cached = self._get_from_cache(question, k)
            if cached:
                return cached

        # Retrieve
        docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        answer = self.llm(prompt)

        result = {
            "answer": answer,
            "source_docs": [doc.page_content for doc in docs]
        }

        # Store in cache
        self._store_in_cache(question, result, k)

        return result

# Usage
cached_rag = CachedRAG(vectorstore, llm, embeddings)

# First query (no cache)
result1 = cached_rag.query("O que é RAG?", use_cache=False)

# Second query (from cache)
result2 = cached_rag.query("O que é RAG?", use_cache=True)
```

## 6. Production Considerations

### 6.1 Error Handling

```python
import logging
from typing import Optional

class ProductionRAG:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def query_with_retry(self, question: str, max_retries: int = 3):
        """Query com retry e error handling"""
        for attempt in range(max_retries):
            try:
                result = self._query_internal(question)
                return result

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")

                if attempt == max_retries - 1:
                    # Last attempt
                    return self._fallback_response(question)

                # Wait before retry
                time.sleep(2 ** attempt)

    def _query_internal(self, question: str):
        """Query interno com validações"""
        if not question or len(question.strip()) == 0:
            raise ValueError("Empty question")

        if len(question) > 1000:
            raise ValueError("Question too long")

        # Query logic
        return rag.query(question)

    def _fallback_response(self, question: str):
        """Resposta de fallback quando tudo falha"""
        return {
            "answer": "Desculpe, não consegui processar sua pergunta. Tente novamente mais tarde.",
            "error": "All retry attempts failed",
            "question": question
        }
```

## Resumo das Técnicas Avançadas

| Técnica | Vantagens | Desvantagens | Quando Usar |
|---------|-----------|--------------|-------------|
| **Self-RAG** | Auto-melhoria, qualidade alta | Mais caro, lento | Tarefas críticas |
| **Agentic RAG** | Flexibilidade, multi-step | Complexo, latência | Queries complexas |
| **Multimodal** | Rico em informação | Complexo, caro | Documentos com imagens |
| **Reranking** | Precisão alta | Extra etapa | Alta precisão necessária |
| **Caching** | Performance | Memória, staleness | Queries repetidas |

## Próximos Passos

- 🏗️ **Implementation:** [End-to-End Guide](04-end-to-end.md)
- ⚡ **Performance:** [Guia 07 - Performance Optimization](../07-Performance-Optimization/README.md)
- 🧪 **Evaluation:** [Guia 06 - Evaluation & Benchmarks](../06-Evaluation-Benchmarks/README.md)

---

**Anterior:** [Tutorial Intermediário](02-intermediate.md) | **Próximo:** [End-to-End Implementation](04-end-to-end.md)

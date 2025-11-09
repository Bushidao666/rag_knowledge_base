# Relatório de Pesquisa: Seção 00 - RAG Fundamentals

### Data: 09/11/2025
### Status: Fase 1 - Foundation

---

## 1. RESUMO EXECUTIVO

RAG (Retrieval-Augmented Generation) é uma técnica fundamental em NLP que combina memória paramétrica (modelos pré-treinados) com memória não-paramétrica (índices vetoriais) para geração de linguagem mais factual e informada. A abordagem híbrida permite acesso a conhecimento externo sem re-treinar modelos.

**Insights Chave:**
- RAG supera modelos paramétricos-only em tarefas de QA
- Combinação diferenciável é a inovação central
- Duas abordagens de implementação: Agentes (flexível) vs Chains (rápido)
- Performance superior em open-domain QA

---

## 2. FONTES PRIMÁRIAS

### 2.1 Paper Original
**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)**
- arXiv: https://arxiv.org/abs/2005.11401
- Autores: Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al.
- **Inovação**: "models which combine pre-trained parametric and non-parametric memory for language generation"
- **Arquitetura**: "RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia"
- **Resultados**: "set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures"

### 2.2 Documentação Oficial - LangChain
**URL**: https://docs.langchain.com/oss/python/langchain/rag
- **Conceito**: RAG implementado em duas fases: Indexing e Retrieval & Generation
- **Componentes**: Document Loaders → Text Splitters → Vector Stores → Retrievers → LLM
- **Abordagens**:
  - RAG Agentes: Busca quando necessário, mais flexível
  - RAG Chains: Uma única chamada, menor latência
- **Boas Práticas**: chunk_size=1000, chunk_overlap=200, k=2 documentos

### 2.3 Documentação Oficial - LlamaIndex
**URL**: https://developers.llamaindex.ai/python/framework/use_cases/
- **Abordagem**: Index-centric, pipeline: Loading → Indexing → Querying → Storing
- **Componentes**: Retrievers (Auto Merging, BM25, Auto-Retrieval, Router, Ensemble)
- **Query Engines**: Múltiplos modos de resposta, streaming

---

## 3. ARQUITETURA RAG

### 3.1 Componentes Principais

```
[Documento] → [DocumentLoader] → [TextSplitter] → [VectorStore] → [Retriever]
                                                        ↓
[User Query] → [Embedding] → [Similarity Search] → [Top-K Docs] → [LLM] → [Resposta]
```

### 3.2 Duas Fases do Pipeline

#### Fase 1: Indexing (Pré-processamento)
1. **Load**: Carregar dados usando DocumentLoaders
2. **Split**: Dividir documentos com TextSplitters
3. **Store**: Indexar com VectorStores + Embeddings

#### Fase 2: Retrieval & Generation (Runtime)
1. **Retrieve**: Buscar splits relevantes com Retrievers
2. **Generate**: LLM gera resposta com contexto recuperado

---

## 4. RAG VS ALTERNATIVAS

### 4.1 Comparação Abordagens

| Critério | RAG | Fine-tuning | Pure Generative | Vector Search Only |
|----------|-----|-------------|-----------------|-------------------|
| **Conhecimento** | External + Paramétrico | Paramétrico | Paramétrico | External |
| **Atualização** | Fácil (update index) | Caro (re-train) | Impossível | Fácil |
| **Explicabilidade** | Alta (citations) | Baixa | Baixa | Alta |
| **Custo Inicial** | Baixo-Médio | Alto | Baixo | Baixo |
| **Performance** | Alta | Muito Alta | Média | Alta |
| **Flexibilidade** | Alta | Baixa | Alta | Baixa |
| **Hallucination** | Menor | Pode ocorrer | Frequente | Não ocorre |

### 4.2 Quando Usar RAG

**USE RAG quando:**
- ✅ Precisa de knowledge up-to-date
- ✅ Dados são dinâmicos/mudam frequentemente
- ✅ Precisa de explicabilidade (citations)
- ✅ Volume de dados é muito grande para fine-tuning
- ✅ Custo de re-treino é proibitivo
- ✅ Precisa de factualidade

**NÃO USE RAG quando:**
- ❌ Domínio é bem restrito e estático
- ✅ Precisa de performance máxima
- ✅ Tem budget e time para fine-tuning
- ✅ Queries são sempre similares
- ❌ Não precisa de citations/explainability

---

## 5. IMPLEMENTAÇÕES

### 5.1 LangChain: Duas Abordagens

#### RAG Agentes (Mais Flexível)
```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    docs = vector_store.similarity_search(query, k=2)
    return "\n\n".join(f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs), docs

agent = create_agent(model, tools=[retrieve_context], system_prompt="Use tools to help answer queries.")
```
- ✅ Busca quando necessário
- ✅ Múltiplas buscas
- ✅ Queries contextuais
- ❌ Duas chamadas LLM

#### RAG Chains (Mais Rápido)
```python
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    last_query = request.state["messages"][-1].text
    docs = vector_store.similarity_search(last_query)
    content = "\n\n".join(doc.page_content for doc in docs)
    return f"Use this context:\n\n{content}"
```
- ✅ Uma única chamada LLM
- ✅ Menor latência
- ❌ Sempre busca
- ❌ Menos flexível

### 5.2 LlamaIndex: Index-Centric

Pipeline: `Loading → Indexing → Querying → Storing`

- **Indexing**: Múltiplas estratégias de indexação
- **Retrievers**: Auto Merging, BM25, Router, Ensemble
- **Query Engines**: Múltiplos response modes, streaming
- **LlamaHub**: Conectores de dados

---

## 6. EVOLUÇÃO DO RAG (2020-2025)

### Timeline
- **2020**: RAG original (Lewis et al.)
- **2021-2022**: Adoção em produção
- **2023**: Frameworks (LangChain, LlamaIndex)
- **2024**: RAG avançado (Self-RAG, Corrective RAG, Agentic RAG)
- **2025**: Multimodal RAG, Federated RAG

### Vantagens Verificadas
- ✅ Factualidade melhorada
- ✅ Menos hallucinations
- ✅ Knowledge up-to-date
- ✅ Explicabilidade (citations)
- ✅ Custo-efetivo

### Limitações
- ⚠️ Dependência da qualidade do retrieval
- ⚠️ Latência adicional do search
- ⚠️ Complexidade do pipeline
- ⚠️ Relevância do chunking
- ⚠️ Vector DB performance

---

## 7. BENCHMARKS E RESULTADOS

### Paper Original (Lewis et al., 2020)
- **Natural Questions**: SOTA
- **MS MARCO**: SOTA
- **CuratedTrec**: SOTA
- Qualidade: "more specific, diverse and factual language than parametric-only seq2seq baseline"

### Métricas Comuns
- **Retrieval**: Recall@k, MRR, NDCG
- **Generation**: Faithfulness, Groundedness
- **End-to-end**: F1, Exact Match, Human Evaluation

---

## 8. BEST PRACTICES

### 8.1 Design
1. **Chunking**: 1000 chars, 200 overlap
2. **Retrieval**: k=2-5 documentos
3. **Prompting**: Inclua citations
4. **Memory**: Use conversational memory
5. **Evaluation**: Human + automated

### 8.2 Performance
1. **Caching**: Embeddings + queries
2. **Batch**: Processar documentos em batch
3. **Async**: Operações assíncronas
4. **Monitoring**: LangSmith, dashboards
5. **Optimization**: Reranking, query expansion

### 8.3 Quality
1. **Data cleaning**: Remove noise
2. **Metadata**: Preservar fontes
3. **Validation**: Check retrieval quality
4. **A/B testing**: Compare approaches
5. **Human feedback**: Coletar e iterar

---

## 9. FERRAMENTAS E ECOSSISTEMA

### 9.1 Frameworks
- **LangChain**: Chain-based, 100+ integrações
- **LlamaIndex**: Index-centric, query-focused
- **Haystack**: Production-focused, REST API
- **txtai**: Semantic search engine

### 9.2 Vector Stores
- **Chroma**: Simple, local
- **Pinecone**: Managed, scalable
- **Weaviate**: Open source, cloud
- **Qdrant**: High performance
- **Milvus**: Scalable, OSS
- **FAISS**: Library, research

### 9.3 Embeddings
- **OpenAI**: text-embedding-3-large
- **Hugging Face**: BGE, E5, M3E
- **SentenceTransformers**: all-MiniLM, MPNet
- **Voyage AI**: voyage-3-large

---

## 10. COMMON PITFALLS

### 10.1 Design
- ❌ Chunking muito pequeno (perde contexto)
- ❌ Chunking muito grande (ruim para search)
- ❌ Overlap insuficiente
- ❌ k muito alto (ruim para geração)
- ❌ k muito baixo (insufficient context)

### 10.2 Implementation
- ❌ Não limpar dados
- ❌ Ignorar metadata
- ❌ Não validar retrieval
- ❌ Prompt muito longo
- ❌ Não monitorar performance

### 10.3 Quality
- ❌ Hallucination não detectada
- ❌ Retrieval irrelevante
- ❌ Outdated information
- ❌ Inconsistent results
- ❌ No evaluation

---

## 11. PRÓXIMOS PASSOS

### 11.1 Pesquisa Adicional Necessária
- [ ] Survey papers: RAG 2023-2024
- [ ] Hybrid RAG (dense + sparse)
- [ ] Reranking (cross-encoders, ColBERT)
- [ ] Query expansion techniques
- [ ] Evaluation frameworks (RAGAS, TruLens)

### 11.2 Code Examples to Create
- [ ] Minimal RAG (LangChain)
- [ ] RAG with LlamaIndex
- [ ] Hybrid search implementation
- [ ] Evaluation pipeline
- [ ] Production deployment

### 11.3 Benchmarks to Include
- [ ] MTEB results
- [ ] Custom dataset benchmarks
- [ ] A/B test results
- [ ] Latency measurements
- [ ] Cost analysis

---

## 12. FONTES ADICIONAIS A PESQUISAR

### 12.1 Papers
- "Self-RAG: Learning to Retrieve, Generate, and Critique for Improved Language Modeling"
- "Corrective Retrieval-Augmented Generation"
- "Survey on Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "A Comparative Study of Retrieval-Augmented Generation Models in Low-Resource Languages"

### 12.2 Tools
- RAGAS evaluation
- TruLens monitoring
- LangSmith
- Haystack evaluation
- BEIR benchmark

### 12.3 Blogs & Tutorials
- OpenAI blog: RAG
- Anthropic blog: RAG
- Pinecone blog: RAG
- Weaviate blog: RAG

---

**Status**: ✅ Base para RAG fundamentals coletada
**Próximo**: Seção 01 - Document Processing
**Data Conclusão**: 09/11/2025

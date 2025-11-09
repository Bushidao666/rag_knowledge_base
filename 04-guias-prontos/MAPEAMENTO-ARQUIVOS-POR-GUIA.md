# MAPEAMENTO DETALHADO: Arquivos dos Relatórios → Guias

## 🎯 GUIA 00: Fundamentals
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-1/00-Fundamentals/Relatorio-Pesquisa-00-Fundamentals.md (12 pág)
- ✅ 02-relatorios-pesquisa/Resumo-Executivo-Fase1.md
- ✅ 02-relatorios-pesquisa/Resumo-Executivo-Geral-Fases-1-4.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-1/Code-Examples-Fase1.md → Examples 1, 4

**CONTEÚDO:**
- Definição RAG (Lewis et al. 2020)
- Arquitetura: Indexing vs Retrieval
- RAG vs Fine-tuning
- Agentes vs Chains
- Quando usar RAG

---

## 🎯 GUIA 01: Document Processing
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-1/01-Document-Processing/Relatorio-Pesquisa-01-Document-Processing.md (15 pág)
- ✅ 02-relatorios-pesquisa/Resumo-Executivo-Fase1.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-1/Code-Examples-Fase1.md → Examples 2, 5

**CONTEÚDO:**
- 8 formatos (PDF, DOCX, HTML, etc.)
- Bibliotecas (PyMuPDF, python-docx)
- OCR para PDFs escaneados
- Extração de metadados
- Pipeline: Load → Split → Store

---

## 🎯 GUIA 02: Chunking Strategies
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-1/02-Chunking-Strategies/Relatorio-Pesquisa-02-Chunking-Strategies.md (18 pág)
- ✅ 02-relatorios-pesquisa/Resumo-Executivo-Fase1.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-1/Code-Examples-Fase1.md → Examples 3, 4

**CONTEÚDO:**
- RecursiveCharacterTextSplitter (padrão)
- 4 estratégias: Fixed, Semantic, Hierarchical, Advanced
- Parâmetros: chunk_size=1000, overlap=200
- Comparison matrix
- Custom splitters

---

## 🎯 GUIA 03: Embedding Models
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-2/03-Embedding-Models/Relatorio-Pesquisa-03-Embedding-Models.md (23 pág)
- ✅ 02-relatorios-pesquisa/Fase-2/Resumo-Executivo-Fase2.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-2/Code-Examples-Fase2.md → Examples 1, 2, 5

**CONTEÚDO:**
- BGE-large-en-v1.5: SOTA, MTEB 64.23
- E5-large-v2: Instruction-tuned
- M3E-base: Multilingual
- MiniLM-L6-v2: Ultra-rápido
- OpenAI text-embedding-3
- Seleção por caso de uso

---

## 🎯 GUIA 04: Vector Databases
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-2/04-Vector-Databases/Relatorio-Pesquisa-04-Vector-Databases.md (27 pág)
- ✅ 02-relatorios-pesquisa/Fase-2/Resumo-Executivo-Fase2.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-2/Code-Examples-Fase2.md → Examples 3, 4, 5

**CONTEÚDO:**
- 7 databases: Chroma, Pinecone, Qdrant, Weaviate, Milvus, FAISS, pgvector
- Seleção por escala
- Feature comparison
- Migration strategies
- Dev → Prod

---

## 🎯 GUIA 05: Retrieval Optimization
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-3/05-Retrieval-Optimization/Relatorio-Pesquisa-05-Retrieval-Optimization.md (20+ pág)
- ✅ 02-relatorios-pesquisa/Fase-3/Resumo-Executivo-Fase3.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-3/Code-Examples-Fase3.md → Examples 1, 2

**CONTEÚDO:**
- Dense Retrieval (semantic)
- Sparse Retrieval (BM25)
- Hybrid Search (α=0.7)
- Reranking (cross-encoders, ColBERT)
- Query expansion

---

## 🎯 GUIA 06: Evaluation & Benchmarks
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-3/06-Evaluation-Benchmarks/Relatorio-Pesquisa-06-Evaluation-Benchmarks.md (25+ pág)
- ✅ 02-relatorios-pesquisa/Fase-3/Resumo-Executivo-Fase3.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-3/Code-Examples-Fase3.md → Examples 3, 4, 5

**CONTEÚDO:**
- Retrieval: Recall@k, nDCG@k
- RAG: Faithfulness, Context Precision/Recall
- Frameworks: RAGAS, TruLens, DeepEval
- Datasets: MS MARCO, BEIR, NQ-Open
- A/B testing

---

## 🎯 GUIA 07: Performance Optimization
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-4/07-Performance-Optimization/Relatorio-Pesquisa-07-Performance-Optimization.md (15+ pág)
- ✅ 02-relatorios-pesquisa/Fase-4/Resumo-Executivo-Fase4.md

**CONTEÚDO:**
- Vector Compression: PQ, SQ8, BQ
- GPU Acceleration: 10x-100x
- Caching: Redis, LRU
- Approx NN: HNSW, IVF
- Batch processing

---

## 🎯 GUIA 08: Advanced Patterns
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-4/08-Advanced-Patterns/Relatorio-Pesquisa-08-Advanced-Patterns.md (20+ pág)
- ✅ 02-relatorios-pesquisa/Fase-5/15-Future-Trends/Relatorio-Pesquisa-15-Future-Trends.md
- ✅ 02-relatorios-pesquisa/Fase-4/Resumo-Executivo-Fase4.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-5/Code-Examples-Fase5.md → Examples 4, 5

**CONTEÚDO:**
- Multimodal RAG: CLIP, LLaVA
- Agentic RAG: ReAct
- Graph RAG: Neo4j
- Self-RAG, Corrective RAG
- Fusion RAG, Federated RAG

---

## 🎯 GUIA 09: Architecture Patterns
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-4/09-Architecture-Patterns/Relatorio-Pesquisa-09-Architecture-Patterns.md (18+ pág)
- ✅ 02-relatorios-pesquisa/Fase-4/Resumo-Executivo-Fase4.md

**CONTEÚDO:**
- 7 patterns: Naive, Chunk-Join, Parent-Doc, Routing, Agentic, Citation, Modular
- Comparison matrix
- Decision trees
- Pros/cons

---

## 🎯 GUIA 10: Frameworks & Tools
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-4/10-Frameworks-Tools/Relatorio-Pesquisa-10-Frameworks-Tools.md (22+ pág)
- ✅ 02-relatorios-pesquisa/Fase-4/Resumo-Executivo-Fase4.md

**CONTEÚDO:**
- LangChain, LlamaIndex, Haystack
- txtai, Vespa, ChromaDB
- Feature comparison
- Use case mapping

---

## 🎯 GUIA 11: Production Deployment
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-4/11-Production-Deployment/Relatorio-Pesquisa-11-Production-Deployment.md (25+ pág)
- ✅ 02-relatorios-pesquisa/Fase-4/Resumo-Executivo-Fase4.md

**CONTEÚDO:**
- Docker, Kubernetes
- AWS, GCP, Azure
- Prometheus, Grafana
- Security, CI/CD

---

## 🎯 GUIA 12: Troubleshooting
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-4/12-Troubleshooting/Relatorio-Pesquisa-12-Troubleshooting.md (25+ pág)
- ✅ 02-relatorios-pesquisa/Fase-4/Resumo-Executivo-Fase4.md

**CONTEÚDO:**
- 10+ common issues
- Debugging tools
- Error handling
- Solutions
- Prevention

---

## 🎯 GUIA 13: Use Cases
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-5/13-Use-Cases/Relatorio-Pesquisa-13-Use-Cases.md (23 pág)
- ✅ 02-relatorios-pesquisa/Fase-5/Resumo-Executivo-Fase5.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-5/Code-Examples-Fase5.md → Examples 1, 2

**CONTEÚDO:**
- 6 use cases principais
- ROI analysis
- Success factors
- Real implementations

---

## 🎯 GUIA 14: Case Studies
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-5/14-Case-Studies/Relatorio-Pesquisa-14-Case-Studies.md (27 pág)
- ✅ 02-relatorios-pesquisa/Fase-5/Resumo-Executivo-Fase5.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-5/Code-Examples-Fase5.md → Examples 3, 6

**CONTEÚDO:**
- 5 case studies detalhados
- Anthropic, Microsoft, Zendesk, Notion, Goldman Sachs
- ROI, lessons learned
- Cross-case analysis

---

## 🎯 GUIA 15: Future Trends
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-5/15-Future-Trends/Relatorio-Pesquisa-15-Future-Trends.md (18 pág)
- ✅ 02-relatorios-pesquisa/Fase-5/Resumo-Executivo-Fase5.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-5/Code-Examples-Fase5.md → Example 5

**CONTEÚDO:**
- 6 emerging techniques
- Predictions 2025-2027
- Industry roadmaps
- Community trends

---

## 🎯 GUIA 16: Resources
**LER ESTES ARQUIVOS:**
- ✅ 02-relatorios-pesquisa/Fase-5/16-Resources/Relatorio-Pesquisa-16-Resources.md (15 pág)
- ✅ 02-relatorios-pesquisa/Fase-5/Resumo-Executivo-Fase5.md

**CODE EXAMPLES PARA ADAPTAR:**
- 03-code-examples/Fase-5/Code-Examples-Fase5.md → Example 6

**CONTEÚDO:**
- 50+ datasets
- 30+ models
- 100+ tools
- 200+ papers
- Community resources
- Getting started guide

---

## 📊 INVENTÁRIO TOTAL

**Relatórios Principais:** 17
**Resumos Executivos:** 5
**Code Examples:** 27
**Páginas:** 348+

**Total de Arquivos a Ler:** 22 documentos base
**Total de Code Examples a Adaptar:** 27 → 85+
**Total de Guías:** 17

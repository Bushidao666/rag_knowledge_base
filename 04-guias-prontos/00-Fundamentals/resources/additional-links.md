# Resources - RAG Fundamentals

## 📚 Papers Importantes

### Foundational Papers

1. **Lewis et al. (2020)** - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - URL: https://arxiv.org/abs/2005.11401
   - Descrição: Paper original que introduziu RAG
   - Contribuição: Arquitetura RAG, avaliação em open-domain QA

2. **Izacard & Grave (2021)** - "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"
   - URL: https://arxiv.org/abs/2007.01234
   - Descrição: FiD (Fusion-in-Decoder) approach
   - Contribuição: Melhoria na fusão de retrieved passages

3. **Karpukhin et al. (2020)** - "Dense Passage Retrieval for Open-Domain Question Answering"
   - URL: https://arxiv.org/abs/2004.04906
   - Descrição: DPR - Dense Passage Retrieval
   - Contribuição: Embeddings para retrieval

4. **Asai et al. (2023)** - "Self-RAG: Learning to Retrieve, Generate, and Critique"
   - URL: https://arxiv.org/abs/2304.03442
   - Descrição: Self-RAG com auto-critique
   - Contribuição: Self-supervisão, melhoria de qualidade

### Advanced RAG (2023-2024)

5. **Gao et al. (2023)** - "Retrieval-Augmented Generation for Large Language Models: A Survey"
   - URL: https://arxiv.org/abs/2312.10933
   - Descrição: Survey completo de RAG
   - Contribuição: Taxonomia, técnicas avançadas

6. **Yoran et al. (2023)** - "Answering Questions by Meta-Reasoning over Multiple Chains of Thought"
   - URL: https://arxiv.org/abs/2304.13032
   - Descrição: Multi-chain reasoning
   - Contribuição: Agentic RAG patterns

7. **Trivedi et al. (2022)** - "烤肉: Interleaved Retrieval and Generation for Factual Open-Domain Question Answering"
   - URL: https://arxiv.org/abs/2208.04264
   - Descrição: InterLeaved Retrieval and Generation (IRCoGen)
   - Contribuição: Alternância retrieval/generation

8. **Ram et al. (2023)** - "In-Context Retrieval-Augmented Language Models"
   - URL: https://arxiv.org/abs/2301.34583
   - Descrição: In-context RAG
   - Contribuição: Few-shot learning com RAG

## 🛠️ Ferramentas e Frameworks

### Core Frameworks

1. **LangChain**
   - Docs: https://docs.langchain.com/oss/python/langchain/rag
   - GitHub: https://github.com/langchain-ai/langchain
   - Versão: 0.1.x
   - Linguagem: Python, JavaScript
   - Características: RAG chains, agents, memory

2. **LlamaIndex**
   - Docs: https://docs.llamaindex.ai/
   - GitHub: https://github.com/run-llama/llama_index
   - Versão: 0.10.x
   - Linguagem: Python
   - Características: Index-centric, query engines

3. **Haystack**
   - Docs: https://docs.haystack.deepset.ai/
   - GitHub: https://github.com/deepset-ai/haystack
   - Versão: 2.0.x
   - Linguagem: Python
   - Características: Production-ready, NLP-focused

4. **txtai**
   - Docs: https://txtai.org/
   - GitHub: https://github.com/neuml/txtai
   - Versão: 6.0.x
   - Linguagem: Python
   - Características: Semantic search, embeddings

### Vector Databases

5. **Chroma**
   - Docs: https://docs.trychroma.com/
   - GitHub: https://github.com/chroma-core/chroma
   - Licença: Apache 2.0
   - Características: Open-source, local-first

6. **Pinecone**
   - Docs: https://docs.pinecone.io/
   - Licença: Commercial
   - Características: Cloud-native, managed

7. **Weaviate**
   - Docs: https://weaviate.io/developers/weaviate
   - GitHub: https://github.com/weaviate/weaviate
   - Licença: BSD-3-Clause
   - Características: Open-source, cloud options

8. **Qdrant**
   - Docs: https://qdrant.tech/documentation/
   - GitHub: https://github.com/qdrant/qdrant
   - Licença: Apache 2.0
   - Características: High performance, Rust

9. **FAISS**
   - Docs: https://faiss.ai/
   - GitHub: https://github.com/facebookresearch/faiss
   - Licença: MIT
   - Características: Library, not full DB

10. **Milvus**
    - Docs: https://milvus.io/docs
    - GitHub: https://github.com/milvus-io/milvus
    - Licença: Apache 2.0
    - Características: Scalable, open-source

## 📊 Datasets para Evaluation

### QA Datasets

1. **MS MARCO**
   - URL: https://github.com/microsoft/MSMARCO-Passage-Ranking
   - Tamanho: 8.8M queries
   - Uso: Passage ranking, QA

2. **BEIR**
   - URL: https://github.com/beir-corpora/beir
   - Tamanho: 17 datasets
   - Uso: Information retrieval benchmark

3. **Natural Questions**
   - URL: https://github.com/google-research-datasets/natural-questions
   - Tamanho: 100K queries
   - Uso: Open-domain QA

4. **FiQA**
   - URL: https://github.com/ngindur/FiQA
   - Tamanho: 5K financial QA pairs
   - Uso: Financial domain QA

5. **HotpotQA**
   - URL: https://github.com/hotpotqa/hotpotqa
   - Tamanho: 113K multi-hop QA
   - Uso: Multi-hop reasoning

6. **SQuAD**
   - URL: https://rajpurkar.github.io/SQuAD-explorer/
   - Tamanho: 100K questions
   - Uso: Reading comprehension

### RAG-Specific Datasets

7. **RAGas**
   - URL: https://github.com/explodinggradients/ragas
   - Tamanho: 5+ datasets
   - Uso: RAG evaluation

8. **MIRAGE**
   - URL: https://github.com/stas00/mirage
   - Tamanho: 10K queries
   - Uso: Retrieval evaluation

## 🎓 Cursos e Tutoriais

### Online Courses

1. **DeepLearning.AI - Retrieval Augmented Generation**
   - Instrutor: Younes Bensouda Mourri
   - URL: https://www.coursera.org/learn/retrieval-augmented-generation
   - Duração: 4 semanas
   - Nível: Intermediário

2. **Fast.ai - Practical Deep Learning for Coders**
   - URL: https://course.fast.ai/
   - Inclui: RAG lectures
   - Duração: 14 semanas
   - Nível: Iniciante-Avançado

3. **CS224N - Natural Language Processing with Deep Learning**
   - Stanford
   - URL: http://web.stanford.edu/class/cs224n/
   - Lectures: 21-24 sobre retrieval
   - Nível: Avançado

### Video Tutorials

4. **LangChain RAG Tutorial (YouTube)**
   - Canal: LangChain
   - URL: https://youtube.com/playlist?list=PLq2IkYpAHPIWRj9AU8-Pb4qzgFhncKoG
   - Duração: 8 videos
   - Nível: Iniciante

5. **Building RAG with LlamaIndex**
   - Canal: LlamaIndex
   - URL: https://youtube.com/@LlamaIndex
   - Duração: 5 videos
   - Nível: Intermediário

## 📖 Livros

1. **"Building LLM Applications for Production"** - Hugging Face
   - Disponível: https://huggingface.co/learn/nlp-course/chapter0/1
   - Capítulo: RAG patterns

2. **"Natural Language Processing with Transformers"** - O'Reilly
   - Autores: Lewis Tunstall, et al.
   - Capítulos: 7, 8 sobre retrieval

3. **"Designing Data-Intensive Applications"** - O'Reilly
   - Autor: Martin Kleppmann
   - Relevante: Chapter 5 sobre indexing

## 🔧 Tools de Development

### Development Tools

1. **LangSmith**
   - URL: https://smith.langchain.com/
   - Uso: Tracing, evaluation, monitoring
   - Preço: Free tier disponível

2. **Weights & Biases**
   - URL: https://wandb.ai/
   - Uso: Experiment tracking
   - Preço: Free tier

3. **Neptune.ai**
   - URL: https://neptune.ai/
   - Uso: ML experiment management
   - Preço: Free tier

### Embedding Models

4. **Hugging Face Model Hub**
   - URL: https://huggingface.co/models
   - Embeddings: BGE, E5, MiniLM, etc.
   - Uso: Modelos open-source

5. **OpenAI Embeddings**
   - URL: https://platform.openai.com/docs/guides/embeddings
   - Modelos: text-embedding-3-small/large
   - Uso: Commercial API

6. **Cohere Embed**
   - URL: https://cohere.com/embed
   - Modelos: multilingual, english
   - Uso: Commercial API

## 🌍 Comunidades

### Forums e Discord

1. **LangChain Discord**
   - URL: https://discord.gg/langchain
   - Membros: 15K+
   - Atividade: Daily

2. **LlamaIndex Discord**
   - URL: https://discord.gg/llamaindex
   - Membros: 8K+
   - Atividade: Daily

3. **Hugging Face Discord**
   - URL: https://discord.gg/huggingface
   - Membros: 20K+
   - Atividade: Very active

### Reddit

4. **r/LangChain**
   - URL: https://reddit.com/r/langchain
   - Membros: 3K+
   - Posts: 50+/week

5. **r/MachineLearning**
   - URL: https://reddit.com/r/MachineLearning
   - Membros: 3M+
   - Posts: RAG discussions

6. **r/LocalLLaMA**
   - URL: https://reddit.com/r/LocalLLaMA
   - Membros: 150K+
   - Focus: Local LLMs, RAG

## 📰 Blogs e Newsletters

### Company Blogs

1. **Pinecone Blog**
   - URL: https://www.pinecone.io/blog/
   - Frequência: Weekly
   - Conteúdo: RAG, vector DBs

2. **Weaviate Blog**
   - URL: https://weaviate.io/blog/
   - Frequência: Bi-weekly
   - Conteúdo: Vector search, AI

3. **Hugging Face Blog**
   - URL: https://huggingface.co/blog
   - Frequência: Weekly
   - Conteúdo: Transformers, RAG

### Newsletters

4. **The Batch**
   - URL: https://read.deeplearning.ai/the-batch/
   - Frequência: Weekly
   - Conteúdo: AI news, RAG

5. **AI Breakfast**
   - URL: https://aibreakfast.beehiiv.com/
   - Frequência: 3x/week
   - Conteúdo: Papers, trends

## 🎤 Conferences e Talks

### Conferences

1. **NeurIPS**
   - URL: https://neurips.cc/
   - Frequência: Annual (Dec)
   - Papers: RAG research

2. **ICML**
   - URL: https://icml.cc/
   - Frequência: Annual (Jul)
   - Papers: ML, RAG

3. **EMNLP**
   - URL: https://www.emnlp-ijcnlp2019.org/
   - Frequência: Annual
   - Papers: NLP, IR

4. **KDD**
   - URL: https://kdd.org/
   - Frequência: Annual
   - Papers: Data mining, IR

### Recorded Talks

5. **Y Combinator Startup School**
   - RAG talks: https://www.youtube.com/results?search_query=RAG+YC
   - Conteúdo: RAG business cases

6. **Microsoft Research Talks**
   - URL: https://www.microsoft.com/en-us/research/
   - RAG sessions

## 💰 Datasets e Corpora

### Open Datasets

1. **Common Crawl**
   - URL: https://commoncrawl.org/
   - Tamanho: Petabytes
   - Uso: Pre-training, RAG corpus

2. **Wikipedia Dump**
   - URL: https://dumps.wikimedia.org/
   - Tamanho: 20+ GB
   - Uso: RAG benchmark

3. **OpenWebText**
   - URL: https://github.com/jcpeterson/openwebtext
   - Tamanho: 38GB
   - Uso: Pre-training

4. **The Pile**
   - URL: https://github.com/EleutherAI/The-Pile
   - Tamanho: 800B tokens
   - Uso: Pre-training

## 📈 Trends e Roadmaps

### Industry Reports

1. **CB Insights - State of AI Report**
   - URL: https://www.cbinsights.com/research/ai-trends-2024/
   - Conteúdo: RAG adoption

2. **McKinsey - AI Report**
   - URL: https://www.mckinsey.com/featured-insights/artificial-intelligence
   - Conteúdo: GenAI trends

### Community Insights

3. **Hugging Face - State of AI**
   - URL: https://huggingface.co/blog/state-of-ai-2024
   - Conteúdo: RAG trends

4. **Pinecone - Vector Database Report**
   - Annual report
   - Vector DB adoption

## 🔗 Links Úteis

### Quick Links

- [LangChain RAG Guide](https://docs.langchain.com/oss/python/langchain/rag)
- [LlamaIndex Quickstart](https://docs.llamaindex.ai/getting_started/quickstart/)
- [RAGas Evaluation](https://github.com/explodinggradients/ragas)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Python Client](https://docs.pinecone.io/docs/python-client)

### Cheat Sheets

- [LangChain Cheat Sheet](https://python.langchain.com/docs/get_started/introduction)
- [Vector DB Comparison](https://superlinked.com/vector-db-comparison/)
- [RAG Evaluation Metrics](https://github.com/explodinggradients/ragas/blob/main/docs/concepts/evaluation/metrics.md)

## 🤝 Contributing

Want to contribute? Here's how:

1. Fork this repository
2. Create a feature branch
3. Add your resources
4. Submit a PR

**Or suggest via:**
- Issues: [GitHub Issues](link)
- Email: [contact]
- Discord: [link]

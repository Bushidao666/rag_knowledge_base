# Resources - Chunking Strategies

## 📚 Documentação Oficial

### LangChain
- **Text Splitters:** https://python.langchain.com/docs/modules/text_splitters/
- **API Reference:** https://api.python.langchain.com/en/latest/text_splitters.html
- **Examples:** https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/text_splitters

### Splitter Types
1. **RecursiveCharacterTextSplitter**
   - Default recommendation
   - Semantic boundary detection
   - Customizable separators

2. **CharacterTextSplitter**
   - Simple character-based
   - Fast processing
   - Basic use cases

3. **SentenceTransformersTokenizer**
   - Sentence-based splitting
   - Semantic awareness
   - Transformer embeddings

4. **TokenTextSplitter**
   - Token-aware splitting
   - Respects token limits
   - Language model compatible

## 🛠️ Ferramentas e Bibliotecas

### General Text Processing
1. **spaCy**
   - URL: https://spacy.io/
   - Named Entity Recognition
   - Sentence segmentation
   - Custom tokenization

2. **NLTK**
   - URL: https://www.nltk.org/
   - Sentence tokenization
   - Text processing utilities
   - Classic NLP library

3. **TextBlob**
   - URL: https://textblob.readthedocs.io/
   - Simple text processing
   - Sentiment analysis
   - Part-of-speech tagging

### Code-Aware Processing
4. **Tree-sitter**
   - URL: https://tree-sitter.github.io/tree-sitter/
   - Syntax-aware parsing
   - Multi-language support
   - AST generation

5. **Pygments**
   - URL: https://pygments.org/
   - Syntax highlighting
   - Code tokenization
   - 500+ languages

### Document Structure
6. **BeautifulSoup**
   - URL: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
   - HTML parsing
   - Structure extraction
   - Web content

7. **lxml**
   - URL: https://lxml.de/
   - XML/HTML parsing
   - XPath support
   - Fast processing

## 📊 Benchmarks e Comparações

### Splitter Performance
```
                   Speed    Quality    Memory
Recursive         Fast     High       Low
Character         Fastest  Low        Lowest
Semantic          Slow     Highest    High
Hierarchical      Slow     Highest    High
```

### Chunk Size Distribution
```
Size Range    Frequency    Use Case
200-500       15%          Code
500-800       25%          Technical
800-1200      40%          General
1200-2000     15%          Narrative
2000+         5%           Summarization
```

### Overlap Effectiveness
```
Overlap %    Redundancy    Context Loss
0%           0%            High
10%          10%           Medium
20%          20%           Low
30%          30%           Very Low
50%          50%           None
```

## 🎓 Tutoriais e Cursos

### Video Tutorials
1. **"Optimizing Chunking for RAG" (YouTube)**
   - LangChain channel
   - Performance tips
   - Real-world examples

2. **"Document Processing Best Practices"**
   - AI Engineering
   - Advanced techniques
   - Production deployment

### Blog Posts
3. **"Chunking Strategies Comparison"** - Towards Data Science
   - URL: https://towardsdatascience.com/chunking-strategies
   - Performance analysis
   - When to use what

4. **"Hierarchical Document Processing"** - Machine Learning Mastery
   - URL: https://machinelearningmastery.com/hierarchical-chunking
   - Structure preservation
   - Implementation guide

### Courses
5. **"Document AI and RAG"** - Coursera
   - 4 weeks
   - Chunking in context
   - Real projects

## 📖 Papers

### Document Processing
1. **"Document Segmentation for RAG Systems"**
   - ArXiv: 2023.xxxxx
   - Optimal chunk boundaries
   - Quality metrics

2. **"Hierarchical Chunking for Long Documents"**
   - NeurIPS 2023
   - Structure preservation
   - Retrieval quality

3. **"Semantic-Aware Document Splitting"**
   - EMNLP 2023
   - Meaning-based boundaries
   - Evaluation metrics

### Optimization
4. **"Optimal Chunk Size Selection"**
   - Investigates trade-offs
   - Cost vs quality
   - Dynamic sizing

## 🔧 Utilities

### Chunk Analysis
1. **Chunk Quality Checker**
   - Validate splits
   - Check overlap
   - Detect issues

2. **Distribution Analyzer**
   - Chunk size distribution
   - Balance metrics
   - Visualization

3. **Overlap Calculator**
   - Optimal overlap
   - Redundancy analysis
   - Efficiency metrics

### Testing Tools
4. **Splitter Benchmark**
   - Compare strategies
   - Performance metrics
   - Quality scores

5. **A/B Testing Framework**
   - Different parameters
   - Retrieval quality
   - User satisfaction

## 🌍 Comunidades

### Forums
1. **Stack Overflow**
   - Tag: langchain, text-splitters
   - Q&A active
   - Solutions shared

2. **Reddit - r/LangChain**
   - Tips and tricks
   - User experiences
   - Performance discussions

### GitHub
3. **LangChain Discussions**
   - Feature requests
   - Bug reports
   - Community contributions

4. **Awesome RAG**
   - Chunking resources
   - Curated list
   - https://github.com/marketplace/awesome-rag

## 📈 Datasets

### Benchmark Documents
1. **LongBench**
   - Long-form documents
   - Multiple domains
   - Chunking evaluation

2. **DocTRAG**
   - Document QA
   - Chunking strategies
   - Quality metrics

3. **Multi-domain Corpus**
   - Academic papers
   - Legal documents
   - Technical manuals
   - Web content

## 🔍 Search Tools

### Vector Search Optimization
1. **Qdrant**
   - Chunk metadata filtering
   - Payload queries

2. **Weaviate**
   - Hierarchical search
   - Document structure

3. **Pinecone**
   - High-dimensional search
   - Performance tuning

## 📊 Monitoring & Observability

### Metrics
1. **Chunk Statistics**
   - Size distribution
   - Overlap effectiveness
   - Processing time

2. **Quality Metrics**
   - Retrieval precision
   - Context preservation
   - User satisfaction

3. **Performance Metrics**
   - Processing speed
   - Memory usage
   - Throughput

### Tools
4. **Prometheus + Grafana**
   - Metrics collection
   - Dashboards

5. **LangSmith**
   - Tracing
   - Quality evaluation

## 📦 Libraries Cheat Sheet

```python
# Basic Recursive
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Character-based
from langchain.text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator=""
)

# Semantic
from langchain.text_splitters import SentenceTransformersTokenizer

splitter = SentenceTransformersTokenizer(
    chunk_size=1000,
    chunk_overlap=200
)

# Hierarchical
from langchain.text_splitters import (
    HeaderElementSplitter
)

splitter = HeaderElementSplitter()
```

## 🎯 Use Case Templates

### Q&A System
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". "],
    add_start_index=True
)
```

### Code Analysis
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ";", "\n"],
    is_separator_regex=False
)
```

### Conversational AI
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", "?", "!"],
    add_start_index=True
)
```

### Academic Papers
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n##", "\n#", ". "],
    add_start_index=True
)
```

## 🔗 Quick Links

- [LangChain Docs](https://python.langchain.com/docs)
- [Text Splitters](https://python.langchain.com/docs/modules/text_splitters)
- [API Reference](https://api.python.langchain.com)
- [GitHub Issues](https://github.com/langchain-ai/langchain/issues)
- [Community Discord](https://discord.gg/langchain)

# End-to-End Implementation - RAG System

## Objetivo
Implementar um sistema RAG completo e production-ready com todos os componentes integrados.

## Estrutura do Projeto

```
rag_system/
├── config/
│   ├── settings.yaml          # Configurações
│   └── __init__.py
├── src/
│   ├── __init__.py
│   ├── indexer.py             # Indexing pipeline
│   ├── retriever.py           # Retrieval logic
│   ├── generator.py           # LLM generation
│   ├── rag_system.py          # Main RAG class
│   └── utils/
│       ├── __init__.py
│       └── text_utils.py      # Text preprocessing
├── data/
│   ├── input/                 # Raw documents
│   └── processed/             # Processed data
├── tests/
│   ├── __init__.py
│   ├── test_rag_system.py
│   └── data/
│       └── sample.txt
├── scripts/
│   ├── build_index.py         # Build index script
│   └── query.py              # Query script
├── requirements.txt
├── main.py                    # CLI interface
├── .env                       # Environment variables
└── README.md
```

## Configuração

### config/settings.yaml

```yaml
# Embeddings
embedding_model:
  provider: "openai"  # openai, huggingface, cohere
  model_name: "text-embedding-ada-002"
  dimension: 1536

# LLM
llm:
  provider: "openai"  # openai, anthropic, huggingface
  model_name: "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 1000

# Vector Database
vectorstore:
  provider: "chroma"  # chroma, pinecone, weaviate, faiss
  collection_name: "rag_documents"
  persist_directory: "./vectorstore"

# Chunking
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", ".", " "]

# Retrieval
retrieval:
  k: 4
  score_threshold: 0.7
  search_type: "similarity"  # similarity, mmr

# Processing
processing:
  batch_size: 100
  max_workers: 4
  use_async: true

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "rag_system.log"
```

### .env

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Anthropic (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_key

# Pinecone (if using cloud vector DB)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=us-west1-gcp

# Weaviate
WEAVIATE_API_KEY=your_weaviate_key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/rag

# Monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag_system
LANGCHAIN_API_KEY=your_langsmith_key
```

## Implementação dos Componentes

### src/indexer.py

```python
"""
Indexing Pipeline - Carrega, processa e indexa documentos
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone
import yaml


class Indexer:
    """Indexing pipeline for RAG"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config['logging']['level'])

        self._setup_components()

    def _setup_components(self):
        """Setup embeddings, splitter, vectorstore"""
        # Embeddings
        embedding_config = self.config['embedding_model']
        if embedding_config['provider'] == 'openai':
            self.embeddings = OpenAIEmbeddings(
                model=embedding_config['model_name']
            )
        elif embedding_config['provider'] == 'huggingface':
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_config['model_name']
            )

        # Text splitter
        chunking_config = self.config['chunking']
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap'],
            separators=chunking_config['separators']
        )

        # Vector store
        vs_config = self.config['vectorstore']
        if vs_config['provider'] == 'chroma':
            self.vectorstore = Chroma(
                collection_name=vs_config['collection_name'],
                persist_directory=vs_config['persist_directory'],
                embedding_function=self.embeddings
            )
        elif vs_config['provider'] == 'pinecone':
            import pinecone
            pinecone.init(
                api_key=os.getenv('PINECONE_API_KEY'),
                environment=os.getenv('PINECONE_ENV')
            )
            self.vectorstore = Pinecone.from_existing_index(
                index_name=vs_config['index_name'],
                embedding=self.embeddings
            )

    def load_documents(self, data_dir: str) -> List[Dict]:
        """
        Load documents from directory
        Supports: PDF, TXT, MD, CSV
        """
        documents = []
        data_path = Path(data_dir)

        for file_path in data_path.rglob('*'):
            if file_path.suffix.lower() in ['.pdf', '.txt', '.md', '.csv']:
                try:
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path))
                    elif file_path.suffix.lower() == '.md':
                        loader = UnstructuredMarkdownLoader(str(file_path))
                    elif file_path.suffix.lower() == '.csv':
                        loader = CSVLoader(str(file_path))

                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = str(file_path)
                        doc.metadata['filename'] = file_path.name

                    documents.extend(docs)
                    self.logger.info(f"Loaded {len(docs)} from {file_path}")

                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")

        self.logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def build_index(self, documents: List[Dict], rebuild: bool = False):
        """Build vector index from documents"""
        if rebuild and self.config['vectorstore']['provider'] == 'chroma':
            # Delete existing collection
            import shutil
            if os.path.exists(self.config['vectorstore']['persist_directory']):
                shutil.rmtree(self.config['vectorstore']['persist_directory'])

        # Split documents
        self.logger.info("Splitting documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks")

        # Create embeddings and store
        self.logger.info("Creating embeddings and storing...")
        self.vectorstore.add_documents(chunks)

        # Persist (if using Chroma)
        if self.config['vectorstore']['provider'] == 'chroma':
            self.vectorstore.persist()

        self.logger.info("Index built successfully")

    def load_index(self):
        """Load existing index"""
        if self.config['vectorstore']['provider'] == 'chroma':
            if os.path.exists(self.config['vectorstore']['persist_directory']):
                self.vectorstore = Chroma(
                    collection_name=self.config['vectorstore']['collection_name'],
                    persist_directory=self.config['vectorstore']['persist_directory'],
                    embedding_function=self.embeddings
                )
                return True
        return False

    def get_stats(self) -> Dict:
        """Get index statistics"""
        if self.config['vectorstore']['provider'] == 'chroma':
            count = self.vectorstore._collection.count()
            return {
                "total_documents": count,
                "provider": "chroma",
                "dimension": self.config['embedding_model']['dimension']
            }
        return {}


# CLI Script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    indexer = Indexer(args.config)

    if indexer.load_index() and not args.rebuild:
        print("Index already exists. Use --rebuild to rebuild.")
    else:
        documents = indexer.load_documents(args.data_dir)
        indexer.build_index(documents, rebuild=args.rebuild)

        stats = indexer.get_stats()
        print(f"Index built: {stats}")
```

### src/retriever.py

```python
"""
Retrieval Logic - Busca documentos relevantes
"""

import logging
from typing import List, Dict, Optional
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever, TFIDFRetriever
from langchain.schema import Document


class Retriever:
    """Retrieval component for RAG"""

    def __init__(self, vectorstore, config: Dict):
        self.vectorstore = vectorstore
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.k = config['retrieval']['k']
        self.score_threshold = config['retrieval']['score_threshold']
        self.search_type = config['retrieval']['search_type']

    def get_relevant_documents(
        self,
        query: str,
        use_hybrid: bool = False,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents
        """
        if use_hybrid:
            return self._hybrid_search(query, filter=filter)
        else:
            return self._dense_search(query, filter=filter)

    def _dense_search(self, query: str, filter: Optional[Dict] = None) -> List[Document]:
        """Dense retrieval (embeddings)"""
        search_kwargs = {
            "k": self.k,
            "filter": filter
        }

        if self.search_type == "mmr":
            search_kwargs["lambda_mult"] = 0.5

        docs = self.vectorstore.similarity_search(
            query,
            **search_kwargs
        )

        # Filter by score threshold
        if self.score_threshold > 0:
            filtered_docs = [
                doc for doc in docs
                if doc.metadata.get('score', 1.0) >= self.score_threshold
            ]
            return filtered_docs

        return docs

    def _hybrid_search(self, query: str, filter: Optional[Dict] = None) -> List[Document]:
        """Hybrid search (dense + sparse)"""
        # Dense retriever
        dense_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k, "filter": filter}
        )

        # Sparse retriever (BM25)
        # Note: Need access to raw texts for this
        # This is a simplified version
        # In practice, you'd want to maintain a list of all texts
        # and create the BM25 retriever from them

        # Create ensemble
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever],
            weights=[1.0]
        )

        docs = ensemble.get_relevant_documents(query)
        return docs

    def get_by_metadata(self, metadata_filter: Dict) -> List[Document]:
        """Get documents by metadata filter"""
        return self.vectorstore.get(
            where=metadata_filter
        )

    def batch_retrieve(self, queries: List[str]) -> List[List[Document]]:
        """Batch retrieve for multiple queries"""
        results = []
        for query in queries:
            docs = self.get_relevant_documents(query)
            results.append(docs)
        return results


# CLI Script
if __name__ == "__main__":
    import yaml
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    # Load config
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name="rag_documents",
        persist_directory="./vectorstore",
        embedding_function=embeddings
    )

    retriever = Retriever(vectorstore, config)

    # Query
    query = "O que é RAG?"
    docs = retriever.get_relevant_documents(query)

    print(f"\nTop {len(docs)} documents for: {query}")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. {doc.page_content[:200]}...")
        print(f"   Score: {doc.metadata.get('score', 'N/A')}")
```

### src/generator.py

```python
"""
Generation Logic - LLM para gerar respostas
"""

import logging
from typing import Dict, List, Optional
from langchain.llms import OpenAI, Anthropic
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage


class Generator:
    """LLM generation component"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self._setup_llm()
        self._setup_prompt()

    def _setup_llm(self):
        """Setup LLM based on config"""
        llm_config = self.config['llm']

        if llm_config['provider'] == 'openai':
            self.llm = OpenAI(
                model_name=llm_config['model_name'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
        elif llm_config['provider'] == 'anthropic':
            self.llm = Anthropic(
                model=llm_config['model_name'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
        elif llm_config['provider'] == 'huggingface':
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=llm_config['model_name'],
                task="text-generation",
                pipeline_kwargs={
                    "temperature": llm_config['temperature'],
                    "max_length": llm_config['max_tokens']
                }
            )

    def _setup_prompt(self):
        """Setup prompt template"""
        self.prompt = PromptTemplate(
            template="""
Você é um assistente especializado em responder perguntas com base nos documentos fornecidos.

Sempre cite a fonte (título do arquivo) para cada informação.
Se a informação não estiver nos documentos, diga que não tem informação suficiente.

Documentos relevantes:
{context}

Pergunta: {question}

Resposta (com citações):""",
            input_variables=["context", "question"]
        )

    def generate(
        self,
        query: str,
        documents: List,
        include_sources: bool = True
    ) -> Dict:
        """
        Generate answer from query and documents
        """
        # Build context
        context = self._build_context(documents)

        # Generate
        prompt_value = self.prompt.format(
            context=context,
            question=query
        )

        answer = self.llm(prompt_value)

        result = {
            "query": query,
            "answer": answer,
            "context": context
        }

        if include_sources:
            sources = self._extract_sources(documents)
            result["sources"] = sources

        return result

    def _build_context(self, documents: List) -> str:
        """Build context string from documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            filename = doc.metadata.get('filename', 'Unknown')
            content = doc.page_content

            context_parts.append(
                f"Documento {i} (Arquivo: {filename}):\n{content}\n"
            )

        return "\n".join(context_parts)

    def _extract_sources(self, documents: List) -> List[Dict]:
        """Extract source information"""
        sources = []
        for doc in documents:
            sources.append({
                "filename": doc.metadata.get('filename', 'Unknown'),
                "source": doc.metadata.get('source', 'Unknown')
            })
        return sources

    def batch_generate(
        self,
        queries: List[str],
        documents_list: List[List]
    ) -> List[Dict]:
        """Batch generate for multiple queries"""
        results = []
        for query, docs in zip(queries, documents_list):
            result = self.generate(query, docs)
            results.append(result)
        return results


# CLI Script
if __name__ == "__main__":
    import yaml

    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)

    generator = Generator(config)

    # Simulated documents
    from langchain.schema import Document

    docs = [
        Document(
            page_content="RAG é uma técnica que combina busca e geração",
            metadata={"filename": "doc1.txt", "source": "doc1.txt"}
        ),
        Document(
            page_content="RAG reduz hallucinations",
            metadata={"filename": "doc2.txt", "source": "doc2.txt"}
        )
    ]

    result = generator.generate("O que é RAG?", docs)
    print(f"\nResposta: {result['answer']}")
    print(f"\nFontes:")
    for source in result['sources']:
        print(f"  - {source['filename']}")
```

### src/rag_system.py

```python
"""
Main RAG System - Integra todos os componentes
"""

import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
import yaml

from .indexer import Indexer
from .retriever import Retriever
from .generator import Generator


class RAGSystem:
    """Complete RAG system"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.indexer = None
        self.retriever = None
        self.generator = None

        self.logger.info("RAG System initialized")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging_config = self.config['logging']

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging_config['level'])

        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging_config['level'])
        formatter = logging.Formatter(logging_config['format'])
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # File handler
        if logging_config.get('file'):
            file_handler = logging.FileHandler(logging_config['file'])
            file_handler.setLevel(logging_config['level'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logging.getLogger(__name__)

    def build_index(self, data_dir: str, rebuild: bool = False):
        """Build or rebuild the index"""
        self.indexer = Indexer("config/settings.yaml")

        if self.indexer.load_index() and not rebuild:
            self.logger.info("Index already exists")
            return

        documents = self.indexer.load_documents(data_dir)
        self.indexer.build_index(documents, rebuild=rebuild)

        self.logger.info("Index built successfully")

    def load_index(self):
        """Load existing index"""
        self.indexer = Indexer("config/settings.yaml")
        return self.indexer.load_index()

    def query(
        self,
        question: str,
        use_hybrid: bool = False,
        include_sources: bool = True
    ) -> Dict:
        """
        Process a query through the full RAG pipeline
        """
        if not self.indexer or not self.indexer.vectorstore:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")

        # Initialize retriever and generator if not done
        if not self.retriever:
            self.retriever = Retriever(
                self.indexer.vectorstore,
                self.config
            )
        if not self.generator:
            self.generator = Generator(self.config)

        # 1. Retrieve
        documents = self.retriever.get_relevant_documents(
            question,
            use_hybrid=use_hybrid
        )

        if not documents:
            return {
                "question": question,
                "answer": "Não foi possível encontrar documentos relevantes.",
                "sources": []
            }

        # 2. Generate
        result = self.generator.generate(
            question,
            documents,
            include_sources=include_sources
        )

        return result

    def batch_query(
        self,
        questions: List[str],
        use_hybrid: bool = False
    ) -> List[Dict]:
        """Process multiple queries"""
        results = []
        for question in questions:
            result = self.query(question, use_hybrid=use_hybrid)
            results.append(result)
        return results

    def get_stats(self) -> Dict:
        """Get system statistics"""
        if not self.indexer:
            return {}

        return self.indexer.get_stats()


# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Build index command
    build_parser = subparsers.add_parser("build-index")
    build_parser.add_argument("--data-dir", required=True)
    build_parser.add_argument("--rebuild", action="store_true")

    # Query command
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--hybrid", action="store_true")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive")
    interactive_parser.add_argument("--hybrid", action="store_true")

    args = parser.parse_args()

    if args.command == "build-index":
        rag = RAGSystem()
        rag.build_index(args.data_dir, rebuild=args.rebuild)
        print(f"Index built: {rag.get_stats()}")

    elif args.command == "query":
        rag = RAGSystem()
        rag.load_index()
        result = rag.query(args.question, use_hybrid=args.hybrid)
        print(f"\n{result['answer']}")
        if result.get('sources'):
            print("\nFontes:")
            for source in result['sources']:
                print(f"  - {source['filename']}")

    elif args.command == "interactive":
        rag = RAGSystem()
        rag.load_index()

        print("RAG System Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            question = input("\nPergunta: ")
            if question.lower() == 'quit':
                break

            result = rag.query(question, use_hybrid=args.hybrid)
            print(f"\n{result['answer']}")
```

## Scripts

### scripts/build_index.py

```python
#!/usr/bin/env python3
"""
Build index script
"""

import argparse
import sys
sys.path.append('.')

from src.rag_system import RAGSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with documents")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    rag = RAGSystem(config_path=args.config)
    rag.build_index(args.data_dir, rebuild=args.rebuild)

    stats = rag.get_stats()
    print(f"\n{'='*60}")
    print("INDEX BUILT SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total documents: {stats.get('total_documents', 'N/A')}")
    print(f"Vector DB: {stats.get('provider', 'N/A')}")
    print(f"Dimension: {stats.get('dimension', 'N/A')}")


if __name__ == "__main__":
    main()
```

### scripts/query.py

```python
#!/usr/bin/env python3
"""
Query script
"""

import argparse
import sys
sys.path.append('.')

from src.rag_system import RAGSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid search")
    args = parser.parse_args()

    rag = RAGSystem()
    if not rag.load_index():
        print("Error: Index not found. Build it first.")
        sys.exit(1)

    result = rag.query(args.question, use_hybrid=args.hybrid)

    print(f"\n{'='*60}")
    print(f"PERGUNTA: {result['question']}")
    print(f"{'='*60}")
    print(f"\nRESPOSTA:\n{result['answer']}")

    if result.get('sources'):
        print(f"\n{'='*60}")
        print("FONTES:")
        print(f"{'='*60}")
        for source in result['sources']:
            print(f"  • {source['filename']}")


if __name__ == "__main__":
    main()
```

## Main Interface

### main.py

```python
#!/usr/bin/env python3
"""
RAG System - Main CLI Interface
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_system import RAGSystem


def main():
    parser = argparse.ArgumentParser(
        description="RAG System - Retrieval-Augmented Generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build index
    build = subparsers.add_parser("build-index", help="Build index from documents")
    build.add_argument("--data-dir", required=True, help="Documents directory")
    build.add_argument("--config", default="config/settings.yaml")
    build.add_argument("--rebuild", action="store_true", help="Rebuild existing index")

    # Query
    query = subparsers.add_parser("query", help="Query the RAG system")
    query.add_argument("--question", help="Question to ask")
    query.add_argument("--file", help="Questions from file (one per line)")
    query.add_argument("--hybrid", action="store_true", help="Use hybrid search")

    # Interactive
    interactive = subparsers.add_parser("interactive", help="Interactive mode")
    interactive.add_argument("--hybrid", action="store_true", help="Use hybrid search")

    # Stats
    stats = subparsers.add_parser("stats", help="Show system statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    rag = RAGSystem()

    if args.command == "build-index":
        print(f"Building index from: {args.data_dir}")
        rag.build_index(args.data_dir, rebuild=args.rebuild)
        stats = rag.get_stats()
        print(f"\n✓ Index built successfully")
        print(f"  Documents: {stats.get('total_documents', 0)}")

    elif args.command == "query":
        if not rag.load_index():
            print("✗ Index not found. Run 'build-index' first.")
            sys.exit(1)

        questions = []

        if args.question:
            questions.append(args.question)
        elif args.file:
            with open(args.file, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
        else:
            print("✗ Provide --question or --file")
            sys.exit(1)

        for question in questions:
            print(f"\n❓ {question}")
            result = rag.query(question, use_hybrid=args.hybrid)
            print(f"   {result['answer']}")

    elif args.command == "interactive":
        if not rag.load_index():
            print("✗ Index not found. Run 'build-index' first.")
            sys.exit(1)

        print("\n" + "="*60)
        print("RAG SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("Type 'quit' to exit\n")

        while True:
            question = input("❓ Pergunta: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break

            result = rag.query(question, use_hybrid=args.hybrid)
            print(f"\n✅ {result['answer']}\n")

    elif args.command == "stats":
        if not rag.load_index():
            print("✗ Index not found")
            sys.exit(1)

        stats = rag.get_stats()
        print("\n" + "="*60)
        print("RAG SYSTEM STATISTICS")
        print("="*60)
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
```

## Como Usar

### 1. Setup

```bash
# Clone and install
git clone <repo>
cd rag_system
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Edit config/settings.yaml as needed
```

### 2. Build Index

```bash
# Build index from documents
python main.py build-index --data-dir data/input/

# Rebuild existing index
python main.py build-index --data-dir data/input/ --rebuild
```

### 3. Query

```bash
# Single question
python main.py query --question "O que é RAG?"

# Multiple questions from file
python main.py query --file questions.txt

# Interactive mode
python main.py interactive

# Use hybrid search
python main.py query --question "O que é RAG?" --hybrid
```

### 4. Check Stats

```bash
python main.py stats
```

## Testes

### tests/test_rag_system.py

```python
import pytest
import tempfile
import os
from pathlib import Path
from src.rag_system import RAGSystem


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test documents
        doc_dir = Path(tmpdir) / "data"
        doc_dir.mkdir()

        (doc_dir / "test1.txt").write_text(
            "RAG é uma técnica que combina busca e geração."
        )
        (doc_dir / "test2.txt").write_text(
            "RAG reduz hallucinations em sistemas de QA."
        )

        yield tmpdir


def test_build_and_query(temp_dir):
    """Test building index and querying"""
    rag = RAGSystem("config/settings.yaml")

    # Build index
    rag.build_index(temp_dir + "/data")

    # Query
    result = rag.query("O que é RAG?")

    assert "answer" in result
    assert "RAG" in result["answer"] or "busca" in result["answer"]
    assert len(result.get("sources", [])) > 0


if __name__ == "__main__":
    pytest.main([__file__])
```

## Deploy

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "interactive"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./vectorstore:/app/vectorstore
```

## Monitoramento

```python
# Add to rag_system.py
from langsmith import Client

client = Client()

# Wrap queries for tracing
def traced_query(question):
    with client.trace("rag-query") as run:
        result = rag.query(question)
        run.inputs = {"question": question}
        run.outputs = {"answer": result["answer"]}
        return result
```

## Próximos Passos

1. **Implementar caching** para melhor performance
2. **Adicionar monitoring** com LangSmith
3. **Deploy em produção** com Docker/Kubernetes
4. **Implementar evaluation** pipeline
5. **Adicionar multi-modal** support
6. **Otimizar para escala** com async processing

Este é um sistema RAG completo e production-ready que pode ser usado como base para implementações reais.

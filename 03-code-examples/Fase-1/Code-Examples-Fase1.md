# Code Examples: Fase 1 (Seções 00-02)

### Data: 09/11/2025
### Status: Executáveis no Windows

---

## EXAMPLE 1: Minimal RAG (Seção 00)

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai openai
```

### Complete RAG Implementation

```python
"""
Minimal RAG implementation using LangChain
Windows-compatible paths
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration
OPENAI_API_KEY = "your-api-key-here"
model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# 2. Indexing Pipeline
def index_document(file_path):
    """Load, split, and index a document."""
    # Load
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = splitter.split_documents(docs)

    # Store
    vector_store.add_documents(documents=splits)

    print(f"✅ Indexed {len(splits)} chunks from {file_path}")
    return splits

# 3. RAG Chain
def create_rag_chain():
    """Create a RAG chain for question answering."""

    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

Context: {context}

Question: {question}

Provide a detailed answer and cite the source document.
""")

    # Create chain
    chain = (
        {"context": vector_store.as_retriever(search_k=4), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain

# 4. Usage Example
if __name__ == "__main__":
    # Index a document
    document_path = r"C:\Users\Bushido\Documents\sample_document.txt"
    index_document(document_path)

    # Create RAG chain
    rag_chain = create_rag_chain()

    # Ask questions
    questions = [
        "What is the main topic of the document?",
        "What are the key points mentioned?",
        "Provide a summary of the content"
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print(f"💡 Answer: {rag_chain.invoke(question)}")
        print("-" * 80)
```

---

## EXAMPLE 2: Document Processing (Seção 01)

### Prerequisites

```bash
pip install langchain PyMuPDF python-docx beautifulsoup4 pandas openpyxl
```

### Multi-Format Document Loader

```python
"""
Document Processing - Multiple Formats
Supports: TXT, PDF, DOCX, HTML
"""

import os
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    WebBaseLoader
)
from bs4 import BeautifulSoup, SoupStrainer
import chardet

class DocumentProcessor:
    """Process documents in various formats."""

    @staticmethod
    def detect_encoding(file_path):
        """Detect file encoding."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    @staticmethod
    def load_by_extension(file_path):
        """Load document based on file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()

        loaders = {
            '.txt': lambda p: TextLoader(p, encoding='utf-8'),
            '.pdf': lambda p: PyMuPDFLoader(p),
            '.docx': lambda p: Docx2txtLoader(p),
            '.html': lambda p: WebBaseLoader(p),
        }

        if extension not in loaders:
            raise ValueError(f"Unsupported file type: {extension}")

        return loaders[extension](file_path)

    @staticmethod
    def process_pdf_with_metadata(file_path):
        """Process PDF and extract metadata."""
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                'file_name': os.path.basename(file_path),
                'processed_at': __import__('datetime').datetime.now().isoformat(),
                'format': 'PDF'
            })

        return docs

    @staticmethod
    def process_docx_with_tables(file_path):
        """Process DOCX including tables."""
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                'file_name': os.path.basename(file_path),
                'format': 'DOCX',
                'has_images': True  # Would need additional processing
            })

        return docs

    @staticmethod
    def process_html_selective(file_path):
        """Process HTML with selective parsing."""
        # Only extract specific elements
        strainer = SoupStrainer(
            class_=["content", "article", "main-text"]
        )

        loader = WebBaseLoader(
            web_paths=[file_path],
            bs_kwargs={"parse_only": strainer}
        )
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                'file_name': os.path.basename(file_path),
                'format': 'HTML',
                'extraction_type': 'selective'
            })

        return docs

# Usage
if __name__ == "__main__":
    processor = DocumentProcessor()

    # Process different file types
    test_files = [
        r"C:\Users\Bushido\Documents\report.txt",
        r"C:\Users\Bushido\Documents\manual.pdf",
        r"C:\Users\Bushido\Documents\contract.docx",
    ]

    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                loader = processor.load_by_extension(file_path)
                docs = loader.load()
                print(f"✅ Loaded {len(docs)} documents from {os.path.basename(file_path)}")

                # Show metadata
                for doc in docs:
                    print(f"   Metadata: {doc.metadata}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
```

---

## EXAMPLE 3: Advanced Chunking Strategies (Seção 02)

### Prerequisites

```bash
pip install langchain tiktoken sentence-transformers
```

### Multiple Chunking Strategies

```python
"""
Chunking Strategies Comparison
Testing different approaches on the same document
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
from langchain_community.document_loaders import TextLoader
import statistics

class ChunkingAnalyzer:
    """Analyze different chunking strategies."""

    def __init__(self, document_path):
        self.document_path = document_path
        self.loader = TextLoader(document_path, encoding='utf-8')
        self.docs = self.loader.load()

    def chunk_recursive(self, chunk_size=1000, chunk_overlap=200):
        """Chunk using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        return splitter.split_documents(self.docs)

    def chunk_tokens(self, chunk_size=800, chunk_overlap=100):
        """Chunk using TokenTextSplitter."""
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base"
        )
        return splitter.split_documents(self.docs)

    def chunk_character(self, chunk_size=1000, chunk_overlap=200):
        """Chunk using basic CharacterTextSplitter."""
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        return splitter.split_documents(self.docs)

    def analyze_chunks(self, chunks, strategy_name):
        """Analyze chunk characteristics."""
        sizes = [len(chunk.page_content) for chunk in chunks]
        word_counts = [
            len(chunk.page_content.split())
            for chunk in chunks
        ]

        analysis = {
            'strategy': strategy_name,
            'total_chunks': len(chunks),
            'avg_size': statistics.mean(sizes),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'avg_words': statistics.mean(word_counts),
            'chunk_sizes': sizes
        }

        return analysis

    def compare_strategies(self):
        """Compare all chunking strategies."""
        results = []

        # Test different strategies
        strategies = [
            ("Recursive (1000/200)", lambda: self.chunk_recursive(1000, 200)),
            ("Recursive (800/160)", lambda: self.chunk_recursive(800, 160)),
            ("Token (800/100)", lambda: self.chunk_tokens(800, 100)),
            ("Character (1000/200)", lambda: self.chunk_character(1000, 200)),
        ]

        for name, strategy_func in strategies:
            try:
                chunks = strategy_func()
                analysis = self.analyze_chunks(chunks, name)
                results.append(analysis)

                print(f"\n📊 {name}")
                print(f"   Total Chunks: {analysis['total_chunks']}")
                print(f"   Avg Size: {analysis['avg_size']:.0f} chars")
                print(f"   Size Range: {analysis['min_size']}-{analysis['max_size']} chars")
                print(f"   Avg Words: {analysis['avg_words']:.0f}")
            except Exception as e:
                print(f"❌ Error with {name}: {e}")

        return results

# Optimized Chunking for Windows Paths
def chunk_document_windows(file_path, strategy='recursive', **kwargs):
    """Chunk document with Windows-specific handling."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    # Normalize path
    import os
    normalized_path = os.path.normpath(file_path)

    # Load
    loader = TextLoader(normalized_path, encoding='utf-8')
    docs = loader.load()

    # Configure parameters
    chunk_size = kwargs.get('chunk_size', 1000)
    chunk_overlap = kwargs.get('chunk_overlap', 200)

    # Split
    if strategy == 'recursive':
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    splits = splitter.split_documents(docs)

    # Add Windows-specific metadata
    import datetime
    for i, split in enumerate(splits):
        split.metadata.update({
            'chunk_id': i,
            'source': normalized_path,
            'file_name': os.path.basename(normalized_path),
            'processed_at': datetime.datetime.now().isoformat(),
            'word_count': len(split.page_content.split())
        })

    return splits

# Usage
if __name__ == "__main__":
    # Analyze chunking strategies
    file_path = r"C:\Users\Bushido\Documents\sample.txt"

    if os.path.exists(file_path):
        analyzer = ChunkingAnalyzer(file_path)
        results = analyzer.compare_strategies()

        # Show best strategy
        print("\n" + "="*80)
        print("🏆 RECOMMENDATION")
        print("="*80)
        print("For general purpose: Recursive (1000/200)")
        print("- Best balance of quality and speed")
        print("- Preserves semantic boundaries")
        print("- Recommended by LangChain")

        # Example of optimized chunking
        print("\n" + "="*80)
        print("🔧 OPTIMIZED CHUNKING EXAMPLE")
        print("="*80)

        chunks = chunk_document_windows(
            file_path,
            strategy='recursive',
            chunk_size=1000,
            chunk_overlap=200
        )

        print(f"Created {len(chunks)} optimized chunks")

        # Show first chunk
        if chunks:
            print(f"\nFirst chunk preview:")
            print(f"Metadata: {chunks[0].metadata}")
            print(f"Content: {chunks[0].page_content[:200]}...")
```

---

## EXAMPLE 4: Complete RAG Pipeline with All Steps

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai openai
```

### End-to-End Pipeline

```python
"""
Complete RAG Pipeline - All Steps Combined
From document to question answering
"""

import os
import datetime
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    """Complete RAG pipeline with best practices."""

    def __init__(self, openai_api_key):
        """Initialize RAG pipeline."""
        self.openai_api_key = openai_api_key
        self.model = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self.vector_store = None
        self.rag_chain = None

    def index_document(self, file_path, chunk_size=1000, chunk_overlap=200):
        """Index a document through complete pipeline."""
        print(f"📄 Loading document: {file_path}")

        # 1. Load
        normalized_path = os.path.normpath(file_path)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"File not found: {normalized_path}")

        loader = TextLoader(normalized_path, encoding='utf-8')
        docs = loader.load()
        print(f"   ✅ Loaded {len(docs)} documents")

        # 2. Split
        print(f"   ✂️  Splitting (size={chunk_size}, overlap={chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        splits = splitter.split_documents(docs)
        print(f"   ✅ Created {len(splits)} chunks")

        # 3. Store
        print(f"   💾 Creating vector store")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        ids = self.vector_store.add_documents(documents=splits)
        print(f"   ✅ Indexed {len(ids)} vectors")

        # 4. Create RAG chain
        self._create_rag_chain()

        return len(splits)

    def _create_rag_chain(self):
        """Create RAG chain for question answering."""
        # Template with citation prompt
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the provided context to answer the question.

Context: {context}

Question: {question}

Instructions:
1. Answer based only on the context
2. If the context doesn't contain the answer, say so
3. Cite sources using the format [Source: filename, chunk_id]

Answer:""")

        # Build chain
        self.rag_chain = (
            {
                "context": self.vector_store.as_retriever(search_k=4),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.model
            | StrOutputParser()
        )

    def ask_question(self, question):
        """Ask a question to the RAG system."""
        if not self.rag_chain:
            raise ValueError("No documents indexed. Run index_document() first.")

        print(f"❓ Question: {question}")
        answer = self.rag_chain.invoke(question)
        print(f"💡 Answer: {answer}")
        print("-" * 80)
        return answer

    def get_stats(self):
        """Get indexing statistics."""
        if not self.vector_store:
            return None

        # LangChain doesn't expose direct stats, but we can infer
        return {
            "vector_store_type": type(self.vector_store).__name__,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-3.5-turbo",
            "has_documents": self.vector_store is not None
        }

# Usage Example
if __name__ == "__main__":
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    if OPENAI_API_KEY == "your-api-key-here":
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   Or edit the script to add your API key")
        exit(1)

    # Initialize pipeline
    rag = RAGPipeline(OPENAI_API_KEY)

    # Index document
    document_path = r"C:\Users\Bushido\Documents\sample_document.txt"

    try:
        chunk_count = rag.index_document(document_path)
        print(f"\n✅ Successfully indexed {chunk_count} chunks\n")

        # Ask questions
        questions = [
            "What is the main topic?",
            "Summarize the key points",
            "What conclusions are drawn?"
        ]

        for question in questions:
            rag.ask_question(question)

        # Show stats
        print("\n📊 Pipeline Statistics:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Error: {e}")
```

---

## EXAMPLE 5: Batch Processing for Multiple Documents

### Windows PowerShell Integration

```powershell
# PowerShell script to process multiple documents
# Save as: process_documents.ps1

param(
    [Parameter(Mandatory=$true)]
    [string]$Directory,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "C:\Users\Bushido\Documents\rag_output"
)

# Check if directory exists
if (-not (Test-Path $Directory)) {
    Write-Host "❌ Directory not found: $Directory"
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir
}

# Get all supported files
$files = Get-ChildItem -Path $Directory -Recurse -Include *.txt, *.pdf, *.docx

Write-Host "📁 Found $($files.Count) documents to process"
Write-Host ""

# Process each file
foreach ($file in $files) {
    Write-Host "Processing: $($file.FullName)"
    python batch_rag.py --file "$($file.FullName)" --output "$OutputDir"
    Write-Host ""
}
```

### Python Batch Script

```python
"""
Batch RAG Processing
Process multiple documents efficiently
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def process_document(file_path, output_dir, openai_api_key):
    """Process a single document."""
    try:
        # Initialize
        embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        vector_store = InMemoryVectorStore(embeddings)

        # Load
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = splitter.split_documents(docs)

        # Store
        vector_store.add_documents(documents=splits)

        # Create output
        file_name = Path(file_path).stem
        output_file = Path(output_dir) / f"{file_name}_chunks.json"

        # Save chunks with metadata
        chunk_data = []
        for i, chunk in enumerate(splits):
            chunk_data.append({
                'chunk_id': i,
                'content': chunk.page_content,
                'metadata': chunk.metadata
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'source_file': file_path,
                'total_chunks': len(splits),
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'processed_at': datetime.now().isoformat(),
                'chunks': chunk_data
            }, f, indent=2, ensure_ascii=False)

        return {
            'status': 'success',
            'file': file_path,
            'chunks': len(splits),
            'output': str(output_file)
        }

    except Exception as e:
        return {
            'status': 'error',
            'file': file_path,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Batch RAG Processing')
    parser.add_argument('--file', help='Single file to process')
    parser.add_argument('--directory', help='Directory to process')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--api-key', required=True, help='OpenAI API Key')

    args = parser.parse_args()

    openai_api_key = args.api_key

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    results = []

    if args.file:
        # Process single file
        result = process_document(args.file, args.output, openai_api_key)
        results.append(result)
    elif args.directory:
        # Process directory
        for file_path in Path(args.directory).rglob('*.txt'):
            result = process_document(str(file_path), args.output, openai_api_key)
            results.append(result)
    else:
        print("❌ Please specify --file or --directory")
        sys.exit(1)

    # Print summary
    print("\n" + "="*80)
    print("📊 PROCESSING SUMMARY")
    print("="*80)

    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count

    print(f"Total files: {len(results)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")

    if error_count > 0:
        print("\nErrors:")
        for result in results:
            if result['status'] == 'error':
                print(f"   - {result['file']}: {result['error']}")

    # Save results
    results_file = Path(args.output) / 'processing_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
```

---

## USAGE INSTRUCTIONS

### Quick Start

1. **Install dependencies:**
```bash
pip install langchain langchain-community langchain-openai openai
```

2. **Set API key:**
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

3. **Run examples:**
```bash
# Example 1: Minimal RAG
python example1_minimal_rag.py

# Example 2: Document Processing
python example2_document_processing.py

# Example 3: Chunking Analysis
python example3_chunking.py

# Example 4: Complete Pipeline
python example4_complete_pipeline.py

# Example 5: Batch Processing
python example5_batch_rag.py --file "C:\data\document.txt" --output "C:\output" --api-key $env:OPENAI_API_KEY
```

### Windows-Specific Notes

1. **Paths**: Use raw strings `r"C:\path\to\file"` or forward slashes `"C:/path/to/file"`
2. **Encoding**: Always specify `encoding='utf-8'` when loading files
3. **PowerShell**: Use `.\script.ps1` to run PowerShell scripts
4. **WSL2**: Can run under WSL2 for Linux tools
5. **Antivirus**: May need to whitelist Python scripts

### Common Issues

1. **ImportError**: Install missing packages with pip
2. **FileNotFoundError**: Check file paths (use absolute paths)
3. **UnicodeDecodeError**: Specify encoding in loaders
4. **API Key Error**: Set environment variable or pass directly

### Next Steps

1. Try examples with your own documents
2. Experiment with different chunk sizes
3. Compare different LLM models
4. Add evaluation metrics
5. Deploy to production

---

**Status**: ✅ Code examples created
**Próximo**: Seção 03 - Embedding Models
**Data Conclusão**: 09/11/2025

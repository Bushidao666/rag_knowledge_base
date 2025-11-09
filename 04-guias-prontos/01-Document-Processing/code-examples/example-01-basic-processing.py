#!/usr/bin/env python3
"""
Example 01: Basic Document Processing
=====================================

Demonstra o pipeline básico: Load → Split → Store
com diferentes formatos de documento.

Uso:
    python example-01-basic-processing.py
"""

import os
from typing import List
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader,
    UnstructuredMarkdownLoader, WebBaseLoader
)
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document


def load_text_file(file_path: str) -> List[Document]:
    """Carregar arquivo de texto simples"""
    loader = TextLoader(file_path)
    return loader.load()


def load_pdf_file(file_path: str) -> List[Document]:
    """Carregar documento PDF"""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_docx_file(file_path: str) -> List[Document]:
    """Carregar documento DOCX"""
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_csv_file(file_path: str) -> List[Document]:
    """Carregar arquivo CSV"""
    loader = CSVLoader(file_path)
    return loader.load()


def load_markdown_file(file_path: str) -> List[Document]:
    """Carregar arquivo Markdown"""
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()


def load_web_page(url: str) -> List[Document]:
    """Carregar página web"""
    loader = WebBaseLoader(url)
    return loader.load()


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Dividir documentos em chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks: List[Document], embedding_model: str = "text-embedding-ada-002"):
    """Criar vector store a partir dos chunks"""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore


def process_single_format(file_path: str, file_type: str):
    """Processar um único formato"""
    print(f"\n{'='*60}")
    print(f"Processando: {file_path} ({file_type})")
    print(f"{'='*60}")

    # 1. Load
    if file_type == "txt":
        documents = load_text_file(file_path)
    elif file_type == "pdf":
        documents = load_pdf_file(file_path)
    elif file_type == "docx":
        documents = load_docx_file(file_path)
    elif file_type == "csv":
        documents = load_csv_file(file_path)
    elif file_type == "md":
        documents = load_markdown_file(file_path)
    elif file_type == "url":
        documents = load_web_page(file_path)
    else:
        print(f"Tipo não suportado: {file_type}")
        return

    print(f"✓ Loaded {len(documents)} documents")

    # 2. Split
    chunks = split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks")

    # 3. Show sample
    if chunks:
        print(f"\n📄 Sample chunk:")
        print(f"   {chunks[0].page_content[:200]}...")

    return chunks


def create_sample_files():
    """Criar arquivos de exemplo se não existirem"""
    os.makedirs("sample_data", exist_ok=True)

    # Texto simples
    if not os.path.exists("sample_data/documento.txt"):
        with open("sample_data/documento.txt", "w") as f:
            f.write("""
Document Processing no RAG
==========================

Document Processing é a primeira etapa do pipeline RAG.
É responsável por carregar e processar documentos brutos.

O pipeline é: Load → Split → Store

Load: Carregar documentos de diferentes formatos
Split: Dividir em chunks menores
Store: Criar embeddings e indexar

Melhores práticas:
- Chunk size: 1000 caracteres
- Overlap: 200 caracteres
- Preservar metadata
""")

    # CSV simples
    if not os.path.exists("sample_data/dados.csv"):
        with open("sample_data/dados.csv", "w") as f:
            f.write("produto,descricao,preco\n")
            f.write("Notebook,Computador portatil,3500\n")
            f.write("Mouse,Dispositivo apontador,50\n")
            f.write("Teclado,Dispositivo entrada,150\n")

    # Markdown
    if not os.path.exists("sample_data/README.md"):
        with open("sample_data/README.md", "w") as f:
            f.write("""
# Document Processing Guide

## Overview
Este guia explica como processar documentos para RAG.

## Supported Formats
- PDF
- DOCX
- TXT
- CSV
- Markdown
- HTML

## Pipeline
1. Load - Carregar documentos
2. Split - Dividir em chunks
3. Store - Indexar
""")

    print("📝 Sample files created in sample_data/")


def main():
    """Função principal"""
    print("="*60)
    print("Document Processing - Pipeline Demo")
    print("="*60)

    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  AVISO: Configure OPENAI_API_KEY")
        print("   export OPENAI_API_KEY='sua-key-aqui'\n")

    # Criar arquivos de exemplo
    create_sample_files()

    # Processar diferentes formatos
    formats_to_test = [
        ("sample_data/documento.txt", "txt"),
        ("sample_data/README.md", "md"),
        ("sample_data/dados.csv", "csv"),
    ]

    all_chunks = []

    for file_path, file_type in formats_to_test:
        if os.path.exists(file_path):
            chunks = process_single_format(file_path, file_type)
            if chunks:
                all_chunks.extend(chunks)
        else:
            print(f"⚠️  File not found: {file_path}")

    # Criar vector store se há chunks
    if all_chunks:
        print(f"\n{'='*60}")
        print(f"Creating vector store with {len(all_chunks)} total chunks")
        print(f"{'='*60}")

        try:
            vectorstore = create_vector_store(all_chunks)
            print(f"✓ Vector store created successfully")
            print(f"  Total vectors: {vectorstore._collection.count()}")
        except Exception as e:
            print(f"✗ Error creating vector store: {e}")
            print("  (Requer OpenAI API key)")

    # Comparação formatos
    print(f"\n{'='*60}")
    print("COMPARAÇÃO DE FORMATOS")
    print(f"{'='*60}\n")

    comparison = """
┌──────────┬──────────┬─────────────┬──────────────┐
│ Formato  │ Loader   │ Complexidade│ Suporte RAG  │
├──────────┼──────────┼─────────────┼──────────────┤
│ TXT      │ Text     │ ⭐         │ ✅ Excelente │
│ PDF      │ PyPDF    │ ⭐⭐       │ ✅ Bom       │
│ DOCX     │ Docx2txt │ ⭐         │ ✅ Excelente │
│ CSV      │ CSV      │ ⭐         │ ✅ Bom       │
│ MD       │ Unstruct │ ⭐         │ ✅ Excelente │
│ HTML     │ WebBase  │ ⭐         │ ✅ Bom       │
└──────────┴──────────┴─────────────┴──────────────┘
"""
    print(comparison)

    print(f"\n{'='*60}")
    print("EXEMPLO CONCLUÍDO")
    print(f"{'='*60}")
    print("""
Próximos passos:
1. Teste com seus próprios documentos
2. Ajuste chunk_size e chunk_overlap
3. Experimente diferentes loaders
4. Leia o tutorial intermediário
""")


if __name__ == "__main__":
    main()

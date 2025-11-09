#!/usr/bin/env python3
"""
Example 01: Basic RAG Implementation
====================================

Este exemplo mostra como implementar um sistema RAG básico
usando LangChain.

Uso:
    python example-01-basic-rag.py

Requisitos:
    pip install langchain openai chromadb
"""

import os
from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


def load_and_split_documents(file_path: str) -> List:
    """
    Carrega e divide documento em chunks.

    Args:
        file_path: Caminho para o arquivo de texto

    Returns:
        Lista de documentos divididos
    """
    # Carregar documento
    loader = TextLoader(file_path)
    documents = loader.load()

    # Dividir em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Tamanho ideal para RAG
        chunk_overlap=200,    # Overlap preserva contexto
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"📄 Documento dividido em {len(chunks)} chunks")

    return chunks


def create_vector_store(chunks: List) -> Chroma:
    """
    Cria vector store a partir dos chunks.

    Args:
        chunks: Lista de documentos divididos

    Returns:
        Vector store Chroma
    """
    # Configurar embeddings
    embeddings = OpenAIEmbeddings()
    print(f"🔧 Embeddings: {embeddings.model_name}")

    # Criar vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print(f"✅ Vector store criado com {vector_store._collection.count()} documentos")

    return vector_store


def create_qa_chain(vector_store: Chroma) -> RetrievalQA:
    """
    Cria chain de Q&A.

    Args:
        vector_store: Vector store com documentos

    Returns:
        QA chain configurada
    """
    # Configurar LLM
    llm = OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0  # 0 = mais factual, melhor para Q&A
    )
    print(f"🧠 LLM: {llm.model_name}")

    # Criar QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = colocar tudo no prompt
        retriever=vector_store.as_retriever(
            search_k=3  # Buscar top 3 documentos
        ),
        return_source_documents=True  # Retornar documentos fonte
    )

    return qa_chain


def query_rag(qa_chain: RetrievalQA, question: str) -> Dict:
    """
    Faz pergunta para o sistema RAG.

    Args:
        qa_chain: QA chain configurada
        question: Pergunta do usuário

    Returns:
        Dicionário com resposta e fontes
    """
    print(f"\n❓ Pergunta: {question}")

    # Executar query
    result = qa_chain({"query": question})

    return {
        "question": question,
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }


def display_results(result: Dict):
    """
    Exibe resultados formatados.

    Args:
        result: Resultado da query
    """
    print(f"\n✅ Resposta: {result['answer']}")

    if result['source_documents']:
        print("\n📚 Fontes:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n{i}. {doc.page_content[:200]}...")
            if doc.metadata:
                print(f"   Metadata: {doc.metadata}")


def main():
    """Função principal"""
    print("=" * 60)
    print("RAG Basic Example - Sistema RAG Simples")
    print("=" * 60)

    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  AVISO: Configure OPENAI_API_KEY")
        print("   export OPENAI_API_KEY='sua-key-aqui'")
        print("   ou defina no código: os.environ['OPENAI_API_KEY'] = 'sua-key'\n")

    # 1. Carregar e dividir documentos
    print("\n1️⃣  Carregando documentos...")
    chunks = load_and_split_documents("sample_document.txt")

    # Se não existe arquivo, criar exemplo
    if not chunks:
        print("📝 Criando documento de exemplo...")
        with open("sample_document.txt", "w") as f:
            f.write("""
RAG (Retrieval-Augmented Generation) é uma técnica que combina
memória paramétrica (modelos pré-treinados) com memória não-paramétrica
(índices vetoriais) para geração de linguagem mais factual.

Vantagens do RAG:
1. Reduz hallucinations
2. Permite knowledge up-to-date
3. Fornece citations/explicabilidade
4. Custo-efetivo vs fine-tuning

Lewis et al. (2020) introduziu RAG para tarefas intensivas em conhecimento.
O paper demonstra que RAG supera modelos paramétricos-only em QA aberto.
""")
        chunks = load_and_split_documents("sample_document.txt")

    # 2. Criar vector store
    print("\n2️⃣  Criando vector store...")
    vector_store = create_vector_store(chunks)

    # 3. Criar QA chain
    print("\n3️⃣  Configurando QA chain...")
    qa_chain = create_qa_chain(vector_store)

    # 4. Fazer perguntas
    print("\n" + "=" * 60)
    print("Fazendo perguntas...")
    print("=" * 60)

    questions = [
        "O que é RAG?",
        "Quais as vantagens do RAG?",
        "Quem introduziu a técnica RAG?"
    ]

    for question in questions:
        result = query_rag(qa_chain, question)
        display_results(result)
        print("-" * 60)

    print("\n✨ Exemplo concluído!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 02: RAG vs Alternatives Comparison
==========================================

Este exemplo compara RAG com alternativas: Fine-tuning, Pure Generative,
e Vector Search Only.

Uso:
    python example-02-rag-vs-alternatives.py
"""

import os
from typing import List, Dict
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


class RAGApproach:
    """Implementação RAG"""
    def __init__(self, documents: List[str]):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.vectorstore = Chroma.from_texts(documents, self.embeddings)
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(k=3)
        )

    def query(self, question: str) -> str:
        result = self.qa.run(question)
        return result


class PureGenerativeApproach:
    """Pure Generative - apenas LLM, sem retrieval"""
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)

    def query(self, question: str) -> str:
        prompt = f"""
Pergunta: {question}

Resposta (baseada no conhecimento geral do modelo):
"""
        return self.llm(prompt)


class VectorSearchOnlyApproach:
    """Vector Search Only - sem LLM, só busca"""
    def __init__(self, documents: List[str]):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_texts(documents, self.embeddings)

    def query(self, question: str) -> str:
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Documentos relacionados encontrados:\n\n{context}"


class FineTuningApproach:
    """Fine-tuning - LLM treinado com dados específicos"""
    def __init__(self):
        self.llm = OpenAI(
            model_name="gpt-3.5-turbo-fine-tuned",
            temperature=0
        )

    def query(self, question: str) -> str:
        # Simular fine-tuned model
        prompt = f"Q: {question}\nA:"
        return self.llm(prompt)


def compare_approaches(question: str, documents: List[str]):
    """Compara todas as abordagens"""
    print(f"\n{'='*60}")
    print(f"Pergunta: {question}")
    print(f"{'='*60}\n")

    # 1. RAG
    print("1️⃣  RAG (Retrieval-Augmented Generation)")
    print("-" * 60)
    try:
        rag = RAGApproach(documents)
        result = rag.query(question)
        print(f"Resposta: {result}")
        print(f"✅ Vantagens: Context-aware, factual, citations")
        print(f"⚠️  Desvantagens: Complexidade, latência extra")
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("   (Requer OpenAI API key)")

    # 2. Pure Generative
    print("\n2️⃣  Pure Generative (Só LLM)")
    print("-" * 60)
    try:
        pure = PureGenerativeApproach()
        result = pure.query(question)
        print(f"Resposta: {result}")
        print(f"✅ Vantagens: Simples, rápido, barato")
        print(f"⚠️  Desvantagens: Hallucinations, knowledge limitado")
    except Exception as e:
        print(f"❌ Erro: {e}")

    # 3. Vector Search
    print("\n3️⃣  Vector Search Only (Só busca)")
    print("-" * 60)
    try:
        vector = VectorSearchOnlyApproach(documents)
        result = vector.query(question)
        print(f"Resposta: {result}")
        print(f"✅ Vantagens: Rápido, não gera, factual")
        print(f"⚠️  Desvantagens: Sem geração, contexto limitado")
    except Exception as e:
        print(f"❌ Erro: {e}")

    # 4. Fine-tuning
    print("\n4️⃣  Fine-tuning (LLM treinado)")
    print("-" * 60)
    try:
        fine = FineTuningApproach()
        result = fine.query(question)
        print(f"Resposta: {result}")
        print(f"✅ Vantagens: Alto desempenho, especializado")
        print(f"⚠️  Desvantagens: Caro, estático, complexo")
    except Exception as e:
        print(f"❌ Erro: {e}")


def comparison_matrix():
    """Exibe matriz de comparação"""
    print(f"\n{'='*80}")
    print("MATRIZ DE COMPARAÇÃO")
    print(f"{'='*80}")

    matrix = """
┌──────────────┬──────────┬──────────────┬──────────┬──────────────┐
│ Critério     │ RAG      │ Fine-tuning  │ Pure Gen │ Vector Search│
├──────────────┼──────────┼──────────────┼──────────┼──────────────┤
│ Conhecimento │ External │ Paramétrico  │ Param.   │ External     │
│              │ + Param. │              │          │              │
├──────────────┼──────────┼──────────────┼──────────┼──────────────┤
│ Atualização  │ Fácil    │ Caro         │ Imposs.  │ Fácil        │
│              │ (update) │ (re-train)   │          │ (update)     │
├──────────────┼──────────┼──────────────┼──────────┼──────────────┤
│ Custo        │ Baixo-   │ Alto         │ Baixo    │ Baixo        │
│              │ Médio    │              │          │              │
├──────────────┼──────────┼──────────────┼──────────┼──────────────┤
│ Performance  │ Alta     │ Muito Alta   │ Média    │ Alta         │
├──────────────┼──────────┼──────────────┼──────────┼──────────────┤
│ Hallucination│ Menor    │ Pode         │ Freq.    │ Não          │
│              │          │ ocorrer      │          │              │
├──────────────┼──────────┼──────────────┼──────────┼──────────────┤
│ Citations    │ Sim      │ Não          │ Não      │ Sim          │
└──────────────┴──────────┴──────────────┴──────────┴──────────────┘
"""
    print(matrix)


def when_to_use_which():
    """Explica quando usar cada abordagem"""
    print(f"\n{'='*80}")
    print("QUANDO USAR CADA ABORDAGEM")
    print(f"{'='*80}\n")

    recommendations = {
        "RAG": {
            "use_when": [
                "✅ Knowledge up-to-date é crítico",
                "✅ Dados mudam frequentemente",
                "✅ Precisa de citations/explicabilidade",
                "✅ Volume de dados é grande",
                "✅ Custo de re-treino é alto"
            ],
            "dont_use_when": [
                "❌ Domínio bem restrito e estático",
                "❌ Performance máxima é prioridade"
            ]
        },
        "Fine-tuning": {
            "use_when": [
                "✅ Domínio bem definido e estável",
                "✅ Tem budget e time para treinar",
                "✅ Performance máxima é crítica",
                "✅ Não precisa de citations"
            ],
            "dont_use_when": [
                "❌ Dados mudam frequentemente",
                "❌ Budget/tempo limitado"
            ]
        },
        "Pure Generative": {
            "use_when": [
                "✅ Tarefas criativas",
                "✅ Não precisa de factualidade",
                "✅ Knowledge geral é suficiente"
            ],
            "dont_use_when": [
                "❌ Precisa de informações factuais",
                "❌ Domain-specific knowledge"
            ]
        },
        "Vector Search": {
            "use_when": [
                "✅ Busca semântica",
                "✅ Não precisa de geração",
                "✅ Apenas recuperar documentos"
            ],
            "dont_use_when": [
                "❌ Precisa de síntese/gerar texto",
                "❌ Respostas complexas necessárias"
            ]
        }
    }

    for approach, details in recommendations.items():
        print(f"\n🎯 {approach.upper()}:")
        for item in details["use_when"]:
            print(f"   {item}")
        print("   Quando NÃO usar:")
        for item in details["dont_use_when"]:
            print(f"   {item}")


def main():
    """Função principal"""
    print("=" * 80)
    print("RAG vs ALTERNATIVES - Comparação de Abordagens")
    print("=" * 80)

    # Documentos de exemplo
    documents = [
        "RAG combina memória paramétrica e não-paramétrica",
        "RAG reduz hallucinations em sistemas de QA",
        "Lewis et al. (2020) introduziu RAG para NLP",
        "RAG permite knowledge up-to-date sem re-treinar"
    ]

    # Perguntas de teste
    questions = [
        "O que é RAG?",
        "Como RAG reduz hallucinations?",
        "Quem introduziu RAG?"
    ]

    # Exibir matriz de comparação
    comparison_matrix()

    # Exibir recomendações
    when_to_use_which()

    # Comparar abordagens
    for question in questions:
        compare_approaches(question, documents)

    print("\n" + "=" * 80)
    print("CONCLUSÃO")
    print("=" * 80)
    print("""
RAG é ideal quando você precisa de:
- Knowledge factual e up-to-date
- Explicabilidade (citations)
- Custo-efetividade
- Flexibilidade

Escolha alternativas se:
- Domínio estático → Fine-tuning
- Performance máxima → Fine-tuning
- Tarefas criativas → Pure Generative
- Só busca → Vector Search
""")
    print("=" * 80)


if __name__ == "__main__":
    main()

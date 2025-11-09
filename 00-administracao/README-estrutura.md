# Estrutura de Diretórios - Detalhada

### Propósito
Esta estrutura foi projetada para organizar o projeto RAG Knowledge Base de forma lógica e escalável.

---

## 📁 00-administracao/

**Conteúdo**: Documentos administrativos e de planejamento estratégico

**Arquivos**:
- `README.md` - Visão geral do projeto
- `GLOSSARIO.md` - Termos técnicos
- `Plano Detalhado de Base de Conhecimento.md` - Plano original (completo)
- `README-estrutura.md` - Este arquivo

**Uso**:
- Stakeholders: Ver README.md
- Novos contribuidores: Ler este arquivo
- Referência: GLOSSARIO.md

---

## 📁 01-planos/

**Conteúdo**: Metodologia e cronograma de execução

**Arquivos**:
- `Plano de Pesquisa - Base Conhecimento RAG.md` - Metodologia de pesquisa
- `Cronograma-execucao.md` - Timeline detalhado

**Uso**:
- Pesquisadores: Seguir metodologia
- PMs: Acompanhar cronograma
- Equipe: Verificar entregáveis

---

## 📁 02-relatorios-pesquisa/

**Conteúdo**: Todos os relatórios de pesquisa coletados

### Estrutura por Fases

**Fase 1 (CONCLUÍDA)**:
- `00-Fundamentals/Relatorio-Pesquisa-00-Fundamentals.md`
- `01-Document-Processing/Relatorio-Pesquisa-01-Document-Processing.md`
- `02-Chunking-Strategies/Relatorio-Pesquisa-02-Chunking-Strategies.md`
- `Resumo-Executivo-Fase1.md`

**Fase 2 (EM ANDAMENTO)**:
- `03-Embedding-Models/` (próximo)
- `04-Vector-Databases/`

**Fases 3-5 (PENDENTES)**:
- `05-Retrieval-Optimization/`
- `06-Evaluation-Benchmarks/`
- `07-Performance-Optimization/`
- `08-Advanced-Patterns/`
- `09-Architecture-Patterns/`
- `10-Frameworks-Tools/`
- `11-Production-Deployment/`
- `12-Troubleshooting/`
- `13-Use-Cases/`
- `14-Case-Studies/`
- `15-Future-Trends/`
- `16-Resources/`

### Padrão de Relatórios

Cada relatório contém:
1. Resumo Executivo
2. Fontes Primárias
3. Insights Técnicos
4. Comparações
5. Code Examples
6. Best Practices
7. Common Pitfalls
8. Próximos Passos

---

## 📁 03-code-examples/

**Conteúdo**: Exemplos de código executáveis

### Por Fase
- `Fase-1/Code-Examples-Fase1.md` - 5 exemplos (minimal RAG, document processing, chunking, etc.)
- `Fase-2/` - Embeddings, vector DBs
- `Fase-3/` - Retrieval, evaluation
- `Fase-4/` - Advanced patterns
- `Fase-5/` - Production deployment

### Padrão de Code Example
```markdown
## Example N: Título

### Prerequisites
```bash
pip install ...
```

### Descrição
- O que faz
- Quando usar
- Windows-specific notes

### Código Completo
```python
# Código executável
```

### Como Executar
```powershell
# Comandos PowerShell
```

### Próximos Passos
- Links para guias
- Variações
```

---

## 📁 04-guias-prontos/

**Conteúdo**: Guias finais para usuários

### Estrutura Final
```
04-guias-prontos/
├── Fundamentals/
├── Document-Processing/
├── Chunking-Strategies/
├── Embedding-Models/
├── Vector-Databases/
├── Retrieval-Optimization/
├── Evaluation/
├── Performance-Optimization/
├── Advanced-Patterns/
├── Architecture-Patterns/
├── Frameworks-Tools/
├── Production-Deployment/
├── Troubleshooting/
├── Use-Cases/
└── Resources/
```

### Padrão de Guia
Cada guia contém:
- Introdução conceitual
- Tutorial step-by-step
- Comparações técnicas
- Code examples
- Benchmarks
- Troubleshooting
- Próximos passos

---

## 📁 05-assets/

**Conteúdo**: Recursos complementares

### Subdiretórios
- `diagrams/` - Arquiteturas, fluxos
- `benchmarks/` - Resultados de performance
- `resources/` - Links, papers, tools

---

## 🔄 FLUXO DE TRABALHO

### 1. Pesquisa (Fase Atual)
```
01-planos/ → 02-relatorios-pesquisa/Fase-X/
        ↓
Pesquisar → Coletar → Organizar → Validar
        ↓
        ↓
   Code Examples
        ↓
   Resumo da Fase
```

### 2. Escrita de Guias
```
02-relatorios-pesquisa/ → 04-guias-prontos/
        ↓
Extrair → Simplificar → Estruturar → Revisar
        ↓
   Guias Prontos
```

### 3. Publicação
```
04-guias-prontos/ → Deploy
        ↓
Build → Test → Release
```

---

## 📋 NOMENCLATURA

### Diretórios
- **Numérica**: 00, 01, 02... (mantém ordem)
- **Hierárquica**: Fase-X/Seção-YY/
- **Descritiva**: Usar hifens (advanced-patterns)

### Arquivos
- **Relatórios**: `Relatorio-Pesquisa-YY-Name.md`
- **Code Examples**: `Code-Examples-Fase-X.md`
- **Resumo**: `Resumo-Executivo-FaseX.md`
- **Guia**: `guia-name.md`

### Commits
- `feat: adicionar seções 03-04`
- `fix: corrigir code example`
- `docs: atualizar estrutura`

---

## 🎯 OBJETIVOS DA ESTRUTURA

### 1. Escalabilidade
- Fácil adicionar novas fases
- Padrão consistente
- Espaços reservados (05-assets/)

### 2. Navegação
- Intuitiva
- Lógica
- Indexável

### 3. Colaboração
- Clara separação de responsabilidades
- Padrões documentados
- Fluxo de trabalho definido

### 4. Reutilização
- Conteúdo modular
- Referencias cruzadas
- Assets compartilhados

---

## 🔍 COMO ENCONTRAR COISAS

### Para um Conceito
1. Verificar `00-administracao/GLOSSARIO.md`
2. Procurar em `02-relatorios-pesquisa/`
3. Code examples em `03-code-examples/`

### Para uma Seção Específica
1. Identificar seção (00-16)
2. Ir para `02-relatorios-pesquisa/Fase-X/YY-Name/`
3. Ler `Relatorio-Pesquisa-YY-Name.md`

### Para um Exemplo
1. Identificar fase
2. Ir para `03-code-examples/Fase-X/`
3. Procurar no arquivo

### Para um Guia Final
1. Ir para `04-guias-prontos/`
2. Selecionar categoria
3. Ler guia correspondente

---

## 📌 BOAS PRÁTICAS

### ✅ Fazer
- Usar nomenclatura padrão
- Adicionar a índice
- Documentar mudanças
- Manter consistência

### ❌ Evitar
- Arquivos na raiz (exceto README)
- Nomes ambiguos
- Duplicação de conteúdo
- Quebrar padrões

---

**Última atualização**: 09/11/2025
**Revisão**: A cada nova fase

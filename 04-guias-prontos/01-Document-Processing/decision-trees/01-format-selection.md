# Decision Tree: Selecionar Document Format

## Fluxo de Decisão

```
                          START
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Tipo de documento?          │
              └─────────────┬───────────────┘
                            │
           ┌────────────────┼────────────────┐
          PDF              Texto            Web
           │                │                │
           ▼                ▼                ▼
    ┌────────────┐  ┌──────────────┐  ┌──────────┐
    │ Scanned?   │  │ Formato?     │  │ Static?  │
    └──────┬─────┘  └──────┬───────┘  └────┬─────┘
           │                │                │
    ┌──────┴──────┐  ┌──────┴───────┐  ┌────┴─────┐
   YES          NO  DOCX    CSV     │  YES      NO
    │             │   │      │      │   │         │
    ▼             ▼   ▼      ▼      ▼   ▼         ▼
OCR Required   PyPDF  Docx   CSV   TXT WebBase   Selenium
(Pytesseract)  Loader Loader Loader Loader Loader (JS)

```

## Matriz de Seleção

| Formato | Loader Principal | Quando Usar | Complexidade |
|---------|------------------|-------------|--------------|
| **PDF** | PyPDFLoader | Documentos formais, manuais | ⭐⭐ |
| **DOCX** | Docx2txtLoader | Documentos editáveis, relatórios | ⭐ |
| **TXT** | TextLoader | Textos simples, logs | ⭐ |
| **HTML** | WebBaseLoader | Páginas web, blogs | ⭐⭐ |
| **CSV** | CSVLoader | Dados tabulares | ⭐ |
| **Markdown** | UnstructuredMarkdownLoader | Documentação, READMEs | ⭐ |
| **PPTX** | PowerPointLoader | Apresentações | ⭐⭐ |
| **XLSX** | ExcelLoader | Planilhas | ⭐⭐ |

## Critérios de Decisão

### 1. Document Type
- **PDF estático** → PyPDFLoader
- **PDF escaneado** → OCR (pytesseract + PyMuPDF)
- **Texto bruto** → TextLoader
- **Estruturado** → Loader específico (CSV, DOCX, etc.)
- **Web** → WebBaseLoader (static) ou SeleniumLoader (JS)

### 2. Content Preservation
- **Tabelas importantes** → CSVLoader, Docx2txtLoader
- **Imagens necessárias** → Unstructured + additional processing
- **Formatação essencial** → Formato nativo (DOCX, PPTX)
- **Apenas texto** → TextLoader

### 3. Performance
- **Arquivos grandes** → Streaming, lazy loading
- **Muitos arquivos** → Batch processing
- **Tempo crítico** → Loaders mais rápidos (Text > DOCX > PDF)

## Exemplos de Decisão

### Exemplo 1: Manual Técnico
- **Formato:** PDF com texto selecionável
- **Decisão:** PyPDFLoader
- **Por quê:** Manuais são tipicamente PDF, PyPDF é otimizado

### Exemplo 2: Relatório Excel
- **Formato:** XLSX com gráficos
- **Decisão:** ExcelLoader (apenas dados) ou python-openpyxl
- **Por quê:** Dados em formato tabular, ExcelLoader para extrair

### Exemplo 3: Site E-commerce
- **Formato:** HTML dinâmico (JavaScript)
- **Decisão:** SeleniumLoader ou WebBaseLoader com soup
- **Por quê:** Conteúdo gerado por JS, precisa renderização

### Exemplo 4: Documentação API
- **Formato:** Markdown (.md)
- **Decisão:** UnstructuredMarkdownLoader
- **Por quê:** Preserva headers, estrutura Markdown

## Comparação Loaders

### Speed (Do mais rápido ao mais lento)
1. TextLoader - instantâneo
2. Docx2txtLoader - rápido
3. CSVLoader - rápido
4. UnstructuredMarkdownLoader - médio
5. WebBaseLoader - médio
6. PyPDFLoader - mais lento
7. PowerPointLoader - lento
8. ExcelLoader - lento

### Quality (Text preservation)
1. TextLoader - 100%
2. Docx2txtLoader - 100%
3. MarkdownLoader - 95%
4. CSVLoader - 100%
5. PyPDFLoader - 90-95%
6. PPTXLoader - 90%
7. WebBaseLoader - 80-90%
8. ExcelLoader - 100%

## Guidelines

### Use TextLoader quando:
- ✅ Arquivo é texto puro
- ✅ Performance é crítica
- ✅ Nenhuma estrutura importante

### Use Docx2txtLoader quando:
- ✅ Documento é DOCX
- ✅ Precisa extrair texto
- ✅ Tabelas são menos importantes

### Use PyPDFLoader quando:
- ✅ Documento é PDF
- ✅ Texto é selecionável (não escaneado)
- ✅ Documentos grandes

### Use WebBaseLoader quando:
- ✅ Página web estática
- ✅ BeautifulSoup disponível
- ✅ Conteúdo público

## Próximos Passos

- **Escolheu o format?** → Ver [Code Examples](../code-examples/)
- **Problemas com loading?** → [Troubleshooting](../troubleshooting/common-issues.md)
- **Otimizar processamento?** → [Tutorial Avançado](../tutorials/)

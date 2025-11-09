# Troubleshooting - Embedding Models

## Problemas Comuns e Soluções

### 1. API Key Issues

**Sintomas:**
- AuthenticationError
- Invalid API key
- RateLimitError

**Soluções:**

```python
# Configurar API key
import os
os.environ["OPENAI_API_KEY"] = "sua-key-aqui"

# Verificar se está configurada
if not os.getenv("OPENAI_API_KEY"):
    print("Configure OPENAI_API_KEY")
```

**Rate limiting:**
```python
import time
from openai.error import RateLimitError

def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)
```

### 2. Model Not Found

**Sintomas:**
- ModelNotFoundError
- Invalid model name
- Connection timeout

**Soluções:**

```python
# Verificar nome do modelo
from langchain.embeddings import HuggingFaceEmbeddings

# Modelos corretos:
embeddings = HuggingFaceEmbeddings('BAAI/bge-large-en-v1.5')  # ✅
embeddings = HuggingFaceEmbeddings('bge-large')  # ❌

# Testar conexão
try:
    model = HuggingFaceEmbeddings('model_name')
    model.embed_query("test")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
```

### 3. Slow Embedding Generation

**Sintomas:**
- Long processing time
- High latency
- Batch processing slow

**Soluções:**

#### A. Usar modelo mais rápido
```python
# Trocar para modelo mais leve
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Mais rápido
)
```

#### B. Batch processing
```python
# Processar em batches
def batch_embed(texts, batch_size=100):
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
    return all_vectors

# Uso
texts = ["text1", "text2", "text3", ...]
vectors = batch_embed(texts, batch_size=50)
```

#### C. Async embedding
```python
import asyncio

async def async_embed(texts):
    # Usar aiohttp ou similar para requests assíncronas
    # Ou usar библиотеки que suportam async
    pass

# Para HuggingFace, não há async nativo
# Para OpenAI, usar OpenAI async client
```

### 4. Out of Memory

**Sintomas:**
- MemoryError
- OOMKilled
- High RAM usage

**Soluções:**

#### A. Processar em batches menores
```python
# Reduzir batch_size
for i in range(0, len(texts), 10):  # Batch de 10
    batch = texts[i:i+10]
    vectors = embeddings.embed_documents(batch)
    # Processar batch
```

#### B. Usar modelo menor
```python
# Modelo menor = menos memória
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
)
```

#### C. Gerar embeddings sob demanda
```python
# Não pré-computar todos
def embed_on_demand(text):
    return embeddings.embed_query(text)
```

### 5. Inconsistent Results

**Sintomas:**
- Same text different vectors
- Non-deterministic
- Quality variations

**Soluções:**

```python
# Verificar seed (se aplicável)
# Alguns modelos permitem seed
embeddings = HuggingFaceEmbeddings(
    model_name="model_name",
    model_kwargs={"device": "cpu"}  # Consistente
)

# Verificar normalization
# Embeddings devem ser normalizados para cosine similarity
```

### 6. High Costs

**Sintomas:**
- Expensive API calls
- High token usage
- Budget exceeded

**Soluções:**

#### A. Cache embeddings
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def cached_embed(text):
    return embeddings.embed_query(text)

# Usage
vector = cached_embed("text to embed")
```

#### B. Usar modelos open source
```python
# Trocar para gratuito
embeddings = HuggingFaceEmbeddings('BAAI/bge-large-en-v1.5')
# vs
embeddings = OpenAIEmbeddings()  # Pago
```

#### C. Reduzir número de embeddings
```python
# Eliminar duplicates
def deduplicate_texts(texts):
    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    return unique_texts
```

### 7. Multilingual Issues

**Sintomas:**
- Poor quality for non-English
- No support for some languages
- Mixed language problems

**Soluções:**

```python
# Usar modelo multilingual
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"  # Good multilingual
)

# Ou
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh"  # Chinese
)
```

### 8. Model Version Issues

**Sintomas:**
- Different results with same model
- Deprecation warnings
- Version conflicts

**Soluções:**

```python
# Fix version
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    # ou especificar versão exata
    model_kwargs={"torch_dtype": "float32"}
)

# Documentar versão usada
print(f"Model: {embeddings.model_name}")
```

### 9. GPU Memory Issues

**Sintomas:**
- CUDA out of memory
- GPU not used
- Slow processing

**Soluções:**

```python
# Forçar CPU
embeddings = HuggingFaceEmbeddings(
    model_name="model_name",
    model_kwargs={"device": "cpu"}  # Usar CPU
)

# Ou configurar GPU
embeddings = HuggingFaceEmbeddings(
    model_name="model_name",
    model_kwargs={
        "device": "cuda",
        "torch_dtype": "float16"  # Half precision
    }
)
```

### 10. Embedding Quality Issues

**Sintomas:**
- Poor similarity results
- Unrelated content returned
- Low precision

**Soluções:**

#### A. Mudar modelo
```python
# Usar modelo de melhor qualidade
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"  # Melhor qualidade
)
```

#### B. Verificar preprocessing
```python
# Limpar texto antes de embed
import re

def clean_text(text):
    # Remover extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remover special chars (se necessário)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

cleaned_text = clean_text(raw_text)
vector = embeddings.embed_query(cleaned_text)
```

## Debug Checklist

- [ ] API key configurada
- [ ] Nome do modelo correto
- [ ] Conexão funcionando
- [ ] Memória suficiente
- [ ] Batch size adequado
- [ ] Rate limits não atingidos
- [ ] Preprocessing adequado
- [ ] Model version fixada

## Performance Optimization

```python
# Profile embedding time
import time
import cProfile

def profile_embedding():
    profiler = cProfile.Profile()
    profiler.enable()

    vectors = embeddings.embed_documents(texts)

    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

## Validation Script

```python
def validate_embedding(embeddings):
    """Validar qualidade do embedding"""
    # Test consistency
    v1 = embeddings.embed_query("test")
    v2 = embeddings.embed_query("test")
    assert v1 == v2, "Non-deterministic embedding"

    # Test similarity
    v1 = embeddings.embed_query("dog")
    v2 = embeddings.embed_query("cat")
    v3 = embeddings.embed_query("car")

    from numpy import dot
    from numpy.linalg import norm

    sim_dog_cat = dot(v1, v2) / (norm(v1) * norm(v2))
    sim_dog_car = dot(v1, v3) / (norm(v1) * norm(v3))

    assert sim_dog_cat > sim_dog_car, "Similarity test failed"

    print("✅ All validation tests passed")
```

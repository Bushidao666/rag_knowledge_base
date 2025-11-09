# Troubleshooting - Frameworks & Tools

## Problemas Comuns

### 1. Import Errors

**Problema:** Framework não encontrado

**Soluções:**
```bash
# LangChain
pip install langchain

# LlamaIndex
pip install llama-index

# Haystack
pip install haystack
```

### 2. Version Conflicts

**Problema:** Incompatibilidade de versões

**Soluções:**
```python
# LangChain
from langchain import __version__
print(f"Version: {__version__}")

# Use requirements.txt
# langchain==0.0.350
```

### 3. Performance Issues

**Problema:** Framework muito lento

**Soluções:**
```python
# LangChain - use async
from langchain.callbacks import AsyncCallbackManager

# Haystack - use pipeline
from haystack import Pipeline
pipeline = Pipeline()

# LlamaIndex - use query engine optimization
query_engine = index.as_query_engine(
    response_mode="compact"
)
```

## Debug Checklist

- [ ] Check installation
- [ ] Verify version compatibility
- [ ] Test with simple example
- [ ] Check documentation
- [ ] Join community Discord

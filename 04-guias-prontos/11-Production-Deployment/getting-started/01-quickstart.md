# Quick Start: Production Deployment

**Tempo estimado:** 15-30 minutos
**Nível:** Avançado
**Pré-requisitos:** RAG testado e validado

## Objetivo
Deploy RAG systems em produção

## Deployment Options

### 1. Docker
Containerização:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

```bash
# Build
docker build -t rag-app .

# Run
docker run -p 8000:8000 -e OPENAI_API_KEY=... rag-app
```

### 2. Kubernetes
Orquestração:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-app
  template:
    metadata:
      labels:
        app: rag-app
    spec:
      containers:
      - name: rag
        image: rag-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

### 3. Serverless (AWS Lambda)
Função sem servidor:
```python
import json
import boto3

def lambda_handler(event, context):
    # Parse request
    body = json.loads(event['body'])
    question = body['question']

    # Query RAG
    answer = rag.query(question)

    # Return response
    return {
        'statusCode': 200,
        'body': json.dumps({'answer': answer})
    }
```

## Architecture Production

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────────────────┐
│  Load Balancer          │
│  (NGINX/AWS ALB)       │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│  API Gateway            │
│  (Kong/AWS API GW)     │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│  RAG Services (3x)      │
│  [Container/K8s Pod]   │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│  Vector Database        │
│  (Pinecone/Weaviate)   │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│  Monitoring             │
│  (Prometheus/Grafana)  │
└─────────────────────────┘
```

## Environment Configuration

### Variables
```bash
# .env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
ENVIRONMENT=production
```

### Secrets Management
```python
# Kubernetes secrets
kubectl create secret generic openai-key \
  --from-literal=api-key=sk-...

# AWS Secrets Manager
import boto3
secrets = boto3.client('secretsmanager')
api_key = secrets.get_secret_value('openai-key')
```

## API Design

### FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    max_results: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = rag.query(request.question)
    return QueryResponse(
        answer=result["answer"],
        sources=[doc.page_content for doc in result["sources"]]
    )
```

## Monitoring & Observability

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('rag_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('rag_request_duration_seconds', 'Request latency')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.time() - start)
    REQUEST_COUNT.inc()
    return response
```

### Health Checks
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "vector_db": check_vector_db(),
            "llm": check_llm()
        }
    }
```

## Security

### API Key Protection
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_api_key(token: str = Depends(security)):
    if not check_api_key(token.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, query: QueryRequest):
    return rag.query(query.question)
```

## CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy RAG

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker
        run: docker build -t rag-app .
      - name: Run Tests
        run: pytest
      - name: Deploy to EKS
        run: |
          kubectl set image deployment/rag-app \
            rag-app=rag-app:latest
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Error Handling

```python
from fastapi import HTTPException

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = rag.query(request.question)
        return result
    except VectorDBError as e:
        raise HTTPException(status_code=503, detail="Vector DB unavailable")
    except LLMError as e:
        raise HTTPException(status_code=502, detail="LLM service error")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")
```

## Checklist

- [ ] Docker containerization
- [ ] Environment variables
- [ ] Secrets management
- [ ] API design
- [ ] Monitoring
- [ ] Health checks
- [ ] Security (auth, rate limiting)
- [ ] CI/CD pipeline
- [ ] Scaling strategy
- [ ] Error handling
- [ ] Logging
- [ ] Backup & recovery

## Próximos Passos

- 💻 **Code Examples:** [Deployment Examples](../code-examples/)
- 🔧 **Monitoring:** [Observability Guide](../troubleshooting/common-issues.md)

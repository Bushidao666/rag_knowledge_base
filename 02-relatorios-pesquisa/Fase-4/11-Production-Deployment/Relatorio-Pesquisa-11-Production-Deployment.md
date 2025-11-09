# Relatório de Pesquisa: Seção 11 - Production Deployment

### Data: 09/11/2025
### Status: Fase 4 - Advanced Topics

---

## 1. RESUMO EXECUTIVO

Production Deployment é critical para sistemas RAG escaláveis, seguros e confiáveis. A escolha de infraestrutura, monitoring e security impacta sucesso em production.

**Insights Chave:**
- **Docker/Kubernetes**: Container orchestration para scalability
- **Cloud Providers**: AWS, GCP, Azure deployment options
- **Monitoring**: Prometheus, Grafana, LangSmith para observability
- **Security**: Authentication, authorization, encryption
- **CI/CD**: Automated deployment pipelines
- **High Availability**: Load balancing, auto-scaling, disaster recovery

---

## 2. DOCKER CONTAINERS

### 2.1 Overview

**Docker** packages applications e dependencies em containers para consistent deployment.

**Vantagens:**
- ✅ Consistent environment
- ✅ Portability
- ✅ Isolation
- ✅ Scalability
- ✅ Easy deployment

### 2.2 RAG Application Dockerfile

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.3 Multi-Stage Build (Production)

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY . .

USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.4 Docker Compose (Development)

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=rag
      - POSTGRES_USER=rag
      - POSTGRES_PASSWORD=ragpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Optional: Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  redis_data:
  postgres_data:
  grafana_data:
```

### 2.5 Docker Compose (Production)

```yaml
version: '3.8'

services:
  app:
    image: my-rag-app:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rag
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - app
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

---

## 3. KUBERNETES

### 3.1 Overview

**Kubernetes** orchestrates containers at scale para production deployment.

**Componentes:**
- **Pods**: Smallest deployable unit
- **Services**: Stable network endpoints
- **Deployments**: Replica management
- **ConfigMaps**: Configuration
- **Secrets**: Sensitive data
- **Volumes**: Storage

### 3.2 RAG Application Manifests

**Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app
  labels:
    app: rag-app
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
      - name: rag-app
        image: my-rag-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: pinecone-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-app-service
spec:
  selector:
    app: rag-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**ConfigMap:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
data:
  app.config: |
    DEBUG=false
    LOG_LEVEL=info
    CORS_ORIGINS=["https://example.com"]
    MAX_CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
```

**Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
type: Opaque
stringData:
  openai-api-key: "your-openai-key"
  pinecone-api-key: "your-pinecone-key"
  database-url: "postgresql://user:pass@postgres:5432/rag"
```

### 3.3 Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3.4 Ingress (Load Balancer)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: rag-app-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-app-service
            port:
              number: 80
```

### 3.5 Production Checklist

- [ ] Resource limits set
- [ ] Liveness/readiness probes configured
- [ ] HPA enabled
- [ ] Ingress configured
- [ ] TLS certificates
- [ ] Secrets management
- [ ] ConfigMaps for config
- [ ] Rolling updates
- [ ] Service mesh (optional)
- [ ] Monitoring enabled

---

## 4. CLOUD DEPLOYMENT

### 4.1 AWS

**ECS (Elastic Container Service):**

```json
{
  "family": "rag-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "rag-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/rag-app:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**EKS (Elastic Kubernetes Service):**
```bash
# Create cluster
eksctl create cluster \
  --name rag-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed

# Deploy application
kubectl apply -f k8s/
```

**Lambda (Serverless):**
```python
import json
from lambda_function import rag_query

def lambda_handler(event, context):
    query = event['query']
    result = rag_query(query)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### 4.2 Google Cloud

**Cloud Run:**
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rag-app
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: gcr.io/your-project/rag-app:latest
        ports:
        - name: http1
          containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        resources:
          limits:
            cpu: 1000m
            memory: 2Gi
```

**GKE (Kubernetes Engine):**
```bash
# Create cluster
gcloud container clusters create rag-cluster \
  --zone us-central1 \
  --num-nodes 3 \
  --machine-type n1-standard-4

# Get credentials
gcloud container clusters get-credentials rag-cluster --zone us-central1

# Deploy
kubectl apply -f k8s/
```

### 4.3 Azure

**Container Instances:**
```yaml
apiVersion: 2021-03-01
location: eastus
name: rag-app
properties:
  containers:
  - name: rag-app
    properties:
      image: your-registry.azurecr.io/rag-app:latest
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: OPENAI_API_KEY
        secureValue: "your-key"
      resources:
        requests:
          cpu: 1.0
          memory: 1.5Gi
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
```

**AKS (Kubernetes Service):**
```bash
# Create cluster
az aks create \
  --resource-group rg-rag \
  --name rag-cluster \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group rg-rag \
  --name rag-cluster

# Deploy
kubectl apply -f k8s/
```

---

## 5. MONITORING & OBSERVABILITY

### 5.1 Prometheus

**Configuration:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-app'
    static_configs:
      - targets: ['rag-app:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'vector-db'
    static_configs:
      - targets: ['chroma:8000']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

**Metrics to Track:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['method', 'endpoint']
)

REQUEST_LATENCY = Histogram(
    'rag_request_duration_seconds',
    'RAG request latency'
)

ACTIVE_QUERIES = Gauge(
    'rag_active_queries',
    'Number of active queries'
)

# Use in code
from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/query")
def query(question: str):
    start = time.time()
    ACTIVE_QUERIES.inc()

    try:
        result = rag_query(question)
        REQUEST_COUNT.labels(method='GET', endpoint='/query').inc()
        return result
    finally:
        REQUEST_LATENCY.observe(time.time() - start)
        ACTIVE_QUERIES.dec()
```

### 5.2 Grafana

**Dashboard Configuration:**
```json
{
  "dashboard": {
    "title": "RAG Application",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Active Queries",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rag_active_queries"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_requests_total{status='error'}[5m]) / rate(rag_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 5.3 LangSmith

**Setup:**
```python
import os
from langsmith import Client

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "rag-production"

from langsmith import trace

# Trace RAG chain
with trace("rag-query"):
    result = rag_chain.invoke("What is AI?")
```

**Feedback Collection:**
```python
from langsmith.feedback import feedback

# Collect user feedback
feedback(
    key="rag-quality",
    value=4.5,  # 1-5 rating
    comment="Good answer with relevant sources",
    metadata={"query": "What is AI?", "user_id": "user123"}
)
```

### 5.4 Application Monitoring

```python
import logging
from structlog import configure, get_logger

# Configure structured logging
configure(
    processors=[
        logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.processors.JSONRenderer()
                }
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "json"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": True
                }
            }
        })
    ]
)

logger = get_logger()

# Use in application
@app.get("/query")
def query(question: str):
    logger.info(
        "Query received",
        question=question,
        user_id=current_user.id
    )

    try:
        result = rag_query(question)
        logger.info(
            "Query successful",
            result_length=len(result),
            latency_ms=latency
        )
        return result
    except Exception as e:
        logger.error(
            "Query failed",
            error=str(e),
            exc_info=True
        )
        raise
```

---

## 6. SECURITY

### 6.1 Authentication & Authorization

**JWT Tokens:**
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def verify_token(token: str = Depends(security)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user_id
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.get("/query")
def query(question: str, user_id: str = Depends(verify_token)):
    return rag_query(question)
```

**API Key Management:**
```python
from fastapi import FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")

API_KEYS = {
    "user1": {"rate_limit": 100, "role": "user"},
    "user2": {"rate_limit": 1000, "role": "admin"}
}

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return API_KEYS[api_key]
```

### 6.2 Encryption

**Data at Rest:**
```python
from cryptography.fernet import Fernet

# Generate key (store securely)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt sensitive data
encrypted = cipher.encrypt(b"敏感数据")

# Decrypt
decrypted = cipher.decrypt(encrypted)
```

**Data in Transit:**
```yaml
# Kubernetes ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-app-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: rag-app-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-app-service
            port:
              number: 80
```

**Environment Variables:**
```python
import os
from cryptography.fernet import Fernet

# Decrypt environment variables
def get_secret(key: str) -> str:
    encrypted_value = os.getenv(key)
    if encrypted_value:
        cipher = Fernet(os.getenv("ENCRYPTION_KEY"))
        return cipher.decrypt(encrypted_value.encode()).decode()
    return None

OPENAI_API_KEY = get_secret("OPENAI_API_KEY_ENC")
```

### 6.3 Secret Management

**AWS Secrets Manager:**
```python
import boto3
import json

secretsmanager = boto3.client('secretsmanager')

def get_secret(secret_name: str) -> dict:
    try:
        response = secretsmanager.get_secret_value(
            SecretId=secret_name
        )
        return json.loads(response['SecretString'])
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        raise
```

**Kubernetes Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  pinecone-api-key: <base64-encoded-key>
```

**Vault (HashiCorp):**
```python
import hvac

client = hvac.Client(url='http://vault:8200')
client.token = os.getenv('VAULT_TOKEN')

# Read secret
response = client.secrets.kv.v2.read_secret_version(path='rag/api-keys')
api_keys = response['data']['data']
```

---

## 7. CI/CD PIPELINES

### 7.1 GitHub Actions

```yaml
name: Deploy RAG App

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest

    - name: Run tests
      run: pytest

    - name: Run linting
      run: |
        pip install black flake8
        black --check .
        flake8 .

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Build Docker image
      run: |
        docker build -t my-rag-app:${{ github.sha }} .
        docker tag my-rag-app:${{ github.sha }} my-rag-app:latest

    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push my-rag-app:${{ github.sha }}
        docker push my-rag-app:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/rag-app rag-app=my-rag-app:${{ github.sha }}
        kubectl rollout status deployment/rag-app
```

### 7.2 Jenkins Pipeline

```groovy
pipeline {
    agent any

    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'rag-app'
        IMAGE_TAG = "${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pytest'
            }
        }

        stage('Build') {
            steps {
                sh 'docker build -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .'
            }
        }

        stage('Push') {
            steps {
                sh 'echo ${DOCKER_PASSWORD} | docker login ${DOCKER_REGISTRY} -u ${DOCKER_USERNAME} --password-stdin'
                sh 'docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}'
            }
        }

        stage('Deploy') {
            steps {
                sh 'kubectl set image deployment/rag-app rag-app=${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}'
                sh 'kubectl rollout status deployment/rag-app'
            }
        }
    }

    post {
        always {
            sh 'docker logout'
        }
    }
}
```

---

## 8. HIGH AVAILABILITY

### 8.1 Load Balancing

**NGINX Config:**
```nginx
upstream rag_backend {
    server rag-app-1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server rag-app-2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server rag-app-3:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 8.2 Health Checks

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/ready")
def readiness_check():
    # Check dependencies
    try:
        # Check database
        db_status = check_database()

        # Check vector DB
        vector_db_status = check_vector_db()

        if db_status and vector_db_status:
            return {"status": "ready"}
        else:
            return {"status": "not ready"}, 503
    except Exception as e:
        return {"status": "error", "detail": str(e)}, 503

def check_database():
    # Implement DB health check
    pass

def check_vector_db():
    # Implement vector DB health check
    pass
```

### 8.3 Disaster Recovery

**Backup Strategy:**
```bash
#!/bin/bash
# Backup script

# Backup database
pg_dump -h postgres -U rag rag > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup vector database
chroma run --db-name rag_backup --path /backups/chroma

# Backup to S3
aws s3 sync /backups s3://my-backup-bucket/rag/

# Cleanup old backups (keep 30 days)
find /backups -name "*.sql" -mtime +30 -delete
```

**Recovery Procedure:**
```bash
#!/bin/bash
# Recovery script

# Stop application
kubectl scale deployment rag-app --replicas=0

# Restore database
psql -h postgres -U rag rag < backup_20231109_120000.sql

# Restore vector database
chroma run --db-name rag --path /data/chroma --restore --backup-path /backups/chroma

# Restart application
kubectl scale deployment rag-app --replicas=3
```

---

## 9. PERFORMANCE TUNING

### 9.1 Application Level

```python
# Caching
from functools import lru_cache
from redis import Redis

redis_client = Redis(host='redis', port=6379)

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return embeddings.embed_query(text)

def get_embedding_cached(text: str):
    cached = redis_client.get(f"emb:{hash(text)}")
    if cached:
        return json.loads(cached)

    embedding = embeddings.embed_query(text)
    redis_client.setex(
        f"emb:{hash(text)}",
        3600,  # 1 hour TTL
        json.dumps(embedding)
    )
    return embedding
```

### 9.2 Database Level

```sql
-- PostgreSQL optimization
CREATE INDEX CONCURRENTLY idx_documents_user_id
ON documents(user_id);

-- Vector index optimization
CREATE INDEX CONCURRENTLY idx_embeddings_vector
ON embeddings USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);
```

### 9.3 System Level

```bash
# Increase file limits
ulimit -n 65536

# Increase shared memory
sysctl -w kernel.shmmax=8589934592

# Enable swap
swapon --all
```

---

## 10. COST OPTIMIZATION

### 10.1 Cloud Costs

**Right-sizing:**
- Use smaller instances para development
- Scale down off-hours
- Spot instances para non-critical workloads
- Reserved instances para predictable load

**Managed Services:**
- Use managed vector DBs (reduces ops cost)
- Serverless para sporadic traffic
- CDN para static content
- Auto-scaling para load-based costs

### 10.2 Resource Optimization

```yaml
# Resource requests/limits
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 1000m
    memory: 2Gi

# Horizontal Pod Autoscaler
hpa:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### 10.3 Monitoring Costs

```python
# Track costs
import boto3

def track_cost(tags: dict):
    cloudwatch = boto3.client('cloudwatch')

    cloudwatch.put_metric_data(
        Namespace='RAG/App',
        MetricData=[
            {
                'MetricName': 'QueryCost',
                'Value': cost,
                'Unit': 'Count',
                'Dimensions': tags
            }
        ]
    )
```

---

## 11. WINDOWS-SPECIFIC

### 11.1 Docker Desktop

```powershell
# Enable WSL2 backend
wsl --set-default-version 2

# Install Docker Desktop
choco install docker-desktop

# Build and run
docker build -t my-rag-app .
docker run -p 8000:8000 my-rag-app
```

### 11.2 IIS Deployment

```xml
<!-- web.config -->
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler"
           path="*"
           verb="*"
           modules="CgiModule"
           scriptProcessor="C:\Python311\python.exe|C:\Python311\Lib\wfastcgi.py"
           resourceType="Unspecified" />
    </handlers>
    <security>
      <requestFiltering>
        <verbs>
          <add verb="GET,POST" allowed="true" />
        </verbs>
      </requestFiltering>
    </security>
  </system.webServer>
</configuration>
```

---

## 12. BEST PRACTICES

### 12.1 Development
- Use Docker para consistency
- Environment parity (dev/staging/prod)
- Automated testing
- Code reviews
- Security scanning

### 12.2 Deployment
- Blue-green deployments
- Canary releases
- Rolling updates
- Health checks
- Monitoring from day 1

### 12.3 Operations
- Documentation
- Runbooks
- Incident response
- Post-mortems
- Continuous improvement

### 12.4 Security
- Principle of least privilege
- Regular updates
- Vulnerability scanning
- Penetration testing
- Security reviews

---

## 13. TROUBLESHOOTING

### 13.1 Common Issues

**Pod Crashes:**
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl get events
```

**High Latency:**
```bash
# Check resource usage
kubectl top pods

# Check network
kubectl exec <pod-name> -- netstat -i

# Check disk
kubectl exec <pod-name> -- df -h
```

**Connection Issues:**
```bash
# Test service connectivity
kubectl run tmp-shell --rm -i --tty --image nicolaka/netshoot -- /bin/bash
nslookup rag-app-service
curl rag-app-service:8000/health
```

---

**Status**: ✅ Base para Production Deployment coletada
**Próximo**: Seção 12 - Troubleshooting
**Data Conclusão**: 09/11/2025

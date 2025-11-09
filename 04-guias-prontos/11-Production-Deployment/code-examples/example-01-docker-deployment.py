#!/usr/bin/env python3
"""
Example 01: Docker Deployment
============================

Demonstra deployment com Docker.

Uso:
    python example-01-docker-deployment.py
"""

import os
import subprocess
import time


def create_dockerfile():
    """Create Dockerfile"""
    dockerfile = """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    print("✅ Dockerfile created")


def create_docker_compose():
    """Create docker-compose.yml"""
    compose = """
version: '3.8'

services:
  rag-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    volumes:
      - ./data:/app/data
      - ./vectorstore:/app/vectorstore
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-app
    restart: unless-stopped
"""
    with open("docker-compose.yml", "w") as f:
        f.write(compose)
    print("✅ docker-compose.yml created")


def build_image():
    """Build Docker image"""
    print("\nBuilding Docker image...")
    result = subprocess.run(
        ["docker", "build", "-t", "rag-app", "."],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Image built successfully")
    else:
        print(f"❌ Build failed: {result.stderr}")


def run_container():
    """Run container"""
    print("\nRunning container...")
    result = subprocess.run(
        ["docker", "run", "-d", "-p", "8000:8000", "rag-app"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        container_id = result.stdout.strip()
        print(f"✅ Container running: {container_id}")

        # Wait for health check
        print("Waiting for app to be ready...")
        time.sleep(10)

        # Test endpoint
        result = subprocess.run(
            ["curl", "-f", "http://localhost:8000/health"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ App is healthy")
        else:
            print("⚠️  App not ready yet")

        return container_id
    else:
        print(f"❌ Container failed: {result.stderr}")
        return None


def stop_container(container_id):
    """Stop container"""
    print(f"\nStopping container {container_id}...")
    subprocess.run(["docker", "stop", container_id])
    subprocess.run(["docker", "rm", container_id])
    print("✅ Container stopped")


def main():
    """Função principal"""
    print("="*60)
    print("DOCKER DEPLOYMENT DEMO")
    print("="*60)

    # Create files
    create_dockerfile()
    create_docker_compose()

    # Build and run
    build_image()
    container_id = run_container()

    if container_id:
        print(f"\n{'='*60}")
        print("SUCCESS! Container is running")
        print(f"{'='*60}\n")

        print("Access your RAG app at: http://localhost:8000")
        print("\nEndpoints:")
        print("  GET  /health - Health check")
        print("  POST /query - Query RAG")
        print("  GET  /metrics - Prometheus metrics")

        # Keep running
        input("\nPress Enter to stop...")

        stop_container(container_id)

    print("\n" + "="*60)
    print("Deployment demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()

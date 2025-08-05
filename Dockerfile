```dockerfile
# Multi-stage Dockerfile for Quantum Annealed Hyperparameter Search

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml setup.cfg README.md ./
COPY quantum_hyper_search/ quantum_hyper_search/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -e ".[simulators,monitoring]"

# Base stage
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml setup.cfg setup.py ./
COPY quantum_hyper_search/__init__.py quantum_hyper_search/

# Install Python dependencies
RUN pip install -e ".[dev,simulators,monitoring]"

# Copy source code
COPY . .

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose port for Jupyter
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app/ /app/

# Copy source code
COPY quantum_hyper_search quantum_hyper_search/
COPY examples examples/

# Create cache and logs directories
RUN mkdir -p /app/cache /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m quantum_hyper_search.monitoring.health_check --quiet || exit 1

# Default command
CMD ["python", "-m", "quantum_hyper_search.examples.basic_usage"]

# D-Wave enabled stage
FROM production as dwave

USER root

# Install D-Wave dependencies
RUN pip install -e ".[dwave]"

USER appuser

# API service stage  
FROM production as api

USER root

# Install additional API dependencies
RUN pip install fastapi uvicorn[standard]

USER appuser

# Expose API port
EXPOSE 8000

# Copy API code
COPY api/ api/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Labels
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL version="0.1.0"
LABEL description="Quantum-Annealed Hyperparameter Search"
```

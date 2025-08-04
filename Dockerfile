# Multi-stage Dockerfile for Quantum Hyperparameter Search

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

# Production stage
FROM python:3.9-slim as production

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app/ /app/

# Set ownership
RUN chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m quantum_hyper_search.monitoring.health_check --quiet || exit 1

# Default command
CMD ["python", "-m", "quantum_hyper_search.monitoring.health_check"]

# Labels
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL version="0.1.0"
LABEL description="Quantum-Annealed Hyperparameter Search"
# Multi-stage production Dockerfile for Quantum Hyperparameter Search
# Built for: quantum-hyper-search v1.0.0

# Stage 1: Build Dependencies
FROM python:3.12-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production Runtime
FROM python:3.12-slim as production

# Labels for metadata
LABEL maintainer="Terragon Labs <contact@terragonlabs.com>" \
      version="1.0.0" \
      description="Enterprise Quantum Hyperparameter Optimization" \
      build-date=$BUILD_DATE \
      vcs-ref=$VCS_REF

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=quantum:quantum . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    QUANTUM_LOG_LEVEL=INFO \
    QUANTUM_CACHE_DIR=/app/cache \
    QUANTUM_DATA_DIR=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from quantum_hyper_search import QuantumHyperSearch; print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "quantum_hyper_search.main"]

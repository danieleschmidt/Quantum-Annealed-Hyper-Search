# Quantum Hyperparameter Search - Production Docker Image
# Multi-stage build for optimized production deployment

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set build arguments
ARG QUANTUM_BACKEND=simulator
ARG BUILD_ENV=production
ARG ENABLE_GPU=false

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt setup.py ./
COPY quantum_hyper_search/ ./quantum_hyper_search/

# Install the package and dependencies
RUN pip install -e .[all]

# Stage 2: Production runtime
FROM python:3.11-slim as runtime

# Set production environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV QHS_ENV=production
ENV QHS_LOG_LEVEL=INFO
ENV QHS_CONFIG_PATH=/app/config
ENV QHS_DATA_PATH=/app/data
ENV QHS_CACHE_PATH=/app/cache

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directories
RUN mkdir -p /app/config /app/data /app/cache /app/logs /var/log/qhs

# Create non-root user for security
RUN groupadd -r qhs && useradd -r -g qhs -d /app -s /bin/bash qhs

# Copy application code
COPY --chown=qhs:qhs quantum_hyper_search/ /app/quantum_hyper_search/
COPY --chown=qhs:qhs examples/ /app/examples/

# Set proper permissions
RUN chown -R qhs:qhs /app /var/log/qhs \
    && chmod 755 /app/config /app/data /app/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from quantum_hyper_search.monitoring.health_check import check_health; exit(0 if check_health() else 1)"

# Expose ports
EXPOSE 8000 8080 9090

# Volume mounts for persistence
VOLUME ["/app/data", "/app/cache", "/app/logs"]

# Switch to non-root user
USER qhs

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "-m", "quantum_hyper_search.examples.production_example"]

# Labels for metadata
LABEL maintainer="Terragon Labs <contact@terragonlabs.com>"
LABEL version="1.0.0"
LABEL description="Enterprise Quantum Hyperparameter Search System"
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/quantum-hyper-search"
LABEL org.opencontainers.image.documentation="https://quantum-hyper-search.terragonlabs.com"
LABEL org.opencontainers.image.licenses="Apache-2.0"

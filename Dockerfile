# Multi-stage Dockerfile for LangGraph Document Processing Pipeline
# Optimized for production deployment with security and performance

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Labels for metadata
LABEL maintainer="Document Processing Team" \
      version="${VERSION}" \
      description="LangGraph Document Processing Pipeline" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Document processing dependencies
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    # Network and security
    ca-certificates \
    curl \
    # Process management
    supervisor \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r docprocessing && \
    useradd -r -g docprocessing -d /app -s /bin/bash docprocessing

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/cache /app/temp /app/logs /app/data && \
    chown -R docprocessing:docprocessing /app

# Copy application code
COPY --chown=docprocessing:docprocessing . /app/

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOC_PROCESSING_ENV=production \
    LOG_LEVEL=INFO \
    ENABLE_MONITORING=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from core.monitoring import HealthChecker; HealthChecker().check_health()" || exit 1

# Switch to non-root user
USER docprocessing

# Expose port (if running web service)
EXPOSE 8000

# Default command
CMD ["python", "enhanced_main.py", "--config", "config.yaml"]

# Development stage (for local development)
FROM production as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    notebook \
    jupyterlab

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER docprocessing

# Override command for development
CMD ["python", "enhanced_main.py", "--config", "config.yaml", "--debug"]
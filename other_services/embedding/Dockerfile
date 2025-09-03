# NVIDIA NIM-compatible Embedding Service

FROM ubuntu:24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies following main project approach
# Skip pip upgrade on Ubuntu 24.04 to avoid RECORD file conflicts

# Install PyTorch with CUDA 12.1 support (same as main project)
# Use PyTorch 2.5+ for better RTX 50 series support  
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies (requirements.txt)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs conversations cache models

# Create non-root user for security (use different UID to avoid conflicts)
RUN useradd -m -u 1002 embedding_user && \
    chown -R embedding_user:embedding_user /app

# Switch to non-root user
USER embedding_user

# Expose port
EXPOSE 8009

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8009/v1/health/ready || exit 1

# Default command (can be overridden)
CMD ["python3", "embedding_service.py", "--host", "0.0.0.0", "--port", "8009"]

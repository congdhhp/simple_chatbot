# Use Ubuntu 24.04 base image
FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements files first for better caching
COPY requirements.txt requirements-api.txt ./

# Install Python dependencies following setup.py approach
# Skip pip upgrade on Ubuntu 24.04 to avoid RECORD file conflicts

# 1. Install PyTorch with CUDA 12.1 support (same as setup.py)
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install other dependencies (requirements.txt)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# 3. Install API dependencies (requirements-api.txt)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements-api.txt

# 4. Download and install flash-attention from wheel (following setup.py logic)
RUN echo "⚡ Downloading flash-attention wheel from GitHub releases..." && \
    wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    echo "⚡ Installing flash-attention from wheel..." && \
    pip3 install --no-cache-dir --break-system-packages flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    rm flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl || \
    echo "⚠️ Flash-attention installation failed, continuing without it..."

# 5. Fix bcrypt version compatibility issues - use compatible versions
RUN pip3 install --no-cache-dir --break-system-packages "bcrypt==4.0.1" "passlib[bcrypt]==1.7.4"

# 6. Additional production dependencies
RUN pip3 install --no-cache-dir --break-system-packages gunicorn

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs conversations

# Create non-root user for security (use different UID to avoid conflicts)
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8008

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

# Default command (can be overridden)
CMD ["python3", "nim_service.py", "--host", "0.0.0.0", "--port", "8008"]

# PerfAI CUDA Performance Testing - Multi-stage Docker Build
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ src/
COPY main.cpp .
COPY Makefile .

# Build CUDA components
RUN make clean && make all

# Runtime stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 perfai && \
    mkdir -p /app /data && \
    chown -R perfai:perfai /app /data

# Set working directory
WORKDIR /app

# Copy built executable from builder stage
COPY --from=builder /app/bin/cuda_benchmark /app/bin/
COPY --from=builder /app/src/ /app/src/

# Copy Python requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy configuration and scripts
COPY config.json .
COPY scripts/ scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh bin/cuda_benchmark

# Create data directories
RUN mkdir -p data/runs data/baselines logs

# Change ownership to perfai user
RUN chown -R perfai:perfai /app

# Switch to non-root user
USER perfai

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD bin/cuda_benchmark --test-mode || exit 1

# Default command
CMD ["python3", "src/pipeline/perfai_pipeline.py", "--config", "config.json"]

# Metadata
LABEL maintainer="PerfAI Project"
LABEL description="CUDA-accelerated performance regression detection using AI"
LABEL version="1.0.0"

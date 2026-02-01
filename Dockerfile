# Agent Orchestration System - Dockerfile (CPU)
# Multi-stage build for optimized image size

FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pyproject.toml README.md ./
COPY counsel/ ./counsel/
COPY main.py .
COPY tests/ ./tests/

# Install console entrypoint without re-resolving deps
RUN pip install --no-cache-dir --no-deps .

# Create projects directory for agent work
RUN mkdir -p /app/projects

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Default command (interactive mode)
CMD ["python", "-m", "counsel", "-i", "-w", "/app/projects"]

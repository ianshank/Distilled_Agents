# Enhanced Agent System Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models

# Expose ports
EXPOSE 8080 9090

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    ENHANCED_SYSTEM_ENV=production

# Run application (basic_inference has no :8080/health endpoint)
CMD ["python", "-m", "enhanced_system.examples.basic_inference"]


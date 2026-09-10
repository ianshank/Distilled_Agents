# Enhanced Agent System runtime image.
# Matches CI Python 3.11. SageMaker training estimators stay on py39/py310.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/data /app/models

EXPOSE 8080 9090

ENV PYTHONUNBUFFERED=1 \
    ENHANCED_SYSTEM_ENV=production \
    MANGOMAS_BIND_HOST=127.0.0.1

# No HEALTHCHECK: this image runs a one-shot example, not an HTTP server.
# Serve HTTP via scripts/inference.py (Flask /ping) when packaging an inference image.
CMD ["python", "-m", "examples.basic_inference"]

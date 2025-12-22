# Backend Dockerfile for Q-Route Alpha FastAPI
# Self-contained build - context is backend/ directory
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the q_route package (now included in backend/)
COPY q_route/ ./q_route/

# Copy backend code
COPY main.py ./

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Expose port (Railway uses PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI - use PORT env var for Railway compatibility
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

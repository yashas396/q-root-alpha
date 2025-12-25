# Root Dockerfile for Railway deployment
# Builds the backend from the repo root context
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies (from backend/)
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the q_route package (from backend/)
COPY backend/q_route/ ./q_route/

# Copy backend code
COPY backend/main.py ./

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Expose port (Railway uses PORT env var)
EXPOSE 8000

# Note: Railway handles health checks via railway.json config
# Removed Docker HEALTHCHECK to avoid conflicts

# Run FastAPI - use PORT env var for Railway compatibility
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

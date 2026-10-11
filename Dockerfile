# Stage A: Production Container Image for FlowInsight API
# Base image: python:3.11-slim
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    FLOWINSIGHT_DB_PATH=/app/data/flowinsight.db \
    FLOWINSIGHT_MODELS_DIR=/app/models_saved

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY flowinsight/ flowinsight/
COPY sql/ sql/
COPY scripts/ scripts/
COPY pyproject.toml .

# Install FlowInsight in editable/local mode
RUN pip install --no-cache-dir -e .

# Create non-root user for container security
RUN groupadd -r flowinsight && useradd -r -g flowinsight -d /app -s /bin/bash flowinsight && \
    mkdir -p /app/data /app/models_saved /app/reports && \
    chown -R flowinsight:flowinsight /app

# Switch to non-root user
USER flowinsight

# Expose API port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default launch command
CMD ["uvicorn", "flowinsight.api:app", "--host", "0.0.0.0", "--port", "8000"]

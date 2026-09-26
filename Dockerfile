# ==============================================================================
# Continuous Legal Memory (CLM) - Production Container Image
# Multi-stage build with non-root security context and air-gapped support.
# ==============================================================================

# --- Stage 1: Build & Dependency Resolution ---
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build essentials if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY continuous_legal_memory ./continuous_legal_memory

# Build isolated wheel
RUN pip install --no-cache-dir --upgrade pip build wheel && \
    python -m build --wheel --no-isolation --outdir /dist

# --- Stage 2: Production Distroless-like Minimal Runtime ---
FROM python:3.10-slim AS runtime

# OCI Metadata Annotations
LABEL org.opencontainers.image.title="Continuous Legal Memory" \
      org.opencontainers.image.description="Cognitive memory framework for legal AI agents." \
      org.opencontainers.image.authors="Vinicius Caridá <vfcarida@gmail.com>" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/vfcarida/continuous-legal-memory"

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLM_DATA_DIR="/data" \
    PORT=8000 \
    HOST="0.0.0.0"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group
RUN groupadd -g 10001 clm && \
    useradd -u 10001 -g clm -s /bin/false -m clm

# Set up data persistence directory
RUN mkdir -p /data && chown -R clm:clm /data

WORKDIR /app

# Copy built wheel from builder stage and install
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Switch to unprivileged user
USER clm:clm

# Mountable volume for SQLite episodic ledger and semantic graphs
VOLUME ["/data"]

# Expose REST API and Prometheus metrics port
EXPOSE 8000

# Container healthcheck querying /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Default command launches REST API server with data persistence
ENTRYPOINT ["clm"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--db-path", "/data/memory.db", "--enable-telemetry"]

# =============================================================================
# SmartCityAI — Backend Dockerfile
# Multi-stage build: deps layer cached separately from code layer.
# Changing code does NOT invalidate the pip install cache.
# =============================================================================

# ── Stage 1: system deps + pip install ───────────────────────────────────────
FROM python:3.11-slim AS builder

# System libraries required by:
#   psycopg2-binary  → libpq-dev
#   opencv-headless  → libglib2.0-0 libgl1
#   prophet/pystan   → gcc g++ (compiled C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        g++ \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy ONLY requirements first — Docker layer cache means this step
# is skipped on code-only changes (no requirements change = no pip reinstall)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy only the installed packages from builder (not gcc, build tools, etc.)
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Runtime system libs only (no gcc/g++ in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code
COPY . .

# Create directories that need to exist at runtime
RUN mkdir -p \
    app/nlp/models_store \
    app/nlp/data \
    app/cv/models_store \
    app/cv/data \
    app/timeseries/models_store \
    mlruns

# Non-root user — never run production containers as root
RUN useradd -m -u 1001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Docker health check — restarts container if API stops responding
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=25s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker for now (scale horizontally with multiple containers, not workers,
# because FastAPI WebSocket state is in-process)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
# ============================================================================
# SAM3 Segment Studio — backend (API only)
#
# Python runtime with uv-managed deterministic deps from uv.lock. The frontend
# is a separate service (frontend/Dockerfile + docker-compose.yml).
#
# Build:  docker build -t sam3-studio-backend:latest .
# Run:    docker run --rm -p 8000:8000 --env-file .env sam3-studio-backend:latest
# ============================================================================

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Fonts used by rendering.py when it draws labels/boxes on results
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app

# Install Python deps from the lockfile (reproducible, no dev group)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App source
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "sam3_studio.main", "--host", "0.0.0.0", "--port", "8000"]

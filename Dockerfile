# ============================================================================
# SAM3 Segment Studio — production image (multi-stage)
#
# Stage 1: build the React UI from source (no reliance on committed assets)
# Stage 2: Python runtime with uv-managed deterministic deps from uv.lock
#
# Build:  docker build -t sam3-studio:latest .
# Run:    docker run --rm -p 7860:7860 --env-file .env sam3-studio:latest
# ============================================================================

# ------------------------------------------------------------ 1) UI builder
FROM node:22-alpine AS ui-builder

WORKDIR /ui

# Install deps first (cached unless the lockfile changes)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Build from source — vite.config.ts emits to /src/sam3_studio/static
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------ 2) Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app

# Install Python deps from the lockfile (reproducible, no dev group)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App source, then fresh UI output (overwrites any committed static assets)
COPY src ./src
COPY --from=ui-builder /src/sam3_studio/static ./src/sam3_studio/static
RUN uv sync --frozen --no-dev

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)" || exit 1

CMD ["python", "-m", "sam3_studio.main", "--host", "0.0.0.0", "--port", "7860"]

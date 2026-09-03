# ============================================================================
# SAM3 Segment Studio — frontend image
#
# Builds the React app and serves it with Nginx, which also reverse-proxies
# API calls to the `backend` service (see docker-compose.yml).
#
# Build:  docker build -t sam3-studio-frontend:latest ./frontend
# ============================================================================

# ------------------------------------------------------------ 1) UI builder
FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .
RUN npm run build

# ------------------------------------------------------------ 2) Nginx
FROM nginx:1.27-alpine AS runtime

# Site config replaces the default (API proxy + SPA fallback)
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]

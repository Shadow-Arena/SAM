# SAM3 Segment Studio

Interactive **image segmentation** app powered by Meta's **SAM3** (Segment Anything Model 3) through 🤗 Transformers.
Segment images using **text**, **boxes**, **points**, or a **mixed combination** — draw on the image in the browser and hit *Run*.

- **API backend** — FastAPI (`src/sam3_studio/api`), **no UI**; serves `/segment`, `/health`, `/config`, `/outputs`.
- **Frontend** — standalone **React + TypeScript + Vite** app in `frontend/`, runs and builds on its own, served by Vite (dev/preview) or **Nginx** (Docker).
- **Production** — `docker compose up --build` runs both as separate services; Nginx serves the UI and reverse-proxies the API.
- **Text** — `"yellow school bus"`, `"ear"`, `"person"` → all matching instances (SAM3 Promptable Concept Segmentation).
- **Box** — one or more positive/negative boxes.
- **Point** — positive/negative clicks (SAM3 Tracker / Promptable Visual Segmentation).
- **Mixed** — text + boxes + points together; results merged and deduplicated.
- Pydantic config (`.env`), CORS allowlist for the separate frontend, and `Makefile` with `help`.

---

## Architecture

```text
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  frontend/  (React + Vite)  │  HTTP  │  src/sam3_studio  (FastAPI)  │
│  :5173 dev · :7860 preview  │ ─────▶ │  :8000 API                   │
│  Nginx in Docker (port 80)  │ proxies│  /segment /health /outputs   │
└─────────────────────────────┘        └──────────────────────────────┘
```

- The backend **never serves HTML** — it is a pure REST API with CORS.
- The frontend is **fully independent**: own `package.json`, build output (`frontend/dist/`), Dockerfile, and Nginx config.
- In production Nginx proxies `/segment`, `/health`, `/config`, `/outputs`, `/docs` to the `backend` container.

## Quick start (local dev, two terminals)

Requires **Python ≥ 3.10** + [uv](https://docs.astral.sh/uv/) and **Node 18+** (20/22 recommended):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pip install uv

# Terminal 1 — API backend
make run          # installs deps (uv sync), loads SAM3 once, serves API on :8000

# Terminal 2 — frontend
make frontend-dev  # Vite dev server on :5173, proxies API to :8000
```

Open **http://localhost:5173**. API docs: **http://localhost:8000/docs**.

> The first real start downloads `facebook/sam3` (~several GB). Smoke-test without the model:
>
> ```bash
> # Terminal 1
> make run-mock
> # Terminal 2
> make frontend-preview    # builds the UI and serves it on :7860
> ```
> Then open **http://localhost:7860**.

## Docker Compose (production)

Two independent images, orchestrated with Compose:

```bash
make env          # create .env (SAM_HF_TOKEN ...)
make docker-up    # docker compose up -d --build  → http://localhost:7860
make docker-logs
make docker-down
```

- `backend`  — built from root `Dockerfile` (uv.lock, no dev deps), internal port `8000`.
- `frontend` — built from `frontend/Dockerfile` (Node build → Nginx), published on `7860:80`.

Both are restart-safe; `outputs/` and the HF model cache live in named volumes.

## 🖥️ Web UI

1. Upload/drop an image.
2. Draw on the canvas:
   - 🟢 **Point +** — click for a **positive** point; **Shift+click** = negative point.
   - 🟢/🔴 **Box + / Box −** — drag a rectangle (positive / negative box).
   - **Clear prompts** empties everything; double-click a point to start over.
3. (optional) type a **text prompt**.
4. Pick **Auto / Text / Box / Point / Mixed** and press **Run segmentation**.

Outputs: overlay composite, per-object masks, an instances table, warnings,
and downloadable `composite.png`, `masks/*.png`, `result.json`, all served from `/outputs/`.

## API

| Method | Path        | Description                                    |
|--------|-------------|------------------------------------------------|
| GET    | `/`         | web UI                                         |
| GET    | `/docs`     | interactive OpenAPI docs                       |
| GET    | `/health`   | engine status (mock / device / model loaded)   |
| GET    | `/config`   | effective configuration (token masked)        |
| POST   | `/segment`  | run segmentation (multipart form)              |
| GET    | `/outputs/…`| saved composites / masks / JSON                |

Example:

```bash
curl -X POST http://localhost:7860/segment \
  -F image=@photos/street.jpg \
  -F mode=text \
  -F text="yellow school bus" \
  -F boxes_positive='[[10,10,200,300]]' \
  -F score_threshold=0.4 \
  -F opacity=0.6
```

Multipart fields: `image`, `mode`, `text`, `points_positive`, `points_negative`,
`boxes_positive`, `boxes_negative` (JSON arrays), `score_threshold`, `mask_threshold`,
`max_masks`, `opacity`, `show_semantic`.
The response contains `composite` and per-instance `mask` PNG data URIs plus `files` URLs.

## Hugging Face login (required for gated private checkpoints)

Some SAM3 checkpoints are gated — first authorize your account on
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then
log in with **one single variable**:

```bash
echo "SAM_HF_TOKEN=hf_xxx" >> .env      # .env is gitignored
```

That's the only login variable. If it's set, the app automatically exports it as
`HF_TOKEN` and calls `huggingface_hub.login(token)` at startup, and downloads
directly from `huggingface.co` (no mirror).

```bash
make login-status     # hf auth whoami
make config           # shows token status: hf_auth = configured (***xxxx)
```

## Makefile targets

```bash
make help        # show all targets (default)
make run         # install deps + load SAM3 once + start FastAPI  (PORT=7861 overrides)
make run-dev     # FastAPI with uvicorn auto-reload (development)
make run-mock    # start without downloading the model (mock engine)
make setup       # uv sync — .venv + deps from uv.lock (auto-run by make run)
make env         # copy .env.example -> .env
make config      # print effective configuration
make login       # log in to Hugging Face (make login TOKEN=hf_xxx)
make login-status# check the Hugging Face login
make test        # uv run pytest
make lint        # uv run ruff
make fmt         # uv run ruff (fix + format)
make check       # lint + tests
make preload     # download & cache the model ahead of time
make lock        # regenerate uv.lock
make update      # upgrade dependencies and refresh uv.lock
make clean purge # cleanup
```

Everything runs through `uv` (`uv sync`, `uv run`, `uv lock`), and `uv.lock` is committed for
reproducible installs. `make run` is the only command you need.

## Configuration (pydantic + `.env`)

Settings live in [`src/sam3_studio/config.py`](src/sam3_studio/config.py) (`SamSettings`, `pydantic-settings`).
Copy `.env.example` → `.env` (`make env`) and override anything with the `SAM_` prefix.

| Variable                  | Default          | Meaning                                 |
|---------------------------|------------------|-----------------------------------------|
| `SAM_MODEL_ID`            | `facebook/sam3`  | PCS model HF ID / local path            |
| `SAM_TRACKER_MODEL_ID`    | `facebook/sam3`  | tracker model (defaults to `model_id`)  |
| `SAM_DEVICE`              | `auto`           | `auto/cpu/cuda/mps/xpu`                 |
| `SAM_DTYPE`               | `auto`           | `auto/float32/float16/bfloat16`         |
| `SAM_LAZY_LOAD`           | `false`          | `false` = load once at startup (default)|
| `SAM_LOCAL_FILES_ONLY`    | `false`          | never contact the hub                   |
| `SAM_SCORE_THRESHOLD`     | `0.30`           | PCS score threshold                     |
| `SAM_MASK_THRESHOLD`      | `0.50`           | mask binarization threshold             |
| `SAM_IOU_MERGE_THRESHOLD` | `0.70`           | dedupe overlapping instances            |
| `SAM_MAX_MASKS`           | `100`            | cap on instances                        |
| `SAM_HOST` / `SAM_PORT`   | `0.0.0.0` / `7860` | server bind address                  |
| `SAM_OUTPUT_DIR`          | `outputs`        | where results are saved                 |
| `SAM_HF_TOKEN`            | —                | the ONLY Hugging Face login variable    |
| `SAM_MOCK`                | `false`          | synthetic engine (dev/tests)            |

Verify with `make config`.

## Project layout (src/ package)

```
src/sam3_studio/
  __init__.py        # package exports
  main.py            # server entry point (uvicorn)
  config.py          # pydantic-settings + .env
  domain.py          # PromptSet / MaskInstance / SegmentationResult
  prompts.py         # point clustering, prompt tensor builders, negative-point→box
  rendering.py       # mask/semantic overlays
  export.py          # result persistence (PNG + JSON)
  engine/            # segmentation engines
    errors.py        #   SegmentationError
    common.py        #   prompt validation + instance merging
    sam3.py          #   SAM3 PCS + tracker (real model)
    mock.py          #   synthetic engine (no download)
    factory.py       #   create_engine / get_engine singleton
  api/               # FastAPI layer (API ONLY — no HTML/static)
    app.py           #   create_app (CORS + /outputs + startup preload)
    deps.py          #   form parsing / upload helpers
    schemas.py       #   pydantic response models
    routes/          #   /segment, /, /health, /config
frontend/            # standalone React + TypeScript + Vite app
  src/               #   components/, App.tsx, api.ts, styles.css
  dist/              #   build output (gitignored — generated)
  Dockerfile         #   Node build → Nginx serving image
  nginx.conf         #   SPA + API reverse proxy
docker-compose.yml   # production stack (backend + frontend)
Dockerfile           # backend image (uv.lock, API only)
tests/               # pytest suite (unit/ + api/ + entry/)
Makefile             # setup/run/test/lint/help + frontend + docker targets
uv.lock              # reproducible Python dependency lockfile
.env.example         # documented backend configuration template
```

## Development

```bash
# Backend (API only)
make setup        # uv sync
make check        # uv run ruff + uv run pytest
make run          # FastAPI API on :8000 (mock: make run-mock)
make run-dev      # FastAPI with auto-reload

# Frontend (separate process)
make frontend-dev      # Vite dev server on :5173 (hot reload, proxies API)
make frontend-build    # production build → frontend/dist
make frontend-preview  # serve the built UI on :7860 (proxies API to :8000)
```

### Serving the frontend standalone

| Tool | Command | URL |
|---|---|---|
| Vite dev (HMR) | `make frontend-dev` | http://localhost:5173 |
| Built preview | `make frontend-preview` | http://localhost:7860 |
| Nginx (Docker) | `make docker-up` | http://localhost:7860 |

Point the built app at a backend on another origin with `VITE_API_BASE`
(see `frontend/.env.example`); by default it uses same-origin paths and the
serving layer proxies to the backend.

## Notes

- Both `Sam3Model` (text/boxes) and `Sam3TrackerModel` (points) are loaded from the
  same `facebook/sam3` checkpoint and loaded **once at startup** (`SAM_LAZY_LOAD=false`
  is the default), so the first text/box and the first point request are both instant.
  Set `SAM_LAZY_LOAD=true` (or `--lazy`) to open the server immediately and load on
  first request.
- **CPU works fine for trying it out** — expect roughly 30–120 s per image and
  ~6 GB RAM. Use a small image and one prompt for the first try. For real work,
  use a GPU (`SAM_DEVICE=cuda` in `.env`).
- Downloads come directly from `huggingface.co` (no mirror). If the download
  fails, check your token with `make login-status` or preload once with
  `make preload`.

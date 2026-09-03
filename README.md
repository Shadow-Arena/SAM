# SAM3 Segment Studio

Interactive **image segmentation** app powered by Meta's **SAM3** (Segment Anything Model 3) through 🤗 Transformers.
Segment images using **text**, **boxes**, **points**, or a **mixed combination** — draw on the image in the browser and hit *Run*.

- **FastAPI** backend (`src/sam3_studio/api`) with a **React + TypeScript + Vite** frontend (`frontend/`), built into `src/sam3_studio/static/`.
- **Text** — `"yellow school bus"`, `"ear"`, `"person"` → all matching instances (SAM3 Promptable Concept Segmentation).
- **Box** — one or more positive/negative boxes.
- **Point** — positive/negative clicks (SAM3 Tracker / Promptable Visual Segmentation).
- **Mixed** — text + boxes + points together; results merged and deduplicated.
- Pydantic config (`.env`) and `Makefile` with `help`.

---

## Quick start

Requires **Python ≥ 3.10** and [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pip install uv

make run        # ONE command: installs deps (uv sync), loads SAM3 once, starts the server
```

`make run` auto-runs `uv sync` first (incremental, so repeated runs are fast), then
loads the model **once** and starts the app at http://0.0.0.0:7860.
Open that URL in your browser; interactive API docs live at `/docs`.

> The first start downloads `facebook/sam3` (~several GB). For a UI smoke test **without** the model:
>
> ```bash
> make run-mock
> ```

## Docker

The image is **multi-stage**: the React UI is built from `frontend/` **inside** the image,
then the Python runtime installs deps from `uv.lock` — fully reproducible, never stale.

```bash
make docker          # docker build -t sam3-studio:latest .
make docker-run      # docker run --rm -p 7860:7860 --env-file .env sam3-studio:latest
```

Or manually:

```bash
docker build -t sam3-studio:latest .
docker run --rm -p 7860:7860 --env-file .env sam3-studio:latest
```

`--env-file .env` passes `SAM_HF_TOKEN` (and any overrides) into the container.
The image runs on port `7860` and includes a `/health` check.

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
  api/               # FastAPI layer
    app.py           #   create_app (routes + /static + startup preload)
    deps.py          #   form parsing / upload helpers
    schemas.py       #   pydantic response models
    routes/          #   /segment, /, /health, /config
  static/            # built React bundle (gitignored source in frontend/)
frontend/            # React + TypeScript + Vite UI source
  src/               #   components/, App.tsx, api.ts, styles.css
tests/               # pytest suite (unit/ + api/ + entry/)
Makefile             # setup/run/test/lint/help + frontend targets (uses uv; `make login` authenticates with HF)
uv.lock              # reproducible dependency lockfile
.env.example         # documented configuration template
```

## Development

```bash
make setup       # uv sync
make check       # uv run ruff + uv run pytest
make run-dev     # FastAPI with auto-reload
make run-mock    # verify the UI without downloading weights
```

### Frontend

The production bundle is committed, so `make run` needs no Node. To work on the UI:

```bash
make frontend-build   # rebuild frontend/ → src/sam3_studio/static
make frontend-dev     # Vite dev server on :5173, proxies the API to :7860
```

### Building the UI yourself (recommended before Docker/deploy)

If you have Node.js on your machine (tested with Node 22), build from source and
**commit the result** so the repo's `static/` bundle is always up to date:

```bash
cd frontend
npm ci --no-audit --no-fund   # install exact deps from package-lock.json
npm run build                 # type-check + production build → ../src/sam3_studio/static
cd ..
git diff --stat src/sam3_studio/static   # confirm the bundle changed/updated
git add -A && git commit -m "ui: rebuild frontend bundle"
git push
```

Then `make run` (or the Docker image) serves that fresh bundle.

**What this is for / what it is NOT for:**

| You build locally… | Result |
|---|---|
| Before deploying / changing the UI | ✅ Updates the committed `static/` bundle — everyone gets the new UI |
| Just to run the app | ❌ Not needed — `make run` uses the committed bundle (no Node) |
| For Docker | ⚠️ Not required — the Docker image rebuilds the UI **from `frontend/` source inside the image**, so it never depends on the committed bundle |

So the rule is simple: **`frontend/` = source of truth, and the build must happen in CI/Docker — but building locally and committing is the correct way to keep the local running bundle fresh** (and it's what you asked for).

### Quick smoke test after a local build

```bash
make run-mock     # starts FastAPI + UI without downloading the model
# open http://localhost:7860 — check the UI looks right
```

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

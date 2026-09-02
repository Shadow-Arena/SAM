# SAM3 Segment Studio

Interactive **image segmentation** app powered by Meta's **SAM3** (Segment Anything Model 3) through 🤗 Transformers.
Segment images using **text**, **boxes**, **points**, or a **mixed combination** — just draw on the image and click *Run*.

- **Text** — `"yellow school bus"`, `"ear"`, `"person"` → all matching instances (SAM3 Promptable Concept Segmentation).
- **Box** — one or more positive/negative boxes.
- **Point** — positive/negative clicks (SAM3 Tracker / Promptable Visual Segmentation).
- **Mixed** — text + boxes + points together; results merged and deduplicated.
- Pydantic config (`.env`), `Makefile` with `help`, Gradio UI, plus a CLI for batch/scripted use.

![architecture](https://img.shields.io/badge/model-facebook%2Fsam3-blue)

---

## Quick start

Requires **Python ≥ 3.10** and [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pip install uv

make setup      # uv sync: creates .venv, installs deps from uv.lock
make run        # launch UI at http://0.0.0.0:7860
```

> The first segmentation downloads `facebook/sam3` (~several GB). For a UI smoke test **without** the model:
>
> ```bash
> make run-mock
> ```

## Hugging Face login (required for gated private checkpoints)

Some SAM3 checkpoints are gated — first authorize your account on
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then
log in **once** with any of these options:

```bash
# 1) token in .env (recommended; auto-login on every app start)
echo "SAM_HF_TOKEN=hf_xxx" >> .env      # .env is gitignored

# 2) interactive login (saved in ~/.cache/huggingface/token)
make login
make login TOKEN=hf_xxx                 # non-interactive with a token

# 3) exactly like your notebook snippet
uv run python -c "from huggingface_hub import login; login('hf_xxx')"
```

If `SAM_HF_TOKEN` is set, the app calls `huggingface_hub.login(SAM_HF_TOKEN)` at
startup automatically (configurable with `SAM_HF_LOGIN=true|false`). Check it with:

```bash
make login-status     # hf auth whoami
make config           # shows whether a token is configured (masked)
```

## Makefile targets

```bash
make help        # show all targets (default)
make setup       # uv sync — .venv + deps from uv.lock (dev group included)
make lock        # regenerate uv.lock
make update      # upgrade dependencies and refresh uv.lock
make env         # copy .env.example -> .env
make run         # start the Gradio app        (PORT=7861 overrides)
make run-mock    # start the app using the synthetic mock engine
make segment     # one-shot CLI:  make segment ARGS="--image a.jpg --text car"
make config      # print effective configuration
make test        # uv run pytest
make lint        # uv run ruff
make fmt         # uv run ruff (fix + format)
make check       # lint + tests
make preload     # download & cache the model ahead of time
make clean purge # cleanup
```

Everything runs through `uv` (`uv sync`, `uv run`, `uv lock`), and `uv.lock` is committed for
reproducible installs. You can also invoke it directly:

```bash
uv run python -m app.main --port 7860
uv run python -m app.cli --image photo.jpg --text car
uv run pytest
```

## Usage

### Interactive UI

1. Upload/paste an image into the editor.
2. Draw strokes with the brush:
   - 🟢 green = **positive** prompt (seed/exclude-from candidate)
   - 🔴 red = **negative** prompt (exclude)
   - small strokes → **points**; large strokes / rectangles → **boxes**
3. (optional) type a **text prompt**.
4. pick **Auto / Text / Box / Point / Mixed** and press **Run segmentation**.

Outputs: overlay composite, per-object mask gallery, results table,
and downloadable `composite.png`, `masks/*.png`, `result.json` (COCO-style annotations).

| Mode    | Backend                                     | Prompts used                       |
|---------|---------------------------------------------|------------------------------------|
| Text    | `Sam3Model` (PCS)                           | text + optional pos/neg boxes      |
| Box     | `Sam3Model` (PCS, visual prompt)            | pos/neg boxes                      |
| Point   | `Sam3TrackerModel` (PVS)                    | pos/neg points                     |
| Mixed   | both                                        | text + boxes + points, merged      |

### CLI (one-shot)

```bash
# text
make segment ARGS="--image photos/street.jpg --text 'yellow school bus'"

# points (positive/negative)
make segment ARGS="--image img.jpg --point 320,240 --point 500,400 --negative-point 100,100"

# boxes
make segment ARGS="--image img.jpg --box 100,150,500,450"

# mixed
make segment ARGS="--image img.jpg --text handle --negative-box 40,183,318,204 --mode mixed"

# raw python
uv run python -m app.cli --image img.jpg --text car --score-threshold 0.4
```

## Configuration (pydantic + `.env`)

Settings live in [`app/config.py`](app/config.py) (`SamSettings`, `pydantic-settings`).
Copy `.env.example` → `.env` (`make env`) and override anything with the `SAM_` prefix.

| Variable                    | Default           | Meaning                                        |
|-----------------------------|-------------------|------------------------------------------------|
| `SAM_MODEL_ID`              | `facebook/sam3`   | PCS model HF ID / local path                   |
| `SAM_TRACKER_MODEL_ID`      | `facebook/sam3`   | tracker model (defaults to `model_id`)         |
| `SAM_DEVICE`                | `auto`            | `auto/cpu/cuda/mps/xpu`                        |
| `SAM_DTYPE`                 | `auto`            | `auto/float32/float16/bfloat16`                |
| `SAM_LAZY_LOAD`             | `true`            | load model on first request                    |
| `SAM_LOCAL_FILES_ONLY`      | `false`           | never contact the hub                          |
| `SAM_SCORE_THRESHOLD`       | `0.30`            | PCS score threshold                            |
| `SAM_MASK_THRESHOLD`        | `0.50`            | mask binarization threshold                    |
| `SAM_IOU_MERGE_THRESHOLD`   | `0.70`            | dedupe overlapping instances                   |
| `SAM_MAX_MASKS`             | `100`             | cap on instances                               |
| `SAM_HOST` / `SAM_PORT`     | `0.0.0.0` / `7860`| UI bind address                                |
| `SAM_SHARE`                 | `false`           | public gradio share link                       |
| `SAM_OUTPUT_DIR`            | `outputs`         | where results are saved                        |
| `SAM_HF_ENDPOINT`            | —                 | HF mirror, e.g. `https://hf-mirror.com`        |
| `SAM_HF_TOKEN`              | —                 | token for gated/private checkpoints            |
| `SAM_HF_LOGIN`               | `true`            | auto `huggingface_hub.login(SAM_HF_TOKEN)`     |
| `SAM_MOCK`                  | `false`           | synthetic engine (dev/tests, no download)      |

Verify with `make config`.

## Project layout

```
app/
  config.py        # pydantic-settings + .env
  schemas.py       # PromptSet / MaskInstance / SegmentationResult
  annotations.py   # editor strokes -> points/boxes (colors, connected components)
  segmentation.py  # SAM3 PCS + tracker engines, merging, lazy loading
  visualization.py # overlay, gallery, result export (PNG + JSON)
  ui.py            # Gradio interface
  main.py          # UI entry point (python -m app.main)
  cli.py           # one-shot CLI (python -m app.cli)
Makefile           # setup/run/test/lint/help (uses uv; `make login` authenticates with HF)
uv.lock            # reproducible dependency lockfile
.env.example       # documented configuration template
tests/             # pytest suite
```

## Development

```bash
make setup       # uv sync
make check       # uv run ruff + uv run pytest
make run-mock    # verify the UI without downloading weights
```

## Notes

- Both `Sam3Model` (text/boxes) and `Sam3TrackerModel` (points) are loaded from the
  same `facebook/sam3` checkpoint; tracker weights are fetched lazily on first point use.
- CPU works but is slow — CUDA strongly recommended (`SAM_DEVICE=cuda`).
- If `huggingface.co` is unreachable, set `SAM_HF_ENDPOINT=https://hf-mirror.com`
  and/or preload with `make preload`.

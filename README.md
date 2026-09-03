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

## 🛠️ Troubleshooting: no output in the browser

If the server starts (logs show `* Running on local URL: http://0.0.0.0:7860`)
but the page is blank / no segmentation appears, check these **in order**:

1. **Colab / Kaggle / remote VM** — `http://localhost:7860` points to *your* machine, not the notebook. Use:
   ```bash
   make run-share        # public https link like https://xxx.gradio.live — works anywhere
   ```
   or, inside the notebook:
   ```python
   from google.colab import output
   output.serve_kernel_port_as_window(7860)
   ```

2. **Enabled GPU?** Without one, SAM3 runs on CPU and a single request can take
   **minutes** — the notebook proxy may time out (and the log shows
   `WARNING: Invalid HTTP request received.`). In Colab choose
   `Runtime → Change runtime type → T4 GPU`, then:
   ```bash
   echo "SAM_DEVICE=cuda" >> .env   # verify with: make config
   ```

3. **Model loads once at startup** (default): `make run` downloads/loads the
   weights before the UI opens — expect a long first start, then instant clicks:
   ```bash
   make run        # "Loading SAM3 model (once) at startup ..." -> "SAM3 ready on cpu."
   make preload    # optional: download weights once, before launching
   ```

4. **Check the status bar** in the UI — `✅ N object(s) segmented …` appears under
   the Run button after each request, with the exact error if anything failed.

5. `WARNING: Invalid HTTP request received.` alone is usually just a proxy
   health-check hitting the HTTP port — ignore it if the app otherwise works.

## Hugging Face login (required for gated private checkpoints)

Some SAM3 checkpoints are gated — first authorize your account on
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then
log in with **one single variable**:

```bash
echo "SAM_HF_TOKEN=hf_xxx" >> .env      # .env is gitignored
```

That's the only login variable. If it's set, the app automatically exports it as
`HF_TOKEN` and calls `huggingface_hub.login(token)` at startup — same as your
`login(SAM_HF_TOKEN)` snippet — and downloads directly from `huggingface.co`
(no mirror).

```bash
make login-status     # hf auth whoami
make config           # shows token status: hf_auth = configured (***xxxx)
```

## Makefile targets

```bash
make help        # show all targets (default)
make setup       # uv sync — .venv + deps from uv.lock (dev group included)
make lock        # regenerate uv.lock
make update      # upgrade dependencies and refresh uv.lock
make env         # copy .env.example -> .env
make run         # start the app — model loads ONCE at startup   (PORT=7861 overrides)
make run-share   # public gradio.live link — Colab/Kaggle/remote
make run-preload # same as make run (explicit preload at startup)
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
| `SAM_LAZY_LOAD`             | `false`           | `false` = load once at startup (default)       |
| `SAM_LOCAL_FILES_ONLY`      | `false`           | never contact the hub                          |
| `SAM_SCORE_THRESHOLD`       | `0.30`            | PCS score threshold                            |
| `SAM_MASK_THRESHOLD`        | `0.50`            | mask binarization threshold                    |
| `SAM_IOU_MERGE_THRESHOLD`   | `0.70`            | dedupe overlapping instances                   |
| `SAM_MAX_MASKS`             | `100`             | cap on instances                               |
| `SAM_HOST` / `SAM_PORT`     | `0.0.0.0` / `7860`| UI bind address                                |
| `SAM_SHARE`                 | `false`           | public gradio share link                       |
| `SAM_OUTPUT_DIR`            | `outputs`         | where results are saved                        |
| `SAM_HF_TOKEN`               | —                 | the ONLY Hugging Face login variable           |
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
- **CPU works fine for trying it out** — expect roughly 30–120 s per image on a
  typical Colab CPU and ~6 GB RAM. Use a small image and one prompt for the first
  try. For real work, use a GPU (`Runtime → Change runtime type → T4`, then
  `echo "SAM_DEVICE=cuda" >> .env`).
- Both models (PCS + tracker) load **once at startup** (`SAM_LAZY_LOAD=false` is
  the default), so the first text/box and the first point request are both
  instant. Set `SAM_LAZY_LOAD=true` (or `make run` + `--lazy`) if you want the
  server to open instantly and load only on the first request.
- Downloads come directly from `huggingface.co` (no mirror). If the download
  fails, check your token with `make login-status` or preload once with
  `make preload`.

"""Gradio UI for interactive SAM3 segmentation."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime

import gradio as gr

from .annotations import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    analyze_layers,
    annotations_to_prompt_set,
    normalize_editor_value,
)
from .config import ModeChoice, SamSettings
from .segmentation import SegmentationError, get_engine
from .visualization import gallery_item, overlay_masks, overlay_semantic, save_result

logger = logging.getLogger(__name__)
_HANDLER_LOCK = threading.Lock()

GUIDE = """
### 🎯 How to prompt
1. **Upload / paste an image** in the editor.
2. **Draw on the image** with the brush:
   - 🟢 **Green** = **positive** prompt (segment this)
   - 🔴 **Red** = **negative** prompt (exclude this)
3. Small strokes become **points**, large strokes or rectangles become **boxes**.
4. Optionally type a **text** prompt (e.g. *"yellow school bus"*, *"handle"*).
5. Press **Run segmentation**.
"""

MODE_INFO = {
    "Auto": "Auto: text/boxes → SAM3 PCS; points → SAM3 tracker.",
    "Text": "Text (+optional green/red boxes) → SAM3 Promptable Concept Segmentation.",
    "Box": "Boxes only → SAM3 PCS visual prompt.",
    "Point": "Points only → SAM3 Tracker (PVS).",
    "Mixed": "Text/boxes + points → both models, results merged.",
}


def make_segment_handler(settings: SamSettings, engine_factory: Callable | None = None):
    """Build the segmentation event handler (also reused by tests)."""

    engine = engine_factory(settings) if engine_factory else get_engine(settings)

    def _noop_progress(*_args, **_kwargs):
        return None

    def segment_click(
        editor_value,
        mode: str,
        text: str,
        score_threshold: float,
        mask_threshold: float,
        opacity: float,
        max_masks: int,
        show_semantic: bool,
        progress=None,
    ):
        progress = progress or _noop_progress
        try:
            with _HANDLER_LOCK:
                background, layers = normalize_editor_value(editor_value)
                if background is None:
                    return None, None, [], [], "⚠️ Upload an image first."
                annotations, layer_warnings = analyze_layers(layers, background.size, settings)
                prompt_set = annotations_to_prompt_set(annotations, settings, text=text)
                if not prompt_set:
                    return None, None, [], [], "⚠️ Provide a prompt: draw points/boxes or type text."
                progress(0.05, desc="Running segmentation…")
                mode_enum = (
                    ModeChoice(mode.lower()) if mode.lower() in ModeChoice._value2member_map_ else ModeChoice.AUTO
                )
                result = engine.segment(
                    background,
                    prompt_set,
                    mode=mode_enum,
                    score_threshold=score_threshold,
                    mask_threshold=mask_threshold,
                    max_masks=max_masks,
                    progress=lambda msg: progress(desc=msg),
                )
                composite = overlay_masks(background, result.instances, opacity=opacity, draw_boxes=True)
                if show_semantic and result.semantic_mask is not None:
                    composite = overlay_semantic(composite, result.semantic_mask, opacity=0.35)

                items = [
                    (gallery_item(background, inst), f"#{inst.object_id} {inst.source}") for inst in result.instances
                ]
                rows = [
                    [
                        inst.object_id,
                        inst.source,
                        round(inst.score, 3) if inst.score is not None else "",
                        *[round(v, 1) for v in inst.box],
                        inst.area_px,
                    ]
                    for inst in result.instances
                ]
                run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                paths = save_result(composite, result, settings.output_dir, run_id)
                files = [paths["composite"], paths.get("json", "")]
                if paths.get("masks"):
                    files += list(paths["masks"])

                warnings = "; ".join(result.warnings or [])
                status = (
                    f"✅ **{len(result.instances)} object(s)** segmented in "
                    f"{result.elapsed_seconds:.2f}s — mode **{mode}**, prompts: {prompt_set.describe()}."
                )
                if warnings:
                    status += f"\n\n⚠️ {warnings}"
                return composite, items, rows, files, status
        except SegmentationError as exc:
            return None, None, [], [], f"❌ {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("segmentation failed")
            return None, None, [], [], f"❌ Unexpected error: {exc}"

    return segment_click


def make_load_handler(settings: SamSettings, engine_factory: Callable | None = None):
    """Status text shown when the page loads."""
    engine = engine_factory(settings) if engine_factory else get_engine(settings)

    def on_load() -> str:
        if settings.mock:
            return "🧪 Mock engine active (`SAM_MOCK=true`) — results are synthetic."
        if settings.lazy_load:
            return "⚙️ Model loads on first segmentation request. Set `SAM_LAZY_LOAD=false` to preload."
        try:
            engine.ensure_pcs(print)
            return f"✅ Model ready on {engine.device}."
        except Exception as exc:  # noqa: BLE001
            return f"❌ Model load failed: {exc}"

    return on_load


def build_app(settings: SamSettings | None = None, engine_factory: Callable | None = None) -> gr.Blocks:
    """Create the Gradio application."""
    settings = settings or SamSettings()
    segment_click = make_segment_handler(settings, engine_factory)
    on_load = make_load_handler(settings, engine_factory)

    def clear_outputs():
        return None, None, [], [], []

    with gr.Blocks(title="SAM3 Segment Studio") as demo:
        gr.Markdown(
            "# 🔬 SAM3 Segment Studio\nInteractive **Promptable Concept Segmentation** — segment images with "
            "**text, boxes, points or mixed prompts**."
        )
        gr.Markdown(
            "Model: **Facebook SAM3** via 🤗 Transformers — all matching instances of a concept, "
            "or prompt-specific objects."
        )

        with gr.Row():
            with gr.Column(scale=7):
                editor = gr.ImageEditor(
                    label="Image + prompts (green = positive, red = negative)",
                    type="pil",
                    image_mode="RGBA",  # required: annotation strokes are extracted from alpha layers
                    sources=["upload", "clipboard"],
                    brush=gr.Brush(
                        colors=[(f"rgb{POSITIVE_COLOR}", 1.0), (f"rgb{NEGATIVE_COLOR}", 1.0)],
                        default_color=f"rgb{POSITIVE_COLOR}",
                        color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=24),
                    layers=gr.LayerOptions(allow_additional_layers=False),
                    buttons=["fullscreen", "download"],
                    format="png",
                    height=620,
                )
            with gr.Column(scale=4):
                gr.Markdown(GUIDE)
                mode = gr.Radio(choices=list(MODE_INFO.keys()), value="Auto", label="Prompt mode")
                mode_info = gr.Markdown(value=MODE_INFO["Auto"])
                text_prompt = gr.Textbox(
                    label="Text prompt (optional)",
                    placeholder='e.g. "ear", "yellow school bus", "person"',
                    lines=2,
                )
                with gr.Accordion("Advanced", open=False):
                    score_slider = gr.Slider(
                        0.0, 1.0, value=settings.score_threshold, step=0.01, label="Score threshold"
                    )
                    mask_slider = gr.Slider(0.0, 1.0, value=settings.mask_threshold, step=0.01, label="Mask threshold")
                    opacity_slider = gr.Slider(
                        0.0, 1.0, value=settings.mask_opacity, step=0.05, label="Overlay opacity"
                    )
                    max_masks = gr.Number(value=settings.max_masks, precision=0, label="Max masks")
                    show_semantic = gr.Checkbox(value=False, label="Show semantic mask too (text mode)")
                run_btn = gr.Button("🚀 Run segmentation", variant="primary", size="lg")
                status = gr.Markdown(value="", elem_id="status")

        gr.Markdown("### Results")
        with gr.Row():
            with gr.Column(scale=6):
                output_image = gr.Image(label="Segmentation overlay", type="pil", interactive=False)
            with gr.Column(scale=6):
                gallery = gr.Gallery(label="Object masks", columns=3, height=420)
        results_table = gr.Dataframe(
            headers=["#", "source", "score", "x1", "y1", "x2", "y2", "area_px"],
            label="Detected instances",
            interactive=False,
        )
        download = gr.File(label="Downloads (composite, masks, JSON)", interactive=False)

        mode.change(lambda m: MODE_INFO[m], inputs=mode, outputs=mode_info)
        run_btn.click(
            segment_click,
            inputs=[editor, mode, text_prompt, score_slider, mask_slider, opacity_slider, max_masks, show_semantic],
            outputs=[output_image, gallery, results_table, download, status],
            api_name="segment",
        )
        clear_btn = gr.Button("🧹 Clear results", size="sm")
        clear_btn.click(clear_outputs, outputs=[output_image, gallery, results_table, download, status], api_name=False)
        demo.load(on_load, outputs=status, api_name=False)

    return demo

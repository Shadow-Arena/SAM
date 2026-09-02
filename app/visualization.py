"""Rendering helpers: mask overlays, galleries and result persistence."""

from __future__ import annotations

import colorsys
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schemas import MaskInstance, SegmentationResult


def colormap(index: int, total: int) -> tuple[int, int, int]:
    """Distinct colors for masks (HSV wheel, avoiding pure red/green)."""
    if total <= 1:
        hue = 0.58  # blue-ish
    else:
        hue = (0.58 + 0.75 * index / max(total - 1, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_masks(
    image: Image.Image,
    instances: Sequence[MaskInstance],
    opacity: float = 0.55,
    draw_boxes: bool = True,
    show_scores: bool = True,
) -> Image.Image:
    """Return the input image with colored mask overlays, outlines, boxes, labels."""
    image = image.convert("RGBA")
    w, h = image.size
    total = max(1, len(instances))
    mask_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for i, inst in enumerate(instances):
        color = colormap(i, total)
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[inst.mask] = int(255 * opacity)
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[..., 0], mask_rgba[..., 1], mask_rgba[..., 2] = color
        mask_rgba[..., 3] = alpha
        mask_overlay = Image.alpha_composite(mask_overlay, Image.fromarray(mask_rgba, "RGBA"))

        # Outline: mask pixels that are not all-neighbors-inside.
        m = inst.mask
        eroded = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
        edge = m & ~eroded
        edge_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        edge_rgba[..., 0], edge_rgba[..., 1], edge_rgba[..., 2] = color
        edge_rgba[..., 3] = np.where(edge, 255, 0)
        mask_overlay = Image.alpha_composite(mask_overlay, Image.fromarray(edge_rgba, "RGBA"))

    result = Image.alpha_composite(image, mask_overlay)
    if draw_boxes:
        draw = ImageDraw.Draw(result)
        font = _load_font(max(12, min(w, h) // 40))
        for i, inst in enumerate(instances):
            color = colormap(i, total)
            x0, y0, x1, y1 = inst.box
            draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=2)
            label = f"#{inst.object_id}"
            if inst.source:
                label += f" {inst.source}"
            if show_scores and inst.score is not None:
                label += f" {inst.score:.2f}"
            text_bbox = draw.textbbox((x0, y0), label, font=font)
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 2],
                fill=color + (235,),
            )
            draw.text((text_bbox[0], text_bbox[1]), label, fill=(20, 20, 20, 255), font=font)

    return result.convert("RGB")


def overlay_semantic(image: Image.Image, semantic_mask: np.ndarray, opacity: float = 0.4) -> Image.Image:
    """Overlay a binary semantic mask in yellow."""
    image = image.convert("RGBA")
    w, h = image.size
    sem = np.asarray(Image.fromarray(semantic_mask.astype(np.uint8)).resize((w, h))) > 0
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1] = 255, 200
    rgba[..., 3] = np.where(sem, int(255 * opacity), 0)
    return Image.alpha_composite(image, Image.fromarray(rgba, "RGBA")).convert("RGB")


def gallery_item(image: Image.Image, inst: MaskInstance) -> Image.Image:
    """A single object: dimmed original with its mask + label."""
    base = image.convert("RGB")
    arr = np.asarray(base).copy().astype(np.float32)
    dimmed = (arr * 0.35).astype(np.uint8)
    arr[inst.mask] = arr[inst.mask] // 2 + 128
    item = Image.fromarray(np.where(inst.mask[..., None], arr, dimmed).astype(np.uint8))
    draw = ImageDraw.Draw(item)
    label = f"#{inst.object_id}" + (f" {inst.source}" if inst.source else "")
    if inst.score is not None:
        label += f" ({inst.score:.2f})"
    font = _load_font(max(12, min(item.size) // 40))
    tbox = draw.textbbox((4, 4), label, font=font)
    draw.rectangle([tbox[0] - 2, tbox[1] - 1, tbox[2] + 2, tbox[3] + 2], fill=(255, 255, 255, 220))
    draw.text((tbox[0], tbox[1]), label, fill=(10, 10, 10), font=font)
    return item


def save_result(image: Image.Image, result: SegmentationResult, output_dir: Path, run_id: str) -> dict[str, str]:
    """Persist composite, per-instance masks, semantic mask and JSON.

    Returns a dict of file paths for the UI.
    """
    run_dir = Path(output_dir) / run_id
    masks_dir = run_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    composite = overlay_masks(image, result.instances, draw_boxes=True)
    composite_path = run_dir / "composite.png"
    composite.save(composite_path)
    paths["composite"] = str(composite_path)

    if result.semantic_mask is not None:
        sem_path = run_dir / "semantic.png"
        Image.fromarray(result.semantic_mask.astype(np.uint8) * 255).save(sem_path)
        paths["semantic"] = str(sem_path)

    for inst in result.instances:
        mask_img = Image.fromarray(inst.mask.astype(np.uint8) * 255)
        path = masks_dir / f"mask_{inst.object_id:04d}.png"
        mask_img.save(path)
        paths.setdefault("masks", [])
        paths["masks"].append(str(path))  # type: ignore[union-attr]

    payload = result.to_json()
    payload["files"] = paths
    json_path = run_dir / "result.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["json"] = str(json_path)
    return paths

"""Result persistence: composite, per-instance masks and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .domain import SegmentationResult

SaveResultPaths = dict[str, str | list[str]]


def save_result(image: Image.Image, result: SegmentationResult, output_dir: Path, run_id: str) -> SaveResultPaths:
    """Persist the already-rendered composite, per-instance masks, semantic
    mask and JSON.

    The caller passes the final composite (rendered with the request opacity);
    we never re-overlay it. Returns a dict of file paths for the API.
    """
    run_dir = Path(output_dir) / run_id
    masks_dir = run_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths: SaveResultPaths = {}

    composite_path = run_dir / "composite.png"
    image.save(composite_path)
    paths["composite"] = str(composite_path)

    if result.semantic_mask is not None:
        sem_path = run_dir / "semantic.png"
        Image.fromarray(result.semantic_mask.astype("uint8") * 255).save(sem_path)
        paths["semantic"] = str(sem_path)

    masks_paths: list[str] = []
    for inst in result.instances:
        mask_img = Image.fromarray(inst.mask.astype("uint8") * 255)
        path = masks_dir / f"mask_{inst.object_id:04d}.png"
        mask_img.save(path)
        masks_paths.append(str(path))
    if masks_paths:
        paths["masks"] = masks_paths

    payload = result.to_json()
    payload["files"] = paths
    json_path = run_dir / "result.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["json"] = str(json_path)
    return paths

"""SAM3 segmentation engine.

Wrapper around the 🤗 Transformers ``Sam3Model``/``Sam3Processor`` (Promptable
Concept Segmentation: text + boxes) and ``Sam3TrackerModel``/
``Sam3TrackerProcessor`` (Promptable Visual Segmentation: points).

Mode support:

* ``text``  — SAM3 PCS text prompt, optional positive/negative boxes,
* ``box``   — SAM3 PCS box prompt(s) (positive/negative),
* ``point`` — SAM3 tracker point prompt(s) (positive/negative clicks),
* ``mixed`` — PCS (text/boxes) plus tracker (points), results merged.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from .annotations import cluster_positive_points, negative_point_to_box
from .config import (
    ModeChoice,
    SamSettings,
    resolve_device,
    resolve_torch_dtype,
)
from .schemas import MaskInstance, PromptSet, SegmentationResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None] | None


class SegmentationError(RuntimeError):
    """Raised when a segmentation request cannot be executed."""


def _report(callback: ProgressCallback, message: str) -> None:
    logger.info(message)
    if callback is not None:
        try:
            callback(message)
        except Exception:  # pragma: no cover - UI callbacks must never break inference
            pass


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


_SOURCE_ORDER = ["text", "box", "point"]


def merge_sources(a: str, b: str) -> str:
    """Combine source labels in canonical order, e.g. 'text' + 'point' -> 'text+point'."""
    parts: list[str] = []
    for part in [a, b]:
        for token in part.split("+"):
            if token and token not in parts:
                parts.append(token)
    parts.sort(key=lambda token: _SOURCE_ORDER.index(token) if token in _SOURCE_ORDER else len(_SOURCE_ORDER))
    return "+".join(parts)


def validate_prompt(mode: ModeChoice, prompt: PromptSet) -> None:
    """Raise :class:`SegmentationError` for invalid prompt/mode combinations."""
    if mode == ModeChoice.TEXT and not prompt.has_text and not prompt.has_boxes:
        raise SegmentationError("Text mode needs a text prompt and/or box prompt.")
    if mode == ModeChoice.BOX and not prompt.has_boxes:
        raise SegmentationError("Box mode needs at least one box annotation.")
    if mode == ModeChoice.POINT and not prompt.points_positive:
        raise SegmentationError("Point mode needs at least one positive point annotation.")
    if mode == ModeChoice.MIXED and not (prompt.has_text or prompt.has_boxes or prompt.points_positive):
        raise SegmentationError("Mixed mode needs a text/box prompt and/or point prompt.")


def merge_instances(instances: Sequence[MaskInstance], iou_threshold: float = 0.7) -> list[MaskInstance]:
    """Deduplicate overlapping instances, keeping the highest score."""
    ordered = sorted(
        enumerate(instances),
        key=lambda pair: pair[1].score if pair[1].score is not None else -1.0,
        reverse=True,
    )
    kept: list[MaskInstance] = []
    kept_by_index: dict[int, int] = {}  # original index -> slot in `kept`
    for original_index, inst in ordered:
        dup_slot: int | None = None
        best_iou = 0.0
        for slot, other in enumerate(kept):
            iou = _bbox_iou(inst.mask, other.mask)
            if iou > best_iou:
                best_iou, dup_slot = iou, slot
        if dup_slot is not None and best_iou > iou_threshold:
            # Merge: record the combined source on the kept (higher-score) mask.
            kept[dup_slot].source = merge_sources(kept[dup_slot].source, inst.source)
            if inst.score is not None and (kept[dup_slot].score is None or inst.score > kept[dup_slot].score):
                kept[dup_slot].score = inst.score
            kept_by_index[original_index] = dup_slot
        else:
            kept.append(inst)
            kept_by_index[original_index] = len(kept) - 1
    for idx, inst in enumerate(kept):
        inst.object_id = idx + 1
    return kept


class SegmentationEngine:
    """Lazy-loading SAM3 inference engine (PCS + tracker)."""

    def __init__(self, settings: SamSettings | None = None):
        self.settings = settings or SamSettings()
        self.settings.apply_hf_environment()
        self.device = resolve_device(self.settings.device)
        self.torch_dtype = resolve_torch_dtype(self.settings.dtype, self.device)
        self._lock = threading.Lock()
        self._pcs: tuple | None = None
        self._tracker: tuple | None = None
        self._pending_semantic: np.ndarray | None = None
        self.last_load_error: str | None = None

    # ------------------------------------------------------------------ lazy loaders
    def ensure_pcs(self, progress: ProgressCallback = None) -> tuple:
        """Load (Sam3Model, Sam3Processor) once."""
        if self._pcs is not None:
            return self._pcs
        with self._lock:
            if self._pcs is not None:
                return self._pcs
            try:
                _report(progress, f"Loading SAM3 PCS model {self.settings.model_id} on {self.device} ...")
                from transformers import Sam3Model, Sam3Processor

                kwargs = dict(
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=self.settings.low_cpu_mem_usage,
                    use_safetensors=self.settings.use_safetensors,
                    local_files_only=self.settings.local_files_only,
                )
                model = Sam3Model.from_pretrained(self.settings.model_id, **kwargs).to(self.device)
                model.eval()
                processor = Sam3Processor.from_pretrained(
                    self.settings.model_id, local_files_only=self.settings.local_files_only
                )
                self._pcs = (model, processor)
                self.last_load_error = None
                _report(progress, f"SAM3 PCS ready on {self.device}.")
                return self._pcs
            except Exception as exc:  # noqa: BLE001
                self.last_load_error = f"Failed to load SAM3 PCS model: {exc}"
                raise SegmentationError(self.last_load_error) from exc

    def ensure_tracker(self, progress: ProgressCallback = None) -> tuple:
        """Load (Sam3TrackerModel, Sam3TrackerProcessor) once."""
        if self._tracker is not None:
            return self._tracker
        with self._lock:
            if self._tracker is not None:
                return self._tracker
            try:
                _report(progress, f"Loading SAM3 tracker model {self.settings.effective_tracker_model_id} ...")
                from transformers import Sam3TrackerModel, Sam3TrackerProcessor

                kwargs = dict(
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=self.settings.low_cpu_mem_usage,
                    use_safetensors=self.settings.use_safetensors,
                    local_files_only=self.settings.local_files_only,
                )
                model = Sam3TrackerModel.from_pretrained(self.settings.effective_tracker_model_id, **kwargs).to(
                    self.device
                )
                model.eval()
                processor = Sam3TrackerProcessor.from_pretrained(
                    self.settings.effective_tracker_model_id, local_files_only=self.settings.local_files_only
                )
                self._tracker = (model, processor)
                _report(progress, "SAM3 tracker ready.")
                return self._tracker
            except Exception as exc:  # noqa: BLE001
                self.last_load_error = f"Failed to load SAM3 tracker model: {exc}"
                raise SegmentationError(self.last_load_error) from exc

    # ------------------------------------------------------------------ inference
    def _run_pcs(
        self,
        image: Image.Image,
        prompt: PromptSet,
        score_threshold: float,
        mask_threshold: float,
        progress: ProgressCallback = None,
    ) -> list[MaskInstance]:
        model, processor = self.ensure_pcs(progress)
        if not prompt.has_text and not prompt.has_boxes:
            raise SegmentationError("Text or a box prompt is required for SAM3 PCS.")
        boxes: list[list[float]] | None = [list(b) for b in prompt.boxes_positive + prompt.boxes_negative] or None
        labels = [1] * len(prompt.boxes_positive) + [0] * len(prompt.boxes_negative) or None
        concept = "text" if prompt.has_text else "visual"
        _report(
            progress,
            f"Running SAM3 PCS ({concept} prompt, {len(boxes) if boxes else 0} box(es)) ...",
        )
        inputs = processor(
            images=image,
            text=prompt.text,  # None -> processor uses 'visual' when boxes are present
            input_boxes=boxes,
            input_boxes_labels=labels,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        import torch

        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )
        if not results:
            return []
        data = results[0]
        if data is None or not len(data["masks"]):
            return []
        instances: list[MaskInstance] = []
        masks = data["masks"].cpu().numpy().astype(bool)
        boxes_np = data["boxes"].cpu().numpy() if "boxes" in data else None
        scores = data["scores"].cpu().numpy() if "scores" in data else np.full((len(masks),), np.nan)
        source = "text" if prompt.has_text else "box"
        if prompt.has_text and prompt.has_boxes:
            source = "text+box"
        for i, mask in enumerate(masks):
            y0, x0 = np.nonzero(mask)
            if y0.size == 0:
                continue
            if boxes_np is not None:
                x1, y1, x2, y2 = boxes_np[i].tolist()
                box = (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                )
            else:
                box = (int(x0.min()), int(y0.min()), int(x0.max()), int(y0.max()))
            score = float(scores[i]) if not np.isnan(float(scores[i])) else None
            instances.append(MaskInstance(mask=mask, score=score, box=box, source=source, label=str(i + 1)))
        # Include semantic segmentation when a text prompt was used.
        if prompt.has_text and hasattr(outputs, "semantic_seg") and outputs.semantic_seg is not None:
            try:
                sem = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=inputs["original_sizes"].tolist(), threshold=mask_threshold
                )[0]
                self._pending_semantic = np.asarray(sem.cpu().numpy() if hasattr(sem, "cpu") else sem).astype(bool)
            except Exception:  # noqa: BLE001
                self._pending_semantic = None
        else:
            self._pending_semantic = None
        return instances

    def _run_tracker(
        self,
        image: Image.Image,
        prompt: PromptSet,
        mask_threshold: float,
        progress: ProgressCallback = None,
    ) -> list[MaskInstance]:
        if not prompt.points_positive:
            raise SegmentationError("At least one positive point is required for the SAM3 tracker.")
        model, processor = self.ensure_tracker(progress)
        clusters = cluster_positive_points(prompt.points_positive, self.settings.cluster_distance_px)
        centres = [
            (
                sum(p[0] for p in cluster) / len(cluster),
                sum(p[1] for p in cluster) / len(cluster),
            )
            for cluster in clusters
        ]
        points_by_obj = [list(c) for c in clusters]
        labels_by_obj = [[1] * len(c) for c in clusters]
        for neg in prompt.points_negative:
            idx = min(
                range(len(clusters)),
                key=lambda i: (neg[0] - centres[i][0]) ** 2 + (neg[1] - centres[i][1]) ** 2,
            )
            points_by_obj[idx].append(neg)
            labels_by_obj[idx].append(0)
        input_points = [[[p for p in pts] for pts in points_by_obj]]
        input_labels = [[list(lbl) for lbl in labels_by_obj]]
        _report(
            progress,
            f"Running SAM3 tracker for {len(clusters)} object(s) (points: {sum(len(p) for p in points_by_obj)}) ...",
        )
        inputs = processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        import torch

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        processed = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            binarize=True,
            mask_threshold=mask_threshold,
        )
        if not processed:
            return []
        arr = processed[0].cpu().numpy().astype(bool)
        # Possible shapes after upscaling:
        #   (1, objects, masks, H, W)  [batch, object, mask candidates]
        #   (objects, masks, H, W)
        #   (objects, H, W)
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim == 4:
            arr = arr[:, 0, ...]  # keep the best 'multimask' candidate per object
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise SegmentationError(f"Unexpected tracker mask shape: {arr.shape}")
        instances: list[MaskInstance] = []
        for obj_idx, mask in enumerate(arr):
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            instances.append(MaskInstance(mask=mask, score=None, box=box, source="point", label=f"point {obj_idx + 1}"))
        self._pending_semantic = None
        return instances

    def segment(
        self,
        image: Image.Image,
        prompt: PromptSet,
        mode: ModeChoice | str = ModeChoice.AUTO,
        score_threshold: float | None = None,
        mask_threshold: float | None = None,
        max_masks: int | None = None,
        progress: ProgressCallback = None,
    ) -> SegmentationResult:
        """Run segmentation for the requested mode."""
        started = time.perf_counter()
        image = image.convert("RGB")
        size = image.size
        mode = ModeChoice(mode)
        score_threshold = (
            self.settings.score_threshold if score_threshold is None else max(0.0, min(1.0, score_threshold))
        )
        mask_threshold = self.settings.mask_threshold if mask_threshold is None else max(0.0, min(1.0, mask_threshold))
        max_masks = self.settings.max_masks if max_masks is None else max(1, int(max_masks))
        prompt = prompt or PromptSet()
        warnings = list(prompt.warnings)

        use_text = prompt.has_text
        use_boxes = prompt.has_boxes
        use_points = prompt.has_points

        if mode == ModeChoice.AUTO:
            mode = ModeChoice.TEXT if (use_text or use_boxes) else ModeChoice.POINT if use_points else ModeChoice.TEXT

        instances: list[MaskInstance] = []
        self._pending_semantic = None
        validate_prompt(mode, prompt)

        if mode == ModeChoice.TEXT:
            if use_points:
                warnings.append("Point prompts are ignored in Text mode; use Mixed mode to combine.")
            instances += self._run_pcs(image, prompt, score_threshold, mask_threshold, progress)
        elif mode == ModeChoice.BOX:
            if use_points:
                warnings.append("Point prompts are ignored in Box mode; use Mixed mode to combine.")
            if use_text:
                text_prompt = PromptSet(
                    text=prompt.text, boxes_positive=prompt.boxes_positive, boxes_negative=prompt.boxes_negative
                )
            else:
                text_prompt = PromptSet(boxes_positive=prompt.boxes_positive, boxes_negative=prompt.boxes_negative)
            instances += self._run_pcs(image, text_prompt, score_threshold, mask_threshold, progress)
        elif mode == ModeChoice.POINT:
            if use_text:
                warnings.append("Text prompts are ignored in Point mode; use Mixed mode to combine.")
            if use_boxes:
                warnings.append("Box prompts are ignored in Point mode; use Mixed mode to combine.")
            instances += self._run_tracker(image, prompt, mask_threshold, progress)
        elif mode == ModeChoice.MIXED:
            pcs_needed = use_text or use_boxes
            tracker_needed = use_points
            if pcs_needed:
                mixed_pcs = PromptSet(
                    text=prompt.text,
                    boxes_positive=prompt.boxes_positive,
                    boxes_negative=prompt.boxes_negative,
                )
                # Convert negative points into small negative boxes when refining
                # a text/box concept (PCS does not accept raw points).
                if prompt.points_negative:
                    for point in prompt.points_negative:
                        mixed_pcs.boxes_negative.append(
                            negative_point_to_box(point, size, self.settings.negative_point_box_size_relative)
                        )
                instances += self._run_pcs(image, mixed_pcs, score_threshold, mask_threshold, progress)
            if tracker_needed:
                instances += self._run_tracker(image, prompt, mask_threshold, progress)
                if pcs_needed:
                    warnings.append(
                        "Negative points were also applied as small negative boxes to the text/box concept."
                    )
        else:  # pragma: no cover
            raise SegmentationError(f"Unsupported mode: {mode}")

        instances = merge_instances(instances, self.settings.iou_merge_threshold)[:max_masks]
        result = SegmentationResult(
            instances=instances,
            image_size=(size[1], size[0]),
            elapsed_seconds=time.perf_counter() - started,
            warnings=warnings,
            semantic_mask=self._pending_semantic,
        )
        logger.info(
            "Segmentation done: %d instance(s) in %.2fs (%s)", len(instances), result.elapsed_seconds, prompt.describe()
        )
        return result


class MockSegmentationEngine:
    """Deterministic fake engine used when ``SAM_MOCK=true``.

    No model is downloaded; used for UI development and tests. It mimics the
    :class:`SegmentationEngine` interface and returns synthetic masks.
    """

    device = "cpu"
    torch_dtype = None
    loaded = True
    last_load_error = None

    def __init__(self, settings: SamSettings | None = None):
        self.settings = settings or SamSettings()
        self.settings.apply_hf_environment()

    def ensure_pcs(self, progress: ProgressCallback = None) -> tuple:
        return (None, None)

    def ensure_tracker(self, progress: ProgressCallback = None) -> tuple:
        return (None, None)

    def segment(
        self,
        image: Image.Image,
        prompt: PromptSet,
        mode: ModeChoice | str = ModeChoice.AUTO,
        score_threshold: float | None = None,
        mask_threshold: float | None = None,
        max_masks: int | None = None,
        progress: ProgressCallback = None,
    ) -> SegmentationResult:
        image = image.convert("RGB")
        w, h = image.size
        mode = ModeChoice(mode)
        if mode == ModeChoice.AUTO:
            mode = (
                ModeChoice.TEXT
                if (prompt.has_text or prompt.has_boxes)
                else ModeChoice.POINT
                if prompt.has_points
                else ModeChoice.TEXT
            )
        validate_prompt(mode, prompt)
        rng = np.random.default_rng(abs(hash(prompt.describe() + str(mode))) % (2**32))
        count = 1 + (len(prompt.points_positive) if prompt.points_positive else 0) + int(bool(prompt.text))
        count = min(max(count, 2), 5)
        max_masks = self.settings.max_masks if max_masks is None else int(max_masks)
        instances: list[MaskInstance] = []
        for i in range(min(count, max_masks)):
            cx, cy = rng.integers(int(0.15 * w), int(0.85 * w)), rng.integers(int(0.15 * h), int(0.85 * h))
            rw, rh = int(rng.integers(int(0.08 * w), int(0.35 * w))), int(rng.integers(int(0.08 * h), int(0.35 * h)))
            yy, xx = np.ogrid[:h, :w]
            mask = ((xx - cx) / max(rw, 1)) ** 2 + ((yy - cy) / max(rh, 1)) ** 2 <= 1.0
            ys, xs = np.nonzero(mask)
            score = float(rng.random() * 0.4 + 0.5)
            source = "point" if prompt.points_positive and not prompt.text else "text" if prompt.text else "box"
            instances.append(
                MaskInstance(
                    mask=mask,
                    score=score,
                    box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    source=source,
                    label=f"mock {i + 1}",
                )
            )
        instances = merge_instances(instances, 0.9)[:max_masks]
        return SegmentationResult(
            instances=instances,
            image_size=(h, w),
            elapsed_seconds=0.01 * count,
            warnings=list(prompt.warnings) + ["MOCK ENGINE — results are synthetic."],
        )


def create_engine(settings: SamSettings | None = None) -> SegmentationEngine | MockSegmentationEngine:
    """Factory used across the app (CLI + UI)."""
    settings = settings or SamSettings()
    if settings.mock:
        return MockSegmentationEngine(settings)
    return SegmentationEngine(settings)


_GLOBAL_ENGINE: SegmentationEngine | MockSegmentationEngine | None = None
_GLOBAL_LOCK = threading.Lock()


def get_engine(settings: SamSettings | None = None) -> SegmentationEngine | MockSegmentationEngine:
    """Process-wide engine singleton (models are heavy)."""
    global _GLOBAL_ENGINE
    with _GLOBAL_LOCK:
        if _GLOBAL_ENGINE is None:
            _GLOBAL_ENGINE = create_engine(settings or SamSettings())
        return _GLOBAL_ENGINE

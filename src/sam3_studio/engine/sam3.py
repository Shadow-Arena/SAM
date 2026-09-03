"""SAM3 segmentation engine (real model).

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

import numpy as np
from PIL import Image

from ..config import ModeChoice, SamSettings, resolve_device, resolve_torch_dtype
from ..domain import MaskInstance, PromptSet, SegmentationResult
from ..prompts import build_pcs_prompts, build_tracker_prompts, negative_point_to_box
from .common import ProgressCallback, merge_instances, report, validate_prompt
from .errors import SegmentationError

logger = logging.getLogger(__name__)


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
                report(progress, f"Loading SAM3 PCS model {self.settings.model_id} on {self.device} ...")
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
                report(progress, f"SAM3 PCS ready on {self.device}.")
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
                report(progress, f"Loading SAM3 tracker model {self.settings.effective_tracker_model_id} ...")
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
                report(progress, "SAM3 tracker ready.")
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
        boxes, labels = build_pcs_prompts(prompt.boxes_positive, prompt.boxes_negative)
        concept = "text" if prompt.has_text else "visual"
        report(
            progress,
            f"Running SAM3 PCS ({concept} prompt, {len(boxes[0]) if boxes else 0} box(es)) ...",
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
        input_points, input_labels = build_tracker_prompts(
            prompt.points_positive,
            prompt.points_negative,
            self.settings.cluster_distance_px,
        )
        report(
            progress,
            f"Running SAM3 tracker for {len(input_points[0])} object(s) "
            f"(points: {sum(len(p) for p in input_points[0])}) ...",
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

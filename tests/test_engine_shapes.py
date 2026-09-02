"""Shape-level tests for the SAM3 engine paths using fake model/processor objects.

These verify our tensor plumbing (input construction, post-processing, mask
normalization, mixed merge) against the shape contract of
``transformers`` 5.x WITHOUT downloading multi-GB weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import BatchEncoding

from app.config import ModeChoice
from app.schemas import PromptSet
from app.segmentation import SegmentationEngine


def _fake_pcs_processor(batch_encoding, instance_shapes, semantic=None):
    """Returns fake Sam3Processor returning documented post-process shapes."""

    class FakePCSProcessor:
        def __call__(self, images=None, text=None, input_boxes=None, input_boxes_labels=None, return_tensors=None):
            data = dict(batch_encoding)
            if text is None:
                data["input_ids"] = torch.zeros(1, 32, dtype=torch.long)
                data["attention_mask"] = torch.ones(1, 32, dtype=torch.long)
            return BatchEncoding(data)

        def post_process_instance_segmentation(self, outputs, threshold=0.3, mask_threshold=0.5, target_sizes=None):
            # (num_instances, H, W)
            masks = torch.zeros(len(instance_shapes), target_sizes[0][0], target_sizes[0][1], dtype=torch.long)
            boxes, scores = [], []
            for i, (y0, y1, x0, x1) in enumerate(instance_shapes):
                masks[i, y0:y1, x0:x1] = 1
                boxes.append([float(x0), float(y0), float(x1), float(y1)])
                scores.append(0.9 - 0.1 * i)
            return [
                {
                    "masks": masks,
                    "boxes": torch.tensor(boxes),
                    "scores": torch.tensor(scores),
                }
            ]

        def post_process_semantic_segmentation(
            self, outputs, target_sizes=None, threshold=0.5, return_segmentation_scores=False
        ):
            sem = torch.zeros(1, 1, 200, 300)
            sem[0, 0, 60:140, 80:180] = 1.0
            return [sem[0, 0] > threshold]

    return FakePCSProcessor()


class FakePCSModel:
    device = torch.device("cpu")

    def __call__(self, **inputs):
        return type(
            "Out",
            (),
            {
                "pred_logits": torch.zeros(1, 3),
                "pred_boxes": torch.zeros(1, 3, 4),
                "pred_masks": torch.zeros(1, 3, 64, 64),
                "presence_logits": torch.zeros(1, 1),
                "semantic_seg": torch.zeros(1, 1, 64, 64),
            },
        )()

    def to(self, device):  # noqa: A003 - interface parity
        return self

    def eval(self):  # noqa: A003 - interface parity
        return self


def _fake_tracker(processor):
    class FakeTrackerModel:
        device = torch.device("cpu")

        def __call__(self, **inputs):
            return type("TOut", (), {"pred_masks": torch.zeros(1, 2, 1, 64, 64)})()

        def to(self, device):  # noqa: A003 - interface parity
            return self

        def eval(self):  # noqa: A003 - interface parity
            return self

    return FakeTrackerModel(), processor


def _fake_tracker_processor():
    class FakeTrackerProcessor:
        def __call__(self, images=None, input_points=None, input_labels=None, input_boxes=None, return_tensors=None):
            return BatchEncoding(
                {
                    "pixel_values": torch.zeros(1, 3, 256, 256),
                    "input_points": torch.zeros(1, 1, 1, 2),
                    "input_labels": torch.zeros(1, 1, 1, dtype=torch.long),
                    "original_sizes": torch.tensor([[200, 300]]),
                }
            )

        def post_process_masks(self, masks, original_sizes, mask_threshold=0.0, binarize=True, **kwargs):
            # (batch=1, objects=2, candidates=1, H, W)
            out = torch.zeros(1, 2, 1, 200, 300, dtype=torch.float32)
            out[0, 0, 0, 30:90, 40:120] = 1.0
            out[0, 1, 0, 120:180, 160:260] = 1.0
            return [out > mask_threshold]

    return FakeTrackerProcessor()


def test_pcs_engine_shape(monkeypatch, settings, sample_image):
    engine = SegmentationEngine(settings)
    batch = {"pixel_values": torch.zeros(1, 3, 256, 256), "original_sizes": torch.tensor([[200, 300]])}
    processor = _fake_pcs_processor(batch, [(60, 140, 80, 180), (10, 40, 20, 60)], semantic=True)
    monkeypatch.setattr("transformers.Sam3Model.from_pretrained", classmethod(lambda cls, *a, **k: FakePCSModel()))
    monkeypatch.setattr("transformers.Sam3Processor.from_pretrained", classmethod(lambda cls, *a, **k: processor))
    result = engine.segment(sample_image, PromptSet(text="car"), ModeChoice.TEXT)
    assert len(result.instances) == 2
    inst = result.instances[0]
    assert inst.mask.shape == (200, 300)
    assert inst.score == pytest.approx(0.9)
    assert inst.box == (80, 60, 180, 140)
    assert result.semantic_mask is not None and result.semantic_mask.shape == (200, 300)


def test_pcs_engine_downsample_and_negative_boxes(monkeypatch, settings, sample_image):
    engine = SegmentationEngine(settings)
    batch = {"pixel_values": torch.zeros(1, 3, 256, 256), "original_sizes": torch.tensor([[200, 300]])}
    processor = _fake_pcs_processor(batch, [(50, 120, 100, 200)])
    monkeypatch.setattr("transformers.Sam3Model.from_pretrained", classmethod(lambda cls, *a, **k: FakePCSModel()))
    monkeypatch.setattr("transformers.Sam3Processor.from_pretrained", classmethod(lambda cls, *a, **k: processor))
    prompt = PromptSet(text="handle", boxes_negative=[(90, 160, 200, 190)])
    result = engine.segment(sample_image, prompt, ModeChoice.TEXT)
    assert len(result.instances) == 1


def test_tracker_engine_shape(monkeypatch, settings, sample_image):
    engine = SegmentationEngine(settings)
    tracker_proc = _fake_tracker_processor()
    tracker_model, tracker_proc = _fake_tracker(tracker_proc)
    monkeypatch.setattr(
        "transformers.Sam3TrackerModel.from_pretrained", classmethod(lambda cls, *a, **k: tracker_model)
    )
    monkeypatch.setattr(
        "transformers.Sam3TrackerProcessor.from_pretrained", classmethod(lambda cls, *a, **k: tracker_proc)
    )
    prompt = PromptSet(points_positive=[(80, 60), (200, 150)], points_negative=[(50, 50)])
    result = engine.segment(sample_image, prompt, ModeChoice.POINT)
    assert len(result.instances) == 2
    for inst in result.instances:
        assert inst.mask.shape == (200, 300)
        assert inst.mask.dtype == np.bool_


def test_mixed_merge(monkeypatch, settings, sample_image):
    engine = SegmentationEngine(settings)
    batch = {"pixel_values": torch.zeros(1, 3, 256, 256), "original_sizes": torch.tensor([[200, 300]])}
    pcs_proc = _fake_pcs_processor(batch, [(60, 140, 80, 180)])
    tracker_proc = _fake_tracker_processor()
    tracker_model, tracker_proc = _fake_tracker(tracker_proc)

    monkeypatch.setattr("transformers.Sam3Model.from_pretrained", classmethod(lambda cls, *a, **k: FakePCSModel()))
    monkeypatch.setattr("transformers.Sam3Processor.from_pretrained", classmethod(lambda cls, *a, **k: pcs_proc))
    monkeypatch.setattr(
        "transformers.Sam3TrackerModel.from_pretrained", classmethod(lambda cls, *a, **k: tracker_model)
    )
    monkeypatch.setattr(
        "transformers.Sam3TrackerProcessor.from_pretrained", classmethod(lambda cls, *a, **k: tracker_proc)
    )

    prompt = PromptSet(text="car", points_positive=[(100, 100)])
    # PCS 1 instance + tracker 2 instances with no overlaps (default 0.7 merge keeps all)
    result = engine.segment(sample_image, prompt, ModeChoice.MIXED)
    assert len(result.instances) == 3
    assert {i.source for i in result.instances} >= {"text", "point"}

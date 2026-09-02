from __future__ import annotations

from app.config import DeviceChoice, DTypeChoice, SamSettings, resolve_device, resolve_torch_dtype


def test_defaults():
    s = SamSettings(_env_file=None)
    assert s.model_id == "facebook/sam3"
    assert s.effective_tracker_model_id == "facebook/sam3"
    assert s.score_threshold == 0.30
    assert s.port == 7860


def test_tracker_fallback():
    s = SamSettings(tracker_model_id="facebook/sam3-large", _env_file=None)
    assert s.effective_tracker_model_id == "facebook/sam3-large"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SAM_MODEL_ID", "facebook/sam3-base")
    monkeypatch.setenv("SAM_PORT", "8000")
    monkeypatch.setenv("SAM_SCORE_THRESHOLD", "0.9")
    s = SamSettings(_env_file=None)
    assert s.model_id == "facebook/sam3-base"
    assert s.port == 8000
    assert abs(s.score_threshold - 0.9) < 1e-9


def test_thresholds_clamped():
    s = SamSettings(score_threshold=2.0, mask_threshold=-1.0, _env_file=None)
    assert s.score_threshold == 1.0
    assert s.mask_threshold == 0.0


def test_auth_tuple():
    assert SamSettings(gradio_auth="user:secret", _env_file=None).auth_tuple == ("user", "secret")
    assert SamSettings(gradio_auth="nocolon", _env_file=None).auth_tuple is None
    assert SamSettings(_env_file=None).auth_tuple is None


def test_model_dump_masks_token():
    s = SamSettings(hf_token="hf_super_secret", _env_file=None)
    assert s.model_dump_safe()["hf_token"] == "***"


def test_without_env_file():
    # Ensure .env from repo root is not accidentally loaded in unit tests.
    s = SamSettings(_env_file=None)
    assert s.host == "0.0.0.0"


def test_resolve_device_types():
    assert resolve_device(DeviceChoice.CPU) == "cpu"
    assert resolve_device(DeviceChoice.AUTO) in {"cpu", "cuda", "mps", "xpu"}


def test_resolve_dtype_auto_cpu():
    import torch

    assert resolve_torch_dtype(DTypeChoice.AUTO, "cpu") == torch.float32
    assert resolve_torch_dtype(DTypeChoice.BFLOAT16, "cpu") == torch.bfloat16

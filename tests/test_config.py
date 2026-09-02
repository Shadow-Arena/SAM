from __future__ import annotations

import os

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


def test_apply_hf_environment_sets_env(monkeypatch):
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_test_token")
    s = SamSettings(_env_file=None)
    s.apply_hf_environment()
    assert os.environ.get("HF_TOKEN") == "hf_test_token"


def test_apply_hf_environment_logs_in(monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "login", lambda token=None, **kw: calls.append((token, kw)))
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("SAM_HF_LOGIN", "true")
    s = SamSettings(_env_file=None)
    s.apply_hf_environment()
    assert calls == [("hf_test_token", {"add_to_git_credential": False})]


def test_apply_hf_environment_login_failure_does_not_raise(monkeypatch):
    import huggingface_hub

    def _offline_login(token=None, **kw):
        raise RuntimeError("offline")

    monkeypatch.setattr(huggingface_hub, "login", _offline_login)
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_test_token")
    s = SamSettings(_env_file=None)
    s.apply_hf_environment()  # must not raise
    assert os.environ.get("HF_TOKEN") == "hf_test_token"


def test_apply_hf_environment_login_disabled(monkeypatch):
    import huggingface_hub

    called = []
    monkeypatch.setattr(huggingface_hub, "login", lambda token=None, **kw: called.append(token))
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_test_token")
    s = SamSettings(hf_login=False, _env_file=None)
    s.apply_hf_environment()
    assert called == []
    assert os.environ.get("HF_TOKEN") == "hf_test_token"


def test_resolve_device_types():
    assert resolve_device(DeviceChoice.CPU) == "cpu"
    assert resolve_device(DeviceChoice.AUTO) in {"cpu", "cuda", "mps", "xpu"}


def test_resolve_dtype_auto_cpu():
    import torch

    assert resolve_torch_dtype(DTypeChoice.AUTO, "cpu") == torch.float32
    assert resolve_torch_dtype(DTypeChoice.BFLOAT16, "cpu") == torch.bfloat16

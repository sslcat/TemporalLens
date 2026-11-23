import numpy as np
import torch
from fastapi.testclient import TestClient
from temporallens import api
from temporallens.gesture import GestureHead, detect, padded
from temporallens.media import decode


def test_gesture_decoder_abstains_and_filters_short_runs():
    probs = np.full((9, 5), 0.2)
    probs[1:4] = [0.04, 0.80, 0.06, 0.06, 0.04]
    probs[6] = [0, 0, 1, 0, 0]
    assert detect(probs, 0.65, 2) == [(1, 1, 4)]
    assert detect(probs, 0.9, 2) == []


def test_head_context_and_pool_order():
    torch.manual_seed(17)
    x = torch.rand(15, 1536)
    order = torch.randperm(15)
    center = GestureHead("pooled").eval()
    torch.testing.assert_close(
        center(x[None]), center(x[order][None]), atol=1e-5, rtol=1e-5
    )
    for kind in ("frame", "pooled", "temporal"):
        head = GestureHead(kind).eval()
        assert head(padded(x)).shape == (1, 15, 5)
        if kind == "frame":
            torch.testing.assert_close(
                head(padded(x))[0, order], head(padded(x[order]))[0]
            )


def test_gesture_sampling_preserves_shape():
    frames, _, duration = decode(api.WEB / "example.mp4", gesture_frames=True)
    assert len(frames) == 60 and duration == 8
    assert frames[0].shape == (213, 320, 3) and frames[0].dtype == np.uint8


def test_mode_validation_and_optional_install(monkeypatch):
    monkeypatch.setenv("TEMPORALLENS_DEVICE", "cpu")
    monkeypatch.setattr(api, "gesture_ready", lambda: False)
    with TestClient(api.app) as client:
        assert client.get("/api/status").json()["gesture_ready"] is False
        assert client.post("/api/analyze?mode=gesture").status_code == 503
        assert client.post("/api/analyze?mode=unknown").status_code == 422

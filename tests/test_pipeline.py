import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from temporallens import api
from temporallens.data import render
from temporallens.evaluate import score
from temporallens.media import decode, embedding
from temporallens.model import MotionHead, segments

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "web/example.mp4"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEMPORALLENS_DEVICE", "cpu")
    with TestClient(api.app) as http:
        yield http


def test_split_integrity():
    rows = [
        json.loads(line)
        for line in (ROOT / "artifacts/manifest.jsonl").read_text().splitlines()
    ]
    assert (
        len(rows)
        == len({r["scene"] for r in rows})
        == len({r["sha256"] for r in rows})
        == 396
    )
    assert [sum(r["split"] == s for r in rows) for s in ("train", "val", "test")] == [
        240,
        60,
        96,
    ]
    assert all(
        len(r["events"]) == 2 and r["events"][0]["end"] < r["events"][1]["start"]
        for r in rows
    )
    assert hashlib.sha256(EXAMPLE.read_bytes()).hexdigest() in {
        r["sha256"] for r in rows if r["split"] == "test"
    }


def test_encoded_frames_match_motion(tmp_path):
    path = tmp_path / "video.mp4"
    labels = render(path, 6433, (1, 4))
    x, valid, duration = decode(path)
    assert len(x) == len(labels) == 80 and duration == 8 and valid.all()
    assert np.diff(x[:, 0])[labels[1:] == 1].mean() < 0
    assert np.diff(x[:, 1])[labels[1:] == 4].mean() > 0


def test_embeddings_and_runs():
    assert not embedding(np.zeros((64, 96, 3), dtype="uint8"))[1]
    assert segments([1, 1, 1, 0, 3, 3, 4, 4, 4]) == [(1, 0, 3), (4, 6, 9)]


@pytest.mark.parametrize("temporal", [True, False])
def test_gradient_and_forward(temporal):
    model = MotionHead(temporal)
    x = torch.rand(3, 10, 2)
    y = model(x)
    assert y.shape == (3, 10, 5)
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    if not temporal:
        assert torch.allclose(model(x.flip(1)).flip(1), y)


def test_metrics_penalize_extra_and_missing():
    y = np.array([1] * 4 + [0] * 4 + [3] * 4)
    truth = np.eye(5)[y]
    assert score(truth, y)["iou"] == 1 and score(truth, y)["exact"]
    bad = truth.copy()
    bad[4:8] = np.eye(5)[2]
    assert not score(bad, y)["exact"]
    bad[8:] = np.eye(5)[0]
    assert score(bad, y)["iou"] == 0.5


def test_status_and_upload(client):
    assert client.get("/api/status").json()["device"] == "cpu"
    resp = client.post(
        "/api/analyze",
        content=EXAMPLE.read_bytes(),
        headers={"Content-Type": "video/mp4"},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["supported"] and result["duration"] == 8
    assert all(
        0 <= e["start"] < e["end"] <= 8 and 0 <= e["score"] <= 1
        for e in result["events"]
    )
    assert len(result["events"]) == 2


@pytest.mark.parametrize(
    "content,mime,code",
    [
        (b"garbage", "video/mp4", 422),
        (b"", "video/webm", 422),
        (b"hello", "text/plain", 415),
    ],
)
def test_reject(client, content, mime, code):
    assert (
        client.post(
            "/api/analyze", content=content, headers={"Content-Type": mime}
        ).status_code
        == code
    )


def test_size_limit(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_BYTES", 8)
    for data in [b"123456789", iter([b"12345", b"6789"])]:
        assert (
            client.post(
                "/api/analyze", content=data, headers={"Content-Type": "video/mp4"}
            ).status_code
            == 413
        )


def test_other_host(client):
    assert client.get("/", headers={"Host": "untrusted.example"}).status_code == 400


def test_short_and_unsupported(tmp_path, client):
    import av

    for count in (20, 60):
        path = tmp_path / f"{count}.mp4"
        with av.open(str(path), "w") as movie:
            stream = movie.add_stream("libx264", rate=10)
            stream.width, stream.height = 96, 64
            for _ in range(count):
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((64, 96, 3), dtype="uint8"), format="rgb24"
                )
                for packet in stream.encode(frame):
                    movie.mux(packet)
            for packet in stream.encode():
                movie.mux(packet)
        r = client.post(
            "/api/analyze",
            content=path.read_bytes(),
            headers={"Content-Type": "video/mp4"},
        )
        if count == 20:
            assert r.status_code == 422
        else:
            assert (
                r.status_code == 200
                and r.json()["events"] == []
                and not r.json()["supported"]
            )

from fractions import Fraction

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from temporallens import api
from temporallens.media import VideoError, decode


def make_movie(path, seconds, fps=10):
    with av.open(str(path), "w") as movie:
        stream = movie.add_stream("libx264", rate=fps)
        stream.width, stream.height = 96, 64
        for t in range(seconds * fps):
            rgb = np.zeros((64, 96, 3), dtype=np.uint8)
            x = 10 if t < fps else 60
            rgb[20:30, x : x + 10] = [40, 220, 210]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = t, Fraction(1, fps)
            for packet in stream.encode(frame):
                movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)


@pytest.mark.parametrize("fps", [5, 24])
def test_time_sampling(tmp_path, fps):
    p = tmp_path / "rate.mp4"
    make_movie(p, 5, fps)
    x, valid, duration = decode(p)
    assert valid.all() and len(x) == 50 and duration == pytest.approx(5)
    assert np.all(x[:10, 0] < 0.2) and np.all(x[10:, 0] > 0.6)


def test_duration_ceiling(tmp_path):
    p = tmp_path / "long.mp4"
    make_movie(p, 16)
    with pytest.raises(VideoError, match="5–15"):
        decode(p)


def test_temp_cleanup_and_busy_response(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("TEMPORALLENS_DEVICE", "cpu")
    with TestClient(api.app) as client:
        for body, expected in [
            (b"bad", 422),
            ((api.WEB / "example.mp4").read_bytes(), 200),
        ]:
            r = client.post(
                "/api/analyze", content=body, headers={"Content-Type": "video/mp4"}
            )
            assert r.status_code == expected
            assert not list(tmp_path.glob("temporallens-*"))
        api.app.state.busy = True
        assert (
            client.post(
                "/api/analyze", content=b"x", headers={"Content-Type": "video/mp4"}
            ).status_code
            == 429
        )

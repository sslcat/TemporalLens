import av
import numpy as np
from fastapi.testclient import TestClient

from temporallens.api import app


def test_webm_stationary_marker(tmp_path, monkeypatch):
    path = tmp_path / "still.webm"
    with av.open(str(path), "w") as movie:
        stream = movie.add_stream("libvpx", rate=10)
        stream.width, stream.height = 96, 64
        pixels = np.zeros((64, 96, 3), dtype=np.uint8)
        pixels[20:30, 50:60] = [40, 220, 210]
        for _ in range(60):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)
    monkeypatch.setenv("TEMPORALLENS_DEVICE", "cpu")
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze",
            content=path.read_bytes(),
            headers={"Content-Type": "video/webm"},
        )
        assert response.status_code == 200
        result = response.json()
        assert (
            result["supported"] and result["duration"] == 6 and result["events"] == []
        )

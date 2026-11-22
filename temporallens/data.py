"""Render independent scenes, then decode the encoded pixels for training."""

import hashlib
import itertools
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from .media import HZ, LABELS, decode

SPLITS = {"train": (10000, 240), "val": (20000, 60), "test": (30000, 96)}
VECTORS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


def render(path, seed, pair):
    rng = np.random.default_rng(seed)
    y = np.zeros(80, dtype=np.int64)
    t = int(rng.integers(4, 15))
    delta = np.zeros((80, 2))
    for label in pair:
        length = int(rng.integers(12, 23))
        y[t : t + length] = label
        delta[t : t + length] = VECTORS[label] * rng.uniform(1.1, 2.6)
        t += length + int(rng.integers(6, 15))
    positions = np.cumsum(delta, axis=0)
    size = np.array([240, 160])
    positions += rng.uniform(15 - positions.min(0), size - 15 - positions.max(0))
    base = Image.new("RGB", tuple(size), (17, 25, int(rng.integers(30, 45))))
    draw = ImageDraw.Draw(base)
    for x in range(0, 240, 16):
        for v in range(0, 160, 16):
            draw.point((x, v), fill=(65, 82, 89))
    radius = int(rng.integers(6, 11))
    color = (
        int(rng.integers(15, 80)),
        int(rng.integers(195, 250)),
        int(rng.integers(191, 246)),
    )
    with av.open(str(path), "w") as movie:
        video = movie.add_stream("libx264", rate=HZ)
        video.width, video.height = map(int, size)
        video.pix_fmt = "yuv420p"
        video.options = {"preset": "ultrafast", "crf": "18", "threads": "1"}
        for px, py in positions.round().astype(int):
            scene = base.copy()
            ImageDraw.Draw(scene).ellipse(
                (px - radius, py - radius, px + radius, py + radius), fill=color
            )
            frame = av.VideoFrame.from_ndarray(np.asarray(scene), format="rgb24")
            for packet in video.encode(frame):
                movie.mux(packet)
        for packet in video.encode():
            movie.mux(packet)
    return y


def generate():
    from .model import segments

    work = Path("work")
    work.mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    rows = []
    pairs = list(itertools.permutations(range(1, 5), 2))
    for split, (first, count) in SPLITS.items():
        clips = work / split
        clips.mkdir(exist_ok=True)
        xs, ys = [], []
        for i in range(count):
            seed = first + i
            file = clips / f"scene-{seed}.mp4"
            y = render(file, seed, pairs[i % len(pairs)])
            x, valid, duration = decode(file)
            assert len(x) == len(y) and valid.all() and abs(duration - 8) < 1e-5
            xs.append(x)
            ys.append(y)
            rows.append(
                dict(
                    split=split,
                    scene=seed,
                    path=file.as_posix(),
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    events=[
                        dict(label=LABELS[k], start=a / HZ, end=b / HZ)
                        for k, a, b in segments(y, 1)
                    ],
                )
            )
        np.savez(work / f"{split}.npz", x=xs, y=ys)
        print(f"Decoded {count} {split} clips", flush=True)
    assert len({row["sha256"] for row in rows}) == len(rows)
    Path("artifacts/manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8"
    )


if __name__ == "__main__":
    generate()

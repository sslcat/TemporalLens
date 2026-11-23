"""Cache the selected streams; verify split integrity before fitting any head."""

from collections import defaultdict
import csv
import io
import json
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors

from .ipn_ingest import WORK, META, OUT, participants
from temporallens.gesture import FrozenEncoder

CODES = {"G05": 1, "G06": 2, "G10": 3, "G11": 4}


def flow_vector(previous, current):
    field = cv2.calcOpticalFlowFarneback(
        previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    border = np.concatenate(
        (
            field[:3].reshape(-1, 2),
            field[-3:].reshape(-1, 2),
            field[:, :3].reshape(-1, 2),
            field[:, -3:].reshape(-1, 2),
        )
    )
    field -= np.median(border, axis=0)
    u, v = field.transpose(2, 0, 1)
    magnitude = np.hypot(u, v)
    angle = ((np.arctan2(v, u) + np.pi) * 4 / np.pi).astype(int) % 8
    yy, xx = np.indices(u.shape)
    mass = magnitude.sum() + 1e-6
    dx = xx - (xx * magnitude).sum() / mass
    dy = yy - (yy * magnitude).sum() / mass
    radial = (dx * u + dy * v) / np.maximum(1, np.hypot(dx, dy))
    values = []
    for rows in np.array_split(np.arange(u.shape[0]), 3):
        for cols in np.array_split(np.arange(u.shape[1]), 4):
            cell = np.ix_(rows, cols)
            values.extend(
                [
                    u[cell].mean(),
                    v[cell].mean(),
                    radial[cell].mean(),
                    magnitude[cell].mean(),
                ]
            )
            values.extend(
                np.bincount(
                    angle[cell].ravel(), weights=magnitude[cell].ravel(), minlength=8
                )
                / len(rows)
                / len(cols)
            )
    return np.asarray(values, np.float32)


def gray(rgb):
    return cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (80, 60), interpolation=cv2.INTER_AREA
    )


def flow_sequence(images):
    return np.stack(
        [flow_vector(images[max(i - 1, 0)], im) for i, im in enumerate(images)]
    )


def checked_hashes(rows):
    owners = {}
    for r in rows:
        hashes = (WORK / (r["source"] + ".hash")).read_bytes()
        assert len(hashes) == 32 * r["frames"]
        for i in range(0, len(hashes), 32):
            h = hashes[i : i + 32]
            old = owners.setdefault(h, r["split"])
            if old != r["split"]:
                raise ValueError("Identical JPEG across splits: " + r["source"])


def validated_annotations(rows):
    with (META.parent / "classes.txt").open() as f:
        ids = {r["label"]: int(r["id"]) for r in csv.DictReader(f)}
    expected = {"G05": 8, "G06": 9, "G10": 13, "G11": 14}
    if any(ids.get(label) != value for label, value in expected.items()):
        raise ValueError("Official target ID mapping differs from the pilot")
    owners_by_field = {field: {} for field in ("participant", "setup")}
    if len({r["source"] for r in rows}) != len(rows):
        raise ValueError("Duplicate recording in pilot manifest")
    for r in rows:
        if participants(r["source"]) != r["participant"]:
            raise ValueError("Participant key mismatch: " + r["source"])
        for field, owners in owners_by_field.items():
            if owners.setdefault(r[field], r["split"]) != r["split"]:
                raise ValueError(f"Shared {field} across splits: {r[field]}")
    original = defaultdict(list)
    with (META / "Annot_List.txt").open() as f:
        for r in csv.DictReader(f):
            if int(r["id"]) != ids[r["label"]]:
                raise ValueError("Mismatched label and ID: " + r["video"])
            original[r["video"]].append(r)
    return original


def main():
    cv2.setNumThreads(1)
    rows = json.loads((OUT / "split.json").read_text())
    checked_hashes(rows)
    original = validated_annotations(rows)
    model = FrozenEncoder("cuda")
    cache = WORK / "cache"
    cache.mkdir(exist_ok=True)
    signatures, owners = [], []
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        source = row["source"]
        n = int(original[source][-1]["t_end"])
        ids = np.arange(1, n + 1, 4)
        dense = np.zeros(n, np.int64)
        for r in original[source]:
            dense[int(r["t_start"]) - 1 : int(r["t_end"])] = CODES.get(r["label"], 0)
        with zipfile.ZipFile(WORK / (source + ".zip")) as z:
            images = [
                np.asarray(Image.open(io.BytesIO(z.read(f"{i}.jpg"))).convert("RGB"))
                for i in ids
            ]
        gray_images = np.stack([gray(im) for im in images])
        features = model(images).astype(np.float16)
        motion = flow_sequence(gray_images)
        np.savez(
            cache / (source + ".npz"),
            x=features,
            motion=motion,
            y=dense[ids - 1],
            dense=dense,
            gray=gray_images,
        )
        # Overlapping 1.07-s signatures, stride four ticks. Check all source pairs across splits.
        thumbs = (
            np.stack(
                [
                    cv2.resize(im, (8, 6), interpolation=cv2.INTER_AREA)
                    for im in gray_images
                ]
            ).astype(np.float32)
            / 255
        )
        for i in range(0, len(thumbs) - 7, 4):
            signatures.append(thumbs[i : i + 8 : 2].flatten())
            owners.append((row["split"], source, i))
        print(source, len(ids), flush=True)
    signatures = np.stack(signatures)
    nearest = []
    for split in ("train", "val", "test"):
        a = np.array([i for i, r in enumerate(owners) if r[0] == split])
        b = np.array([i for i, r in enumerate(owners) if r[0] != split])
        dist, idx = (
            NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1)
            .fit(signatures[b])
            .kneighbors(signatures[a])
        )
        j = dist[:, 0].argmin()
        nearest.append(
            dict(
                left=owners[a[j]],
                right=owners[b[idx[j, 0]]],
                rms=float(dist[j, 0] / np.sqrt(signatures.shape[1])),
            )
        )
    audit = dict(
        seconds=time.perf_counter() - t0,
        peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        cross_split_identical_jpegs=0,
        closest_low_resolution_segments=nearest,
    )
    (OUT / "feature_audit.json").write_text(json.dumps(audit, indent=2))
    if min(r["rms"] for r in nearest) < 0.025:
        raise ValueError("Review possible near-duplicate footage before training")


if __name__ == "__main__":
    main()

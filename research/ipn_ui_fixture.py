"""Make private validation uploads for QA against the running local gesture app."""

import argparse
import csv
import io
import json
import statistics
from time import perf_counter
import urllib.request
import zipfile

import av
import numpy as np
from PIL import Image
from .ipn_ingest import META, OUT, WORK
from temporallens.gesture import LABELS
from .ipn_features import CODES


def encode(path, images):
    with av.open(str(path), "w") as movie:
        stream = movie.add_stream("libx264", rate=30)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for im in images:
            for _ in range(4):
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(im, format="rgb24")
                ):
                    movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)


def post(path, base):
    req = urllib.request.Request(
        base + "/api/analyze?mode=gesture",
        data=path.read_bytes(),
        headers={"Content-Type": "video/mp4"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18731")
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Time ten warm calls on the existing local QA clip",
    )
    args = parser.parse_args()
    base = args.base
    ui = WORK / "ui"
    ui.mkdir(exist_ok=True)
    if args.latency:
        clip = ui / "gesture.mp4"
        warm = post(clip, base)
        samples = []
        for _ in range(10):
            start = perf_counter()
            response = post(clip, base)
            samples.append(1000 * (perf_counter() - start))
            assert response == warm
        report = dict(
            device=warm["device"],
            upload_seconds=warm["duration"],
            warmups=1,
            requests=len(samples),
            latency_ms=samples,
            median_ms=statistics.median(samples),
            method="Sequential loopback POST of the same validation clip to the loaded model; includes transfer, decode, inference and JSON.",
        )
        (OUT / "app_latency.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report)
        return
    labels = list(csv.DictReader((META / "Annot_List.txt").read_text().splitlines()))
    sources = [
        r for r in json.loads((OUT / "split.json").read_text()) if r["split"] == "val"
    ]
    empty = ui / "still.mp4"
    encode(empty, [np.zeros((240, 320, 3), np.uint8)] * 45)
    assert post(empty, base)["events"] == []
    attempts = 0
    for row in sources:
        with zipfile.ZipFile(WORK / (row["source"] + ".zip")) as z:
            for ann in labels:
                if ann["video"] != row["source"] or ann["label"] not in CODES:
                    continue
                first = max(
                    0,
                    min(
                        row["frames"] // 4 - 75,
                        (int(ann["t_start"]) + int(ann["t_end"])) // 8 - 37,
                    ),
                )
                ims = [
                    np.asarray(
                        Image.open(io.BytesIO(z.read(f"{4 * i + 1}.jpg"))).convert(
                            "RGB"
                        )
                    )
                    for i in range(first, first + 75)
                ]
                path = ui / "gesture.mp4"
                encode(path, ims)
                response = post(path, base)
                attempts += 1
                expected = LABELS[CODES[ann["label"]]]
                if any(
                    e["label"] == expected and e["start"] > 0
                    for e in response["events"]
                ):
                    record = dict(
                        attempts=attempts,
                        source=row["source"],
                        start_frame=4 * first + 1,
                        expected=expected,
                        response=response,
                    )
                    (ui / "qa.json").write_text(json.dumps(record, indent=2))
                    print(record)
                    return
    raise RuntimeError("No validation clip yielded a seekable expected detection")


if __name__ == "__main__":
    main()

"""Build a private validation-only seekable fixture for the two-stage UI."""

import csv
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from .ipn_ingest import META, OUT, WORK
from .ipn_features import CODES
from .ipn_ui_fixture import encode
from temporallens.gesture import LABELS

UI = Path("work/two-stage/ui")
URL = "http://127.0.0.1:18731/api/analyze?mode=two-stage"


def call(path):
    req = urllib.request.Request(
        URL, data=path.read_bytes(), headers={"Content-Type": "video/mp4"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    UI.mkdir(parents=True, exist_ok=True)
    path = UI / "gesture.mp4"
    annotations = list(
        csv.DictReader((META / "Annot_List.txt").read_text().splitlines())
    )
    rows = json.loads((OUT / "split.json").read_text())
    attempts = 0
    for row in rows:
        if row["split"] != "val":
            continue
        with zipfile.ZipFile(WORK / (row["source"] + ".zip")) as z:
            for ann in annotations:
                if ann["video"] != row["source"] or ann["label"] not in CODES:
                    continue
                a, b = int(ann["t_start"]) - 1, int(ann["t_end"])
                first = max(0, min((row["frames"] // 4) - 75, (a + b) // 8 - 37))
                images = [
                    np.asarray(
                        Image.open(io.BytesIO(z.read(f"{4 * i + 1}.jpg"))).convert(
                            "RGB"
                        )
                    )
                    for i in range(first, first + 75)
                ]
                encode(path, images)
                result = call(path)
                attempts += 1
                left, right = (a - first * 4) / 30, (b - first * 4) / 30
                expected = LABELS[CODES[ann["label"]]]
                if any(
                    e["label"] == expected
                    and e["start"] > 0
                    and min(e["end"], right) > max(e["start"], left)
                    for e in result["events"]
                ):
                    report = dict(
                        source=row["source"],
                        attempts=attempts,
                        expected=expected,
                        annotation=[left, right],
                        response=result,
                    )
                    (UI / "qa.json").write_text(json.dumps(report, indent=2))
                    print(report)
                    return
    raise RuntimeError(
        "No overlapping validation prediction was found for UI verification"
    )


if __name__ == "__main__":
    main()

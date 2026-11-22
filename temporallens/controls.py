"""Post-hoc diagnostics with frozen checkpoints and the original held-out seeds."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import render
from .evaluate import matrix_metrics, score
from .media import LABELS, decode
from .model import MotionHead, segments


def velocity_rule(x):
    # Convert normalized observations back to native pixels. One-sided velocity.
    displacement = np.diff(x, axis=0, prepend=x[:1]) * [240, 160]
    dx, dy = displacement.T
    major = np.maximum(abs(dx), abs(dy))
    ids = np.where(abs(dx) >= abs(dy), np.where(dx < 0, 1, 2), np.where(dy < 0, 3, 4))
    return np.eye(5)[np.where(major >= 0.5, ids, 0)]


def main():
    torch.set_num_threads(1)
    net = MotionHead(True).eval()
    net.load_state_dict(
        torch.load("artifacts/temporal.pt", weights_only=True, map_location="cpu")
    )

    def infer(x):
        with torch.inference_mode():
            return net(torch.tensor(x)[None])[0].numpy()

    scenes = [
        r
        for line in Path("artifacts/manifest.jsonl").read_text().splitlines()
        if (r := json.loads(line))["split"] == "test"
    ]
    rule, replay, cut_results, extra_at_seam = [], [], [], []
    first_failure = None
    work = Path("work/controls")
    work.mkdir(exist_ok=True)
    with np.load("work/test.npz") as data:
        for row, x, y in zip(scenes, data["x"], data["y"]):
            rule.append(score(velocity_rule(x), y))
            (one, _, end), (two, start, _) = segments(y, 1)
            path = work / f"replayed-{row['scene']}.mp4"
            target = render(path, row["scene"], (two, one))
            obs, valid, _ = decode(path)
            assert valid.all()
            replay.append(score(infer(obs), target))
            cut = (end + start) // 2
            altered = np.roll(x, -cut, axis=0)
            truth = np.roll(y, -cut)
            result = score(infer(altered), truth)
            cut_results.append(result)
            seam = len(y) - cut
            seam_event = any(a <= seam < b for _, a, b in result["predicted"])
            extra_at_seam.append(seam_event and not result["exact"])
            if first_failure is None and not result["exact"]:
                first_failure = (row["scene"], altered, truth, result, seam)

    def aggregate(rows):
        acc, macro, _ = matrix_metrics(sum(r["cm"] for r in rows))
        return dict(
            accuracy=acc,
            macro_f1=macro,
            exact=np.mean([r["exact"] for r in rows]),
            iou=np.mean([r["iou"] for r in rows]),
        )

    scene, altered, truth, result, seam = first_failure
    report = dict(
        velocity_rule=aggregate(rule),
        continuous_replay=aggregate(replay),
        spliced=aggregate(cut_results),
        failed_with_prediction_at_seam=sum(extra_at_seam),
        example=dict(
            scene=scene,
            seam_seconds=seam / 10,
            truth=segments(truth, 1),
            predicted=result["predicted"],
        ),
    )
    Path("artifacts/controls.json").write_text(json.dumps(report, indent=2))
    fig, (ax, timeline) = plt.subplots(
        2, 1, figsize=(9, 5), sharex=True, layout="constrained"
    )
    ax.plot(np.arange(len(truth)) / 10, altered[:, 0], label="x position")
    ax.plot(np.arange(len(truth)) / 10, altered[:, 1], label="y position")
    ax.set(
        ylabel="Image fraction",
        title=f"Spliced scene {scene}   /   First incorrect held-out sequence",
    )
    ax.legend()
    for i, runs in enumerate([segments(truth, 1), result["predicted"]]):
        for k, a, b in runs:
            timeline.barh(
                i,
                (b - a) / 10,
                left=a / 10,
                color=["", "#85b1e3", "#93dab5", "#c7a7dc", "#efb789"][k],
                height=0.5,
            )
            timeline.text((a + b) / 20, i, LABELS[k][0], ha="center", va="center")
    for panel in (ax, timeline):
        panel.axvline(seam / 10, color="#ab4d43", linestyle="--")
    timeline.set(
        xlim=(0, 8),
        yticks=[0, 1],
        yticklabels=["Truth", "Temporal"],
        xlabel="Seconds · dashed line marks the splice",
    )
    fig.savefig("artifacts/splice-failure.png", dpi=140)


if __name__ == "__main__":
    main()

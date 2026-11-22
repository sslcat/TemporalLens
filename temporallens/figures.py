"""Visualize recorded metrics and actual decoded held-out frames."""

import json
from pathlib import Path
import shutil

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .media import LABELS

OUT = Path("artifacts")
COLORS = ("#91abc7", "#167a66")


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    training, metrics, preds = (
        read("training.json"),
        read("metrics.json"),
        read("predictions.json"),
    )
    manifest = {
        r["scene"]: r
        for line in (OUT / "manifest.jsonl").read_text().splitlines()
        if (r := json.loads(line))
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for j, name in enumerate(("frame", "temporal")):
        h = np.array(training[name]["history"])
        axes[0, 0].plot(h[:, 0], h[:, 1], color=COLORS[j], label=f"{name} train")
        axes[0, 0].plot(h[:, 0], h[:, 2], "--", color=COLORS[j], label=f"{name} val")
        axes[0, 0].set(
            xlabel="Epoch",
            ylabel="Weighted cross-entropy",
            yscale="log",
            title="Learning curves",
        )
        m = metrics[name]
        ax = axes[0, 1 + j]
        cm = np.array(m["confusion"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=48)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(
                c,
                r,
                str(v),
                ha="center",
                va="center",
                color="white" if v > 26 else "black",
            )
        ax.set(
            xticks=range(4),
            xticklabels=LABELS[1:],
            yticks=range(4),
            yticklabels=LABELS[1:],
            ylabel="True event",
            xlabel="Class in oracle window",
            title=f"{name.capitalize()} confusion",
        )
        for col, keys, title, labels, source in [
            (
                0,
                ["accuracy", "macro_f1", "exact", "iou"],
                "Clean test clips",
                ["Acc.", "F1", "List", "IoU"],
                m,
            ),
            (
                1,
                [
                    "clean",
                    "reversed_original",
                    "shuffled_original",
                    "reversed_semantic",
                ],
                "Oracle-window sensitivity",
                ["Clean", "Reverse¹", "Shuffle¹", "Reverse²"],
                m["sensitivity"],
            ),
        ]:
            ax = axes[1, col]
            ax.bar(
                np.arange(4) + (j - 0.5) * 0.36,
                [source[k] for k in keys],
                0.36,
                color=COLORS[j],
                label=name,
            )
            ax.set(
                ylim=(0, 1.08),
                yticks=[0, 0.25, 0.5, 0.75, 1],
                xticks=range(4),
                xticklabels=labels,
                title=title,
            )
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0,
        0.8,
        "96 held-out scenes\n192 annotated events\n\nList = exact predicted sequence\nIoU = same-class temporal overlap\n¹ Original direction targets\n² Reversed direction targets",
        va="top",
        fontsize=12,
        linespacing=1.6,
    )
    fig.suptitle("TemporalLens   /   Controlled motion study", fontsize=18)
    fig.savefig("web/results.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 6),
        gridspec_kw={"width_ratios": [1, 3]},
        layout="constrained",
    )
    for i, clip in enumerate(
        [
            max(preds["temporal"], key=lambda p: p["iou"]),
            min(preds["temporal"], key=lambda p: p["iou"]),
        ]
    ):
        row = manifest[clip["scene"]]
        with av.open(row["path"]) as movie:
            axes[i, 0].imshow(next(movie.decode(video=0)).to_image())
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Scene {clip['scene']} · first frame")
        ref = [(LABELS.index(e["label"]), e["start"], e["end"]) for e in row["events"]]
        frame = next(p for p in preds["frame"] if p["scene"] == clip["scene"])
        for r, events in enumerate(
            [
                ref,
                [(k, a / 10, b / 10) for k, a, b in frame["predicted"]],
                [(k, a / 10, b / 10) for k, a, b in clip["predicted"]],
            ]
        ):
            for k, a, b in events:
                axes[i, 1].barh(
                    r,
                    b - a,
                    left=a,
                    color=["", "#85b1e3", "#93dab5", "#c7a7dc", "#efb789"][k],
                    height=0.62,
                )
                axes[i, 1].text((a + b) / 2, r, LABELS[k][0], ha="center", va="center")
        axes[i, 1].set(
            xlim=(0, 8),
            ylim=(-0.6, 2.6),
            yticks=[0, 1, 2],
            yticklabels=["Truth", "Frame", "Temporal"],
            xlabel="Seconds",
            title=f"{['Highest', 'Lowest'][i]} temporal IoU · {clip['iou']:.3f}",
        )
    fig.savefig(OUT / "cases.png", dpi=140)
    shutil.copyfile(manifest[30000]["path"], "web/example.mp4")


if __name__ == "__main__":
    main()

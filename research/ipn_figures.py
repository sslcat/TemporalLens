"""Plot saved IPN measurements without redistributing IPN images."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ipn_ingest import OUT
from temporallens.gesture import LABELS


def main():
    train = json.loads((OUT / "training.json").read_text())
    result = json.loads((OUT / "evaluation.json").read_text())
    names = list(train["selected"])
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), layout="constrained")
    for j, k in enumerate(names):
        ax = axes[0, j]
        h = np.array(train["selected"][k][0]["history"])
        ax.plot(h[:, 0], h[:, 1], ".-", label="Training")
        ax.plot(h[:, 0], h[:, 2], ".--", label="Validation")
        ax.set(
            title=k.capitalize(),
            xlabel="Epoch or fitted solution",
            ylabel="Weighted cross-entropy",
        )
        ax.legend(fontsize=8)
        cm = np.array(result[k]["primary"]["cm"])
        ax = axes[1, j]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=16)
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=LABELS[1:],
            yticklabels=LABELS[1:],
            xlabel="Predicted",
            ylabel="True gesture",
        )
        ax.tick_params(axis="x", labelrotation=32)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(
                c,
                r,
                str(v),
                ha="center",
                va="center",
                color="white" if v > 9 else "black",
            )
    fig.suptitle(
        "IPN Hand pilot  /  Validation-selected fits and known-boundary test recognition"
    )
    fig.savefig(OUT / "learning-confusion.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for i, k in enumerate(names):
        r = result[k]
        m = r["primary"]
        axes[0].bar(
            np.arange(4) + (i - 1.5) * 0.19,
            [m["macro_f1"], *m["segment_f1"]],
            0.19,
            label=k,
        )
        axes[1].bar(
            np.arange(4) + (i - 1.5) * 0.19,
            [r["controls"][a]["accuracy"] for a in ("clean", "reverse", "shuffle")]
            + [r["twin_pairs"]["correct"] / r["twin_pairs"]["total"]],
            0.19,
            label=k,
        )
    for ax, labels, title in zip(
        axes,
        [
            ["Oracle F1", "IoU .10", "IoU .25", "IoU .50"],
            ["Isolated", "Reverse", "Shuffle", "AB/BA¹"],
        ],
        ["Recognition and natural-stream segmentation", "Isolated-bout controls"],
    ):
        ax.set(xticks=range(4), xticklabels=labels, ylim=(0, 1.08), title=title)
        ax.legend(fontsize=8)
    fig.supxlabel(
        "¹ Forced assignment across two known bouts; twins share every frame."
    )
    fig.savefig(OUT / "comparisons.png", dpi=140)
    plt.close(fig)
    streams = result["temporal"]["by_source"]

    def event_f1(s):
        tp, fp, fn = s["counts"][-1]
        return 2 * tp / (2 * tp + fp + fn)

    ordered = sorted(streams, key=lambda s: (event_f1(s), s["source"]))
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), layout="constrained")
    colors = ["", "#6393bd", "#55b79c", "#9676c0", "#db985a"]
    for ax, s in zip(axes, [ordered[-1], ordered[0]]):
        for i, events in enumerate([s["ref"], s["pred"]]):
            for k, a, b in events:
                ax.barh(i, (b - a) / 30, left=a / 30, color=colors[k], height=0.52)
        ax.set(
            ylim=(-0.6, 1.6),
            xlim=(0, s["frames"] / 30),
            yticks=[0, 1],
            yticklabels=["Reference", "Predicted"],
            title=f"{s['source']}  /  segment F1@.50 = {event_f1(s):.3f}",
            xlabel="Nominal released-timeline seconds",
        )
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[Patch(color=colors[k], label=LABELS[k]) for k in range(1, 5)],
        ncol=4,
        fontsize=8,
    )
    fig.savefig(OUT / "cases.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

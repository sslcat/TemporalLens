"""Validation-only decision tradeoffs, reliability and candidate confusion."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .rejection_data import OUT


def main():
    metrics = json.loads((OUT / "results.json").read_text())["validation"]
    names = {
        "released": "Released CNN",
        "temperature": "Temperature",
        "margin": "Margin",
        "explicit_other": "Original five-way",
        "learned_gate": "Verified-background gate",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    axes[1].plot([0, 1], [0, 1], ":", color="0.5")
    for k, name in names.items():
        if k not in metrics:
            continue
        row = metrics[k]
        points = row["curve"]["points"]
        if points:
            axes[0].step(
                [p["recall"] for p in points],
                [p["precision"] for p in points],
                where="post",
                alpha=0.85,
                label=name,
            )
            m = row["candidate"]
            if m["precision"] is not None:
                axes[0].scatter(
                    m["recall"],
                    m["precision"],
                    s=32,
                    color=axes[0].lines[-1].get_color(),
                )
        bins = row["target_correctness"]["bins"]
        axes[1].plot(
            [b["confidence"] for b in bins],
            [b["frequency"] for b in bins],
            ".-",
            label=name,
        )
    axes[0].set(
        xlabel="Recall among matched proposals",
        ylabel="Correct labels / accepted proposals",
        title="Ranking and operating points",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
    )
    axes[1].set(
        xlabel="Mean top-target score",
        ylabel="Fraction correctly localized and labeled",
        title="Reliability on all validation proposals",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
    )
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "tradeoffs.png", dpi=150)
    plt.close(fig)
    keys = [
        "released",
        "no_rejection",
        "temperature",
        "per_class",
        "margin",
        "explicit_other",
        "constant_gate",
        "learned_gate",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for ax, k in zip(axes.flat, keys):
        if k not in metrics:
            ax.axis("off")
            continue
        mat = np.array(metrics[k]["confusion"])
        ax.imshow(np.log1p(mat), cmap="Blues", vmin=0, vmax=np.log(86))
        for (i, j), n in np.ndenumerate(mat):
            ax.text(
                j,
                i,
                str(n),
                ha="center",
                va="center",
                color="white" if n > 14 else "black",
            )
        ax.set(
            title=k.replace("_", " ").capitalize(),
            xticks=range(5),
            yticks=range(5),
            xticklabels=["Reject", "Left", "Right", "In", "Out"],
            yticklabels=["Unmatched", "Left", "Right", "In", "Out"],
        )
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle(
        "Validation candidates: fixed one-to-one reference assignment, predicted decisions on columns"
    )
    fig.savefig(OUT / "confusions.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

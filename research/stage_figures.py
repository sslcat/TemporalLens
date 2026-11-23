"""Render measured natural-stream performance and representative study cases."""

import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from temporallens.two_stage import OUT
from temporallens.gesture import LABELS


def main():
    result = json.loads((OUT / "test.json").read_text())
    metrics = result["summary"]
    names = ["baseline", "temporal", "flow", "fusion"]
    labels = ["Frozen CNN", "Proposals + CNN", "Proposals + flow", "Proposals + fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    x = np.arange(4)
    for j, t in enumerate((".10", ".25", ".50")):
        axes[0].bar(
            x + (j - 1) * 0.24,
            [metrics[k]["segment_f1"][j] for k in names],
            0.24,
            label=t,
        )
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Class-aware segment F1",
        title="Uninterrupted test streams",
    )
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].legend(title="Temporal IoU")
    for k, label in [
        ("energy", "Motion energy"),
        ("binary", "Motion logistic"),
        ("hybrid", "Motion + visual logistic"),
    ]:
        axes[1].plot([0.1, 0.25, 0.5], metrics[k]["recall"], "o-", label=label)
    axes[1].set(
        xlabel="Temporal IoU",
        ylabel="Class-agnostic recall",
        ylim=(0, 1),
        xticks=[0.1, 0.25, 0.5],
        title="Candidate detection",
    )
    axes[1].legend()
    fig.savefig(OUT / "comparison.png", dpi=140)
    plt.close(fig)
    primary = json.loads((OUT / "selection.json").read_text())["primary"]

    def rank(item):
        counts = item[1]["metrics"][primary]["counts"][-1]
        tp, fp, fn = counts
        return (2 * tp / (2 * tp + fp + fn), item[0])

    ordered = sorted(result["by_source"].items(), key=rank)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), layout="constrained")
    colors = ["#959595", "#2e77a6", "#268273", "#9c5e91", "#bd7436"]
    for ax, (source, data) in zip(axes, (ordered[-1], ordered[0])):
        for row, events in enumerate(
            [
                data["reference"],
                data["baseline"],
                data["candidates"],
                data["predictions"][primary],
            ]
        ):
            for k, a, b in events:
                ax.barh(
                    row,
                    (b - a) / 30,
                    left=a / 30,
                    height=0.58,
                    color=colors[0 if row == 2 else k],
                )
        ax.set(
            yticks=range(4),
            yticklabels=["Reference", "Frozen CNN", "Proposals", "Two-stage"],
            title=source,
            xlabel="Nominal seconds",
        )
        ax.invert_yaxis()
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[Patch(color=colors[i], label=LABELS[i]) for i in range(1, 5)],
        ncols=4,
        fontsize=8,
    )
    fig.savefig(OUT / "cases.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Supplementary readouts from the frozen primary models and training-only time prior."""

import json
from collections import Counter
from itertools import combinations

import numpy as np
import torch
from .ipn_ingest import WORK, OUT
from .ipn_features import flow_vector
from .ipn_metrics import overlap, runs, source_score, summarize
from .ipn_train import FlowLogit, load_split, predict, deterministic
from temporallens.gesture import GestureHead, detect


def timeline_bins(n):
    return np.arange(n) * 20 // n


def joined_pairs(model, kind, state, record):
    """Splice the same two real bouts in both orders and infer across the cut."""
    counts = Counter()
    bouts = [
        (k, slice((a + 3) // 4, (b + 3) // 4)) for k, a, b in runs(record["dense"])
    ]
    if kind == "flow":
        with np.load(WORK / "cache" / (record["row"]["source"] + ".npz")) as f:
            gray = f["gray"]
    for first, second in combinations(bouts, 2):
        both_exact = True
        for (a, left), (b, right) in ((first, second), (second, first)):
            cut = left.stop - left.start
            key = "motion" if kind == "flow" else "x"
            stream = torch.cat((record[key][left], record[key][right]))
            if kind == "flow":
                # Recompute the first flow in each bout for its new predecessor.
                for tick, prev, curr in (
                    (0, left.start, left.start),
                    (cut, left.stop - 1, right.start),
                ):
                    stream[tick] = torch.as_tensor(
                        flow_vector(gray[prev], gray[curr]), device=stream.device
                    )
            logits = predict(model, kind, {key: stream})
            lscore = logits[:cut].mean(0)
            rscore = logits[cut:].mean(0)
            margin = (lscore[a] + rscore[b] - lscore[b] - rscore[a]).item()
            counts["given_labels_correct"] += (margin > 0) + 0.5 * (margin == 0)
            counts["free_top1_correct"] += int(
                lscore[1:].argmax().item() + 1 == a
                and rscore[1:].argmax().item() + 1 == b
            )
            guesses = [
                k
                for k, _, _ in detect(
                    logits.softmax(-1).cpu().numpy(),
                    state["threshold"],
                    state["minimum"],
                )
            ]
            exact = guesses == [a, b]
            counts["decoded_exact"] += int(exact)
            both_exact = both_exact and exact
            counts["presentations"] += 1
        counts["twins"] += 1
        counts["both_orders_decoded_exact"] += int(both_exact)
    return counts


def main():
    deterministic(17)
    records = load_split("test")
    counts = np.ones((20, 5))
    for row in json.loads((OUT / "split.json").read_text()):
        if row["split"] == "train":
            with np.load(WORK / "cache" / (row["source"] + ".npz")) as f:
                y = f["y"]
                np.add.at(counts, (timeline_bins(len(y)), y), 1)
    prior = counts / counts.sum(1, keepdims=True)
    prior_stats = summarize(
        [
            source_score(prior[timeline_bins(len(r["y"]))], r["dense"], 0.65, 2)
            for r in records
        ]
    )
    report = dict(time_only_oracle_macro_f1=prior_stats["macro_f1"])
    for kind in ("frame", "pooled", "temporal", "flow"):
        state = torch.load(
            OUT / ("gesture.pt" if kind == "temporal" else kind + ".pt"),
            weights_only=True,
            map_location="cuda",
        )
        model = (FlowLogit() if kind == "flow" else GestureHead(kind)).cuda().eval()
        model.load_state_dict(state["weights"])
        dense_cm = np.zeros((5, 5), int)
        mid_cm = np.zeros((4, 4), int)
        pairs = exact = 0
        joined = Counter()
        ious = []
        for r in records:
            prob = predict(model, kind, r).softmax(-1).cpu().numpy()
            joined.update(joined_pairs(model, kind, state, r))
            dense = r["dense"]
            truth = runs(dense)
            np.add.at(dense_cm, (dense[::4], prob.argmax(1)), 1)
            pred = [
                (k, a * 4, min(b * 4, len(dense)))
                for k, a, b in detect(prob, state["threshold"], state["minimum"])
            ]
            for k, a, b in truth:
                mid_cm[k - 1, prob[(a + b - 1) // 8, 1:].argmax()] += 1
                ious.append(
                    max((overlap((k, a, b), p) for p in pred if p[0] == k), default=0)
                )
            for left, right in zip(truth, truth[1:]):
                # 112 whole sample ticks (14.93 nominal seconds), containing both complete gestures.
                span = right[2] - left[1]
                if span > 448:
                    continue
                first = int(
                    np.clip((left[1] - (448 - span) // 2) // 4, 0, len(prob) - 112)
                )
                if 4 * first > left[1] or 4 * (first + 112) < right[2]:
                    continue
                clip = {k: r[k][first : first + 112] for k in ("motion", "x")}
                if kind == "flow":
                    clip["motion"] = clip["motion"].clone()
                    clip["motion"][0] = 0
                p = predict(model, kind, clip).softmax(-1).cpu().numpy()
                ids = [k for k, a, b in detect(p, state["threshold"], state["minimum"])]
                ref = [k for k, a, b in runs(dense[4 * first : 4 * (first + 112)])]
                pairs += 1
                exact += int(ids == ref)

        def macro(cm):
            return float((2 * cm.diagonal() / (cm.sum(0) + cm.sum(1)).clip(1)).mean())

        report[kind] = dict(
            raw_dense_macro_f1=macro(dense_cm),
            midpoint_macro_f1=macro(mid_cm),
            mean_best_tiou=float(np.mean(ious)),
            joined_pairs=dict(joined),
            enriched_pairs=pairs,
            enriched_exact=exact,
        )
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()

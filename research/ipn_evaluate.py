"""Evaluate frozen validation selections on uninterrupted test sources and gesture bouts."""

import json
from collections import defaultdict
from itertools import combinations
import numpy as np
import torch

from .ipn_ingest import OUT, WORK
from .ipn_metrics import runs, source_score, summarize
from .ipn_features import flow_sequence
from .ipn_train import FlowLogit, load_split, predict, deterministic
from temporallens.gesture import GestureHead, padded, detect


def classification(cm):
    return dict(
        accuracy=float(cm.trace() / cm.sum()),
        macro_f1=float((2 * cm.diagonal() / (cm.sum(0) + cm.sum(1)).clip(1)).mean()),
        cm=cm.tolist(),
    )


@torch.inference_mode()
def controls(model, kind, r, seed):
    clean = []
    matrices = {s: np.zeros((4, 4), int) for s in ("clean", "reverse", "shuffle")}
    rng = np.random.default_rng(seed)
    with np.load(WORK / "cache" / (r["row"]["source"] + ".npz")) as f:
        gray = f["gray"] if kind == "flow" else None
    for k, a, b in runs(r["dense"]):
        ids = np.arange((a + 3) // 4, (b + 3) // 4)
        for key, idx in [
            ("clean", ids),
            ("reverse", ids[::-1].copy()),
            ("shuffle", rng.permutation(ids)),
        ]:
            if kind == "flow":
                obs = torch.tensor(flow_sequence(gray[idx]), device="cuda")
                logits = model(obs)
            else:
                obs = r["x"][idx]
                if kind == "pooled":
                    # Given a bout, make the complete-bout bag exactly order invariant.
                    obs = obs.mean(0, keepdim=True)
                logits = model(padded(obs))[0]
            score = logits.mean(0).cpu().numpy()
            matrices[key][k - 1, score[1:].argmax()] += 1
            if key == "clean":
                clean.append((k, score))
    hits = free = total = 0
    for a, b in combinations(clean, 2):
        for first, second in ((a, b), (b, a)):
            first_label, first_scores = first
            second_label, second_scores = second
            diff = (first_scores[first_label] + second_scores[second_label]) - (
                first_scores[second_label] + second_scores[first_label]
            )
            hits += float(diff > 0) + 0.5 * float(diff == 0)
            free += int(
                first_scores[1:].argmax() + 1 == first_label
                and second_scores[1:].argmax() + 1 == second_label
            )
            total += 1
    return matrices, hits, free, total


def bootstrap(grouped):
    subjects = sorted(grouped)
    rng = np.random.default_rng(731)
    draws = []
    for _ in range(2000):
        sample = [
            s for person in rng.choice(subjects, len(subjects)) for s in grouped[person]
        ]
        m = summarize(sample)
        draws.append([m["macro_f1"], m["segment_f1"][2], m["exact"]])
    return np.asarray(draws)


def main():
    deterministic(17)
    records = load_split("test")
    training = json.loads((OUT / "training.json").read_text())
    result = {}
    boot = {}
    for kind, settings in training["selected"].items():
        seeds = []
        for i, conf in enumerate(settings):
            state = torch.load(conf["path"], map_location="cuda", weights_only=True)
            model = (FlowLogit() if kind == "flow" else GestureHead(kind)).cuda().eval()
            model.load_state_dict(state["weights"])
            scores = []
            groups = defaultdict(list)
            ctrl = defaultdict(lambda: np.zeros((4, 4), int))
            pair = np.zeros(3)
            tiles = []
            for n, r in enumerate(records):
                prob = predict(model, kind, r).softmax(-1).cpu().numpy()
                score = source_score(
                    prob, r["dense"], state["threshold"], state["minimum"]
                )
                score["source"] = r["row"]["source"]
                scores.append(score)
                groups[r["row"]["participant"]].append(score)
                if i == 0:
                    mats, h, f, t = controls(model, kind, r, 1729 + n)
                    for k, m in mats.items():
                        ctrl[k] += m
                    pair += [h, f, t]
                    # Nonoverlapping app-size crops. Keep only complete 12-second tiles.
                    for a in range(0, len(r["dense"]) - 359, 360):
                        j = a // 4
                        clip = {k: r[k][j : j + 90] for k in ("x", "motion")}
                        if kind == "flow":
                            clip["motion"] = clip["motion"].clone()
                            clip["motion"][0] = 0
                        sub = predict(model, kind, clip).softmax(-1).cpu().numpy()
                        accepted = [
                            k
                            for k, _, _ in detect(
                                sub, state["threshold"], state["minimum"]
                            )
                        ]
                        truth = [k for k, _, _ in runs(r["dense"][a : a + 360])]
                        tiles.append((truth, accepted))
            stats = summarize(scores)
            seeds.append(dict(seed=conf["seed"], **stats))
            if i == 0:
                boot[kind] = bootstrap(groups)
                result[kind] = dict(
                    primary=stats,
                    by_source=scores,
                    ci95=dict(
                        zip(
                            ("macro_f1", "segment_f1_50", "exact"),
                            np.percentile(boot[kind], [2.5, 97.5], axis=0).T.tolist(),
                        )
                    ),
                    controls={k: classification(v) for k, v in ctrl.items()},
                    twin_pairs=dict(
                        correct=pair[0], free_correct=pair[1], total=int(pair[2])
                    ),
                    tiles={
                        s: dict(n=len(sub), correct=sum(a == b for a, b in sub))
                        for s, sub in [
                            ("all", tiles),
                            ("empty", [(a, b) for a, b in tiles if not a]),
                            ("multi", [(a, b) for a, b in tiles if len(a) >= 2]),
                        ]
                    },
                )
        result[kind]["seeds"] = seeds
        print(kind, result[kind]["primary"], flush=True)
    result["paired_gain_ci95"] = {
        k: np.percentile(boot["temporal"] - boot[k], [2.5, 97.5], axis=0).T.tolist()
        for k in ("frame", "pooled", "flow")
    }
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

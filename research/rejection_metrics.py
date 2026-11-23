"""Candidate-level scoring and participant-excluded operating points."""

import numpy as np
import torch
from torch.nn import functional as F

from .ipn_metrics import match

PROB_CUTS = (0.35, 0.45, 0.60, 0.75, 0.85)
JOINT_CUTS = (0, 0.20, 0.35, 0.45, 0.60)
MARGINS = (0, 0.10, 0.20, 0.30, 0.40, 0.50)
METHODS = (
    "released",
    "no_rejection",
    "global_cutoff",
    "temperature",
    "per_class",
    "margin",
    "explicit_other",
    "constant_gate",
    "learned_gate",
)


def softmax(x):
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def confusion(y, pred):
    return np.bincount(5 * y + pred, minlength=25).reshape(5, 5)


def score(cm):
    tp = int(cm.diagonal()[1:].sum())
    fp = int(cm[:, 1:].sum()) - tp
    fn = int(cm[1:, :].sum()) - tp
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / max(1, tp + fn),
        f1=2 * tp / max(1, 2 * tp + fp + fn),
    )


def best_cut(values, labels, y, eligible, grid, default):
    best = None
    for t in grid:
        pred = np.where(eligible & (values >= t), labels, 0)
        m = score(confusion(y, pred))
        key = (m["f1"], -m["fp"], -abs(t - default))
        if best is None or key > best[0]:
            best = key, t
    return best[1]


def temperature(logits, y):
    if not np.any(y > 0):
        return 1.0
    x = torch.as_tensor(logits[y > 0, 1:], device="cuda", dtype=torch.float64)
    target = torch.as_tensor(y[y > 0] - 1, device="cuda")
    ts = torch.logspace(np.log10(0.25), np.log10(4), 81, device="cuda", dtype=x.dtype)
    nll = -F.log_softmax(x[None, :, :] / ts[:, None, None], dim=-1)[
        :, torch.arange(len(x), device=x.device), target
    ].mean(1)
    return ts[nll.argmin()].item()


def signals(method, logits, gate, prior, temp=1.0):
    cond = softmax(logits[:, 1:] / temp)
    top = cond.argmax(1) + 1
    ordered = np.sort(cond, axis=1)
    confidence = ordered[:, -1]
    eligible = np.ones(len(top), bool)
    if method == "explicit_other":
        prob = softmax(logits)
        confidence = prob[np.arange(len(top)), top]
        eligible = confidence > prob[:, 0]
    elif method in ("constant_gate", "learned_gate"):
        presence = gate if method == "learned_gate" else np.full(len(top), prior)
        confidence = presence * confidence
        eligible = confidence > 1 - presence
    value = ordered[:, -1] - ordered[:, -2] if method == "margin" else confidence
    return top, confidence, value, eligible


def calibrate(method, logits, y, gate, prior):
    cfg = dict(temperature=1.0, cutoff=0.60)
    if method == "temperature":
        cfg["temperature"] = temperature(logits, y)
    top, _, values, eligible = signals(method, logits, gate, prior, cfg["temperature"])
    if method == "no_rejection":
        cfg["cutoff"] = 0.0
    elif method in (
        "global_cutoff",
        "per_class",
        "margin",
        "explicit_other",
        "constant_gate",
        "learned_gate",
    ):
        grid, default = (
            (MARGINS, 0.20)
            if method == "margin"
            else (JOINT_CUTS, 0.60)
            if method in ("explicit_other", "constant_gate", "learned_gate")
            else (PROB_CUTS, 0.60)
        )
        cfg["cutoff"] = best_cut(values, top, y, eligible, grid, default)
        if method == "per_class":
            cuts = []
            for k in range(1, 5):
                if np.any((top == k) & (y == k)):
                    # False negatives include every true k, even if its predicted name is wrong.
                    cuts.append(
                        best_cut(
                            values,
                            top,
                            np.where(y == k, k, 0),
                            eligible & (top == k),
                            grid,
                            default,
                        )
                    )
                else:
                    cuts.append(cfg["cutoff"])
            cfg["per_class"] = cuts
    return cfg


def apply_rule(method, cfg, logits, gate, prior):
    top, confidence, values, eligible = signals(
        method, logits, gate, prior, cfg["temperature"]
    )
    cut = np.array(cfg["per_class"])[top - 1] if "per_class" in cfg else cfg["cutoff"]
    return np.where(eligible & (values >= cut), top, 0), confidence, values, eligible


def reliability(conf, correct):
    if len(conf) == 0:
        return None
    bins = []
    ece = 0.0
    for i in range(5):
        ix = np.minimum((conf * 5).astype(int), 4) == i
        if ix.any():
            a = float(conf[ix].mean())
            b = float(correct[ix].mean())
            bins.append(dict(bin=i, n=int(ix.sum()), confidence=a, frequency=b))
            ece += ix.mean() * abs(a - b)
    return dict(
        n=len(conf),
        ece=float(ece),
        brier=float(np.mean((conf - correct.astype(float)) ** 2)),
        bins=bins,
    )


def pr_curve(top, values, eligible, y):
    rows = []
    for t in sorted(set(values[eligible]), reverse=True):
        m = score(confusion(y, np.where(eligible & (values >= t), top, 0)))
        rows.append(
            dict(
                cut=float(t),
                precision=m["precision"],
                recall=m["recall"],
                tp=m["tp"],
                fp=m["fp"],
            )
        )
    ap = 0.0
    prev = 0.0
    for r in rows:
        ap += (r["recall"] - prev) * r["precision"]
        prev = r["recall"]
    return dict(points=rows, average_precision=ap)


def evaluate(rows, sources, pred, conf, ranking, eligible):
    labels = np.array([r["label"] for r in rows])
    top = np.array([np.argmax(r["logits"][1:]) + 1 for r in rows])
    cm = confusion(labels, pred)
    stream_tp = stream_fp = stream_fn = frames = 0
    for name, source in sources.items():
        found = [
            (int(k), r["start"], r["end"])
            for r, k in zip(rows, pred)
            if r["source"] == name and k
        ]
        pairs = match(source["ref"], found, 0.5)
        stream_tp += len(pairs)
        stream_fp += len(found) - len(pairs)
        stream_fn += len(source["ref"]) - len(pairs)
        frames += source["frames"]
    people = {}
    for person in sorted({r["person"] for r in rows}):
        ix = np.array([r["person"] == person for r in rows])
        people[person] = score(confusion(labels[ix], pred[ix]))
    m = score(cm)
    return dict(
        candidate=m,
        confusion=cm.tolist(),
        people=people,
        matched_rejected=int(cm[1:, 0].sum()),
        matched_wrong=int(cm[1:, 1:].sum()) - m["tp"],
        target_correctness=reliability(conf, top == labels),
        curve=pr_curve(top, ranking, eligible, labels),
        stream=dict(
            tp=stream_tp,
            fp=stream_fp,
            fn=stream_fn,
            f1=2 * stream_tp / max(1, 2 * stream_tp + stream_fp + stream_fn),
            false_per_min=1800 * stream_fp / frames,
        ),
    )

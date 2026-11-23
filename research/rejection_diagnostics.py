"""Describe candidate score errors and shifts without fitting a decision rule."""

from collections import Counter
import numpy as np
from .rejection_metrics import confusion, reliability, softmax


def quantiles(values):
    return np.percentile(values, [10, 50, 90]).tolist() if len(values) else None


def describe(rows):
    logits = np.array([r["logits"] for r in rows])
    y = np.array([r["label"] for r in rows])
    top = logits[:, 1:].argmax(1) + 1
    cond = softmax(logits[:, 1:])
    ordered = np.sort(cond, axis=1)
    result = {}
    for stratum in ("all", "matched", "verified_other", "partial_or_duplicate"):
        ix = np.array([stratum == "all" or r["stratum"] == stratum for r in rows])
        subset = [r for r, keep in zip(rows, ix) if keep]
        labels = Counter()
        for r in subset:
            labels.update(r["original_annotation_frames"])
        result[stratum] = dict(
            n=len(subset),
            participants=len({r["person"] for r in subset}),
            true_counts=np.bincount(y[ix], minlength=5).tolist(),
            logit_p10_p50_p90=[quantiles(logits[ix, k]) for k in range(5)],
            top_target_score=quantiles(ordered[ix, -1]),
            top_two_prob_gap=quantiles(ordered[ix, -1] - ordered[ix, -2]),
            target_minus_other_logit=quantiles(logits[ix, 1:].max(1) - logits[ix, 0]),
            other_wins_5way=int((logits[ix].argmax(1) == 0).sum()),
            threshold_accepts=int((ordered[ix, -1] >= 0.6).sum()),
            top_target_correct=int((top[ix] == y[ix]).sum()),
            original_annotation_frame_counts=dict(labels),
        )
    matched = [r for r in rows if r["label"]]
    paired = []
    for r in matched:
        k = r["label"]

        def summarize(logits):
            p = softmax(np.array([logits[1:]]))[0]
            return dict(
                top=int(p.argmax() + 1),
                true_score=float(p[k - 1]),
                max_score=float(p.max()),
                accepted=bool(p.max() >= 0.6),
            )

        a, b = r["reference"]
        paired.append(
            dict(
                source=r["source"],
                person=r["person"],
                candidate=[r["start"], r["end"]],
                truth=k,
                iou=r["iou"],
                target_purity=(min(r["end"], b) - max(r["start"], a))
                / (r["end"] - r["start"]),
                start_error=r["start"] - a,
                end_error=r["end"] - b,
                candidate_scores=summarize(r["logits"]),
                oracle_scores=summarize(r["oracle"]),
            )
        )

    def accepted_correct(row, key):
        return row[key]["accepted"] and row[key]["top"] == row["truth"]

    result["alignment"] = dict(
        rows=paired,
        oracle_correct=sum(r["oracle_scores"]["top"] == r["truth"] for r in paired),
        candidate_correct=sum(
            r["candidate_scores"]["top"] == r["truth"] for r in paired
        ),
        fixed_by_oracle=sum(
            not accepted_correct(r, "candidate_scores")
            and accepted_correct(r, "oracle_scores")
            for r in paired
        ),
        lost_by_oracle=sum(
            accepted_correct(r, "candidate_scores")
            and not accepted_correct(r, "oracle_scores")
            for r in paired
        ),
        median_true_score_delta=float(
            np.median(
                [
                    r["oracle_scores"]["true_score"]
                    - r["candidate_scores"]["true_score"]
                    for r in paired
                ]
            )
        )
        if paired
        else None,
    )
    return result


def shift(train, val):
    tr, _, td, te, _ = train
    vr, _, vd, ve, _ = val
    result = {}
    for group in ("matched", "verified_other"):
        a = np.array([r["stratum"] == group for r in tr])
        b = np.array([r["stratum"] == group for r in vr])
        train_people = np.array([r["person"] for r in tr])[a]
        part = {}
        for name, t, v in (("dino", te[a], ve[b]), ("head_descriptor", td[a], vd[b])):
            if not len(v) or len(set(train_people)) < 2:
                continue
            standardized = (v.mean(0) - t.mean(0)) / np.maximum(t.std(0), 0.015)
            t = t / np.linalg.norm(t, axis=1, keepdims=True).clip(1e-12)
            v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)
            dt = np.maximum(0, 1 - t @ t.T)
            dt[train_people[:, None] == train_people] = np.inf
            dv = np.maximum(0, 1 - v @ t.T)
            part[name] = dict(
                median_abs_standardized_mean_shift=float(np.median(abs(standardized))),
                train_other_person_nearest_cosine=quantiles(dt.min(1)),
                val_nearest_train_cosine=quantiles(dv.min(1)),
            )
        result[group] = part
    return result


def conditional_metrics(logits, y, temperatures):
    mask = y > 0
    probs = softmax(logits[mask, 1:] / temperatures[mask, None])
    labels = y[mask] - 1
    if not len(labels):
        return None
    return dict(
        n=len(labels),
        nll=float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-15)).mean()),
        multiclass_brier=float(np.square(probs - np.eye(4)[labels]).sum(1).mean()),
        reliability=reliability(probs.max(1), probs.argmax(1) == labels),
        confusion=confusion(labels + 1, probs.argmax(1) + 1)[1:, 1:].tolist(),
    )

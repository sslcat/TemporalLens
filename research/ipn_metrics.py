"""Source-level sufficient statistics. Reference intervals are original JPEG frames."""

import numpy as np
from temporallens.gesture import detect, runs


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[1], b[1])) / (
        max(a[2], b[2]) - min(a[1], b[1])
    )


def match(ref, pred, cutoff):
    candidates = sorted(
        (
            (overlap(a, b), i, j)
            for i, a in enumerate(ref)
            for j, b in enumerate(pred)
            if a[0] == b[0] and overlap(a, b) >= cutoff
        ),
        reverse=True,
    )
    used_r, used_p = set(), set()
    pairs = []
    for _, i, j in candidates:
        if i not in used_r and j not in used_p:
            used_r.add(i)
            used_p.add(j)
            pairs.append((i, j))
    return pairs


def source_score(prob, dense, threshold, minimum):
    ref = runs(dense)
    pred = [
        (k, a * 4, min(b * 4, len(dense)))
        for k, a, b in detect(prob, threshold, minimum)
    ]
    cm = np.zeros((4, 4), int)
    for k, a, b in ref:
        i, j = (a + 3) // 4, (b + 3) // 4
        guess = np.log(np.maximum(prob[i:j, 1:], 1e-12))
        cm[k - 1, guess.mean(0).argmax()] += 1
    counts = []
    errors = []
    for cut in (0.1, 0.25, 0.5):
        pairs = match(ref, pred, cut)
        counts.append([len(pairs), len(pred) - len(pairs), len(ref) - len(pairs)])
    for i, j in pairs:
        errors.extend([abs(ref[i][1] - pred[j][1]), abs(ref[i][2] - pred[j][2])])
    order = [k for k, a, b in ref]
    reported = [k for k, a, b in pred]
    found = dict(pairs)
    correct = covered = 0
    for i in range(len(ref)):
        for j in range(i + 1, len(ref)):
            if i in found and j in found:
                covered += 1
                correct += int(pred[found[i]][1] < pred[found[j]][1])
    return dict(
        cm=cm.tolist(),
        counts=counts,
        ref=ref,
        pred=pred,
        error=sum(errors),
        endpoints=len(errors),
        exact=int(order == reported),
        bag=int(sorted(order) == sorted(reported)),
        frames=len(dense),
        pair=[correct, covered, len(ref) * (len(ref) - 1) // 2],
    )


def summarize(scores):
    cm = np.sum([s["cm"] for s in scores], axis=0)
    precision = cm.diagonal() / cm.sum(0).clip(1)
    recall = cm.diagonal() / cm.sum(1).clip(1)
    f1 = 2 * cm.diagonal() / (cm.sum(0) + cm.sum(1)).clip(1)
    tp, fp, fn = np.sum([s["counts"] for s in scores], axis=0).T
    endpoints = sum(s["endpoints"] for s in scores)
    correct, covered, total = np.sum([s["pair"] for s in scores], axis=0)
    return dict(
        accuracy=float(cm.trace() / cm.sum()),
        macro_f1=float(f1.mean()),
        cm=cm.tolist(),
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1=f1.tolist(),
        segment_f1=(2 * tp / (2 * tp + fp + fn).clip(1)).tolist(),
        matches=int(tp[-1]),
        false_alarms=int(fp[-1]),
        misses=int(fn[-1]),
        false_alarms_per_minute=float(fp[-1] * 1800 / sum(s["frames"] for s in scores)),
        boundary_mae_frames=sum(s["error"] for s in scores) / endpoints
        if endpoints
        else None,
        exact=float(np.mean([s["exact"] for s in scores])),
        bag=float(np.mean([s["bag"] for s in scores])),
        pair_coverage=float(covered / total) if total else None,
        pair_correct_all=float(correct / total) if total else None,
        pair_correct_matched=float(correct / covered) if covered else None,
    )

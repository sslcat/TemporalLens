"""One-to-one stream scoring in unrounded, original frame coordinates."""

import numpy as np
from .ipn_metrics import match

CUTS = (0.1, 0.25, 0.5)


def stream_stats(ref, pred, n, cm=None):
    stats = dict(
        frames=n,
        oracle=np.zeros((4, 4), int).tolist() if cm is None else cm.tolist(),
        exact=int([r[0] for r in ref] == [p[0] for p in pred]),
    )
    counts = []
    for threshold in CUTS:
        pairs = match(ref, pred, threshold)
        counts.append([len(pairs), len(pred) - len(pairs), len(ref) - len(pairs)])
    by_class = np.zeros((4, 3), int)
    for k in range(1, 5):
        tp = sum(ref[i][0] == k for i, _ in pairs)
        by_class[k - 1] = (
            tp,
            sum(x[0] == k for x in pred) - tp,
            sum(x[0] == k for x in ref) - tp,
        )
    offsets = [[pred[j][1] - ref[i][1], pred[j][2] - ref[i][2]] for i, j in pairs]
    return dict(**stats, counts=counts, by_class=by_class.tolist(), offsets=offsets)


def aggregate(rows):
    tp, fp, fn = np.sum([s["counts"] for s in rows], 0).T
    tc, fc, nc = np.sum([s["by_class"] for s in rows], 0).T
    cm = np.sum([s["oracle"] for s in rows], 0)
    offsets = np.array([pair for s in rows for pair in s["offsets"]])
    return dict(
        counts=np.stack((tp, fp, fn), 1).tolist(),
        segment_f1=(2 * tp / (2 * tp + fp + fn).clip(1)).tolist(),
        recall=(tp / (tp + fn).clip(1)).tolist(),
        false_per_min=float(fp[-1] * 1800 / sum(s["frames"] for s in rows)),
        endpoint_mae_frames=float(np.abs(offsets).mean()) if offsets.size else None,
        event_macro_f1=float((2 * tc / (2 * tc + fc + nc).clip(1)).mean()),
        oracle_macro_f1=float(
            (2 * cm.diagonal() / (cm.sum(0) + cm.sum(1)).clip(1)).mean()
        )
        if cm.sum()
        else None,
        oracle_cm=cm.tolist(),
        exact=sum(s["exact"] for s in rows),
        sources=len(rows),
    )


def frame_spans(spans, n):
    return [(k, 4 * a, min(4 * b, n)) for k, a, b in spans]

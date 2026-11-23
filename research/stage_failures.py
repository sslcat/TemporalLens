"""Descriptive frozen-baseline diagnostics. Test mode requires the model-selection lock."""

import argparse
import csv
from collections import Counter, defaultdict
import json
import numpy as np
import torch
from .ipn_ingest import META, WORK
from .ipn_metrics import match, overlap
from .stage_metrics import aggregate, frame_spans, stream_stats
from .stage_study import prepare, sample_bounds, dump, hashes, LOCK_FILES
from .ipn_train import deterministic
from temporallens.gesture import detect
from temporallens.two_stage import OUT


def probes(train, records):
    def bouts(rows):
        entries = []
        for r in rows:
            for k, a, b in r["ref"]:
                i, j = sample_bounds(a, b)
                x = r["x"][i:j]
                n = max(1, len(x) // 3)
                entries.append((k, x.mean(0), x[-n:].mean(0) - x[:n].mean(0)))
        return entries

    train, test = bouts(train), bouts(records)
    report = {}
    for group in ((1, 2), (3, 4)):
        summary = {}
        for idx, name in ((1, "mean"), (2, "late_minus_early")):
            tr = [e for e in train if e[0] in group]
            te = [e for e in test if e[0] in group]
            x = torch.stack([e[idx] for e in tr])
            y = torch.tensor(
                [1.0 if e[0] == group[1] else -1.0 for e in tr], device=x.device
            )
            center = x.mean(0)
            scale = x.std(0).clamp_min(0.015)

            def design(raw):
                return torch.nn.functional.normalize((raw - center) / scale, dim=1)

            X = design(x)
            # Dual ridge on a normalized linear kernel; both training classes balanced.
            beta = X.T @ torch.linalg.solve(
                X @ X.T + torch.eye(len(X), device=x.device), y
            )
            raw = torch.stack([e[idx] for e in te])
            truth = np.array([e[0] == group[1] for e in te])

            def correct(raw):
                return int(((design(raw) @ beta > 0).cpu().numpy() == truth).sum())

            summary[name] = dict(
                n=len(te),
                clean=correct(raw),
                reverse=correct(-raw if idx == 2 else raw),
            )
        report[str(group)] = summary
    return report


def category(event, others):
    hits = [s for s in others if overlap(event, s) > 0]
    same = [s for s in hits if s[0] == event[0]]
    if not hits:
        return "no_target_overlap"
    if not same:
        return "wrong_class_only"
    if max(overlap(event, s) for s in same) < 0.5:
        return "same_class_low_iou"
    return "one_to_one_competition"


def edge_motion(source):
    with np.load(WORK / "cache" / (source + ".npz")) as f:
        g = f["gray"]
    mask = np.ones(g.shape[1:], bool)
    mask[8:-8, 10:-10] = False
    border = g[:, mask].astype(np.int16)
    return (np.abs(np.diff(border, axis=0, prepend=border[:1])) > 12).mean(1)


def audit(records, split):
    annotations = defaultdict(list)
    with (META / "Annot_List.txt").open() as f:
        for a in csv.DictReader(f):
            annotations[a["video"]].append(
                (int(a["t_start"]) - 1, int(a["t_end"]), a["label"])
            )
    alarms, misses, codes = Counter(), Counter(), Counter()
    fragments = Counter()
    edges = np.zeros((2, 2), int)  # high/low versus ticks / background false alarms
    offset = []
    loose_offset = []
    all_p = []
    all_y = []
    thresholds = (0.45, 0.65, 0.80, 0.9) if split == "val" else (0.65,)
    sweeps = defaultdict(list)
    per_source = []
    for r in records:
        p = r["visual"].softmax(-1).cpu().numpy()
        truth = r["ref"]
        n = len(r["dense"])
        preds = frame_spans(detect(p, 0.65, 3), n)
        pairs = match(truth, preds, 0.5)
        loose = match(
            [(1, a, b) for _, a, b in truth], [(1, a, b) for _, a, b in preds], 0.10
        )
        for pairs_here, collector in [(pairs, offset), (loose, loose_offset)]:
            for i, j in pairs_here:
                collector.append([preds[j][1] - truth[i][1], preds[j][2] - truth[i][2]])
        tg, pg = {i for i, j in pairs}, {j for i, j in pairs}
        source = r["row"]["source"]
        active = edge_motion(source)
        other = r["dense"][::4] == 0
        hot = active > np.quantile(active[other], 0.75)
        edges[0, 0] += np.sum(hot & other)
        edges[1, 0] += np.sum(~hot & other)
        for j, event in enumerate(preds):
            if j in pg:
                continue
            cat = category(event, truth)
            alarms[cat] += 1
            mid = (event[1] + event[2] - 1) // 2
            codes[next(c for a, b, c in annotations[source] if a <= mid < b)] += 1
            if cat == "no_target_overlap":
                edges[int(not hot[mid // 4]), 1] += 1
        for i, event in enumerate(truth):
            if i not in tg:
                misses[category(event, preds)] += 1
            hits = [p for p in preds if overlap(event, p) > 0]
            fragments[str(len(hits))] += 1
        per_source.append(
            dict(
                source=source,
                predictions=len(preds),
                matched=len(pairs),
                missed=len(truth) - len(pairs),
                unmatched=len(preds) - len(pairs),
            )
        )
        all_p.append(p)
        all_y.append(r["dense"][::4])
        for threshold in thresholds:
            preds_t = frame_spans(detect(p, threshold, 3), n)
            sweeps[threshold].append(stream_stats(truth, preds_t, n))
    prob = np.concatenate(all_p)
    labels = np.concatenate(all_y)
    target = 1 - prob[:, 0]
    bins = []
    for lo, hi in zip(np.arange(5) / 5, np.arange(1, 6) / 5):
        take = (target >= lo) & ((target < hi) if hi < 1 else (target <= hi))
        if take.any():
            bins.append(
                dict(
                    lower=float(lo),
                    upper=float(hi),
                    n=int(take.sum()),
                    mean=float(target[take].mean()),
                    observed=float((labels[take] > 0).mean()),
                )
            )
    top = prob.argmax(1)
    mask = (top > 0) & (prob.max(1) >= 0.65)

    def offsets(data):
        x = np.array(data)
        return dict(
            matches=len(x),
            median_start_end=np.median(x, 0).tolist() if x.size else None,
            mean_start_end=np.mean(x, 0).tolist() if x.size else None,
        )

    return dict(
        false_alarms=dict(alarms),
        misses=dict(misses),
        false_alarm_midpoint_original_labels=dict(codes),
        intersecting_runs_per_true_event=dict(fragments),
        offsets_strict=offsets(offset),
        offsets_class_blind_10=offsets(loose_offset),
        edge_exposure_ticks_and_background_alarms_high_low=edges.tolist(),
        binary_target_population=float((labels > 0).mean()),
        binary_target_brier=float(np.mean((target - (labels > 0)) ** 2)),
        binary_calibration_bins=bins,
        binary_ece=float(
            sum(b["n"] * abs(b["mean"] - b["observed"]) for b in bins) / len(labels)
        ),
        accepted_target_ticks=dict(
            n=int(mask.sum()),
            correct=int((top[mask] == labels[mask]).sum()),
            background=int((labels[mask] == 0).sum()),
            mean_score=float(prob.max(1)[mask].mean()),
        ),
        threshold_sweep={
            str(t): dict(
                f1_50=aggregate(s)["segment_f1"][2], counts_50=aggregate(s)["counts"][2]
            )
            for t, s in sweeps.items()
        },
        per_source=per_source,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["val", "test"])
    split = parser.parse_args().split
    deterministic(17)
    if (
        split == "test"
        and hashes(LOCK_FILES)
        != json.loads((OUT / "selection.json").read_text())["sha256"]
    ):
        raise ValueError("Invalid selection lock")
    records = prepare(split)
    report = audit(records, split)
    report["direction_probes"] = probes(prepare("train"), records)
    dump(OUT / (split + "-failures.json"), report)
    print(
        {
            k: v
            for k, v in report.items()
            if k not in ("per_source", "binary_calibration_bins")
        }
    )


if __name__ == "__main__":
    main()

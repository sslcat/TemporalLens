"""Fit and lock on development data, then evaluate the exact locked pipeline."""

import argparse
import hashlib
import json
from itertools import product
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F

from .ipn_train import deterministic, load_split
from .stage_metrics import aggregate, frame_spans, stream_stats
from temporallens.gesture import ROOT, detect, padded, runs
from temporallens.two_stage import (
    OUT,
    classify,
    proposal_input,
    energy,
    fusion_input,
    intervals,
    labeled_intervals,
    linear_scores,
    pilot_states,
    proposal_trace,
)

KINDS = ("temporal", "flow", "fusion")
BASE = ROOT / "artifacts/ipn"
PINNED = {
    "cnn": BASE / "gesture.pt",
    "flow": BASE / "flow.pt",
    "manifest": BASE / "split.json",
}
LOCK_FILES = {**PINNED, "heads": OUT / "heads.pt"}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def hashes(files):
    digests = {}
    for key, path in files.items():
        data = path.read_bytes()
        if key == "manifest":
            # The published lock hashes the original Windows checkout of split.json.
            data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        digests[key] = hashlib.sha256(data).hexdigest()
    return digests


def prepare(split):
    records = load_split(split)
    head, flow = pilot_states("cuda")
    with torch.inference_mode():
        for r in records:
            r["visual"] = head(padded(r["x"]))[0]
            r["flow"] = linear_scores(r["motion"], flow)
            r["ref"] = runs(r["dense"])
    return records


def sample_bounds(a, b):
    return (a + 3) // 4, (b + 3) // 4


def fit_linear(x, y, classes, penalty, balanced=False):
    mean = x.mean(0)
    scale = x.std(0).clamp_min(0.015)
    x = (x - mean) / scale
    weight = torch.zeros((classes, x.shape[1]), device=x.device, requires_grad=True)
    bias = torch.zeros(classes, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=60, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        logits = F.linear(x, weight, bias)
        if balanced:
            pos_weight = (y == 0).sum() / (y == 1).sum()
            loss = F.binary_cross_entropy_with_logits(
                logits[:, 0], y.float(), pos_weight=pos_weight
            )
        else:
            loss = F.cross_entropy(logits, y)
        loss = loss + 0.5 * penalty * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return dict(mean=mean, scale=scale, weight=weight.detach(), bias=bias.detach())


def training_heads(train):
    labels = torch.cat([r["y"] > 0 for r in train])
    detectors = {}
    for method in ("binary", "hybrid"):
        features = torch.cat(
            [proposal_input(r["motion"], r["visual"], method) for r in train]
        )
        detectors[method] = fit_linear(features, labels, 1, 0.01, balanced=True)
    bouts = []
    labels = []
    for r in train:
        for k, a, b in r["ref"]:
            i, j = sample_bounds(a, b)
            bouts.append(fusion_input(r["visual"][i:j], r["flow"][i:j]))
            labels.append(k - 1)
    fusion = fit_linear(
        torch.stack(bouts), torch.tensor(labels, device="cuda"), 4, 0.05
    )
    return dict(**detectors, fusion=fusion)


def candidates(record, cfg, weights):
    trace = (
        proposal_trace(record["motion"], cfg, weights, record["visual"])
        .detach()
        .cpu()
        .numpy()
    )
    return intervals(trace, cfg["threshold"], cfg["gap"], cfg["minimum"])


def score_candidates(r, spans):
    n = len(r["dense"])
    ref = [(1, a, b) for _, a, b in r["ref"]]
    pred = frame_spans([(1, a, b) for a, b in spans], n)
    return stream_stats(ref, pred, n)


def labeled_record(r, spans, kind, cutoff, weights):
    n = len(r["dense"])
    classified = labeled_intervals(spans, kind, cutoff, r["visual"], r["flow"], weights)
    pred = frame_spans([(k, a, b) for k, a, b, _ in classified], n)
    cm = np.zeros((4, 4), int)
    for k, a, b in r["ref"]:
        i, j = sample_bounds(a, b)
        logits = classify(kind, i, j, r["visual"], r["flow"], weights)
        cm[k - 1, int(logits.argmax())] += 1
    return stream_stats(r["ref"], pred, n, cm), pred


def rank(metrics):
    return (
        metrics["segment_f1"][2],
        metrics["segment_f1"][1],
        -metrics["false_per_min"],
    )


def select(train, val, weights):
    train_energy = torch.cat([energy(r["motion"])[r["y"] == 0] for r in train])
    levels = torch.quantile(
        train_energy, torch.tensor([0.70, 0.80, 0.90, 0.95], device="cuda")
    ).tolist()
    grid = []
    best_per_method = {}
    for method, thresholds in [
        ("energy", levels),
        ("binary", [0.25, 0.40, 0.55, 0.70, 0.85]),
        ("hybrid", [0.25, 0.40, 0.55, 0.70, 0.85, 0.90, 0.95]),
    ]:
        for t, gap, minimum in product(thresholds, (1, 3, 6), (3, 6)):
            cfg = dict(detector=method, threshold=t, gap=gap, minimum=minimum)
            scores = aggregate(
                [score_candidates(r, candidates(r, cfg, weights)) for r in val]
            )
            grid.append(dict(cfg=cfg, criteria=list(rank(scores))))
            if (
                method not in best_per_method
                or rank(scores) > best_per_method[method][0]
            ):
                best_per_method[method] = rank(scores), cfg
    proposal = max(best_per_method.values(), key=lambda v: v[0])[1]
    spans = [candidates(r, proposal, weights) for r in val]
    choices = {}
    classifier_grid = []
    for kind in KINDS:
        for cutoff in (0.0, 0.45, 0.60, 0.75, 0.85):
            scores = aggregate(
                [
                    labeled_record(r, s, kind, cutoff, weights)[0]
                    for r, s in zip(val, spans)
                ]
            )
            classifier_grid.append(
                dict(kind=kind, cutoff=cutoff, criteria=list(rank(scores)))
            )
            if kind not in choices or rank(scores) > choices[kind][0]:
                choices[kind] = rank(scores), cutoff
    primary = max(choices, key=lambda k: choices[k][0])
    return (
        dict(
            proposal=proposal,
            energy_control=best_per_method["energy"][1],
            binary_control=best_per_method["binary"][1],
            hybrid_control=best_per_method["hybrid"][1],
            primary=primary,
            cutoffs={k: v[1] for k, v in choices.items()},
        ),
        dict(proposals=grid, classifiers=classifier_grid),
    )


def measure(records, cfg, weights):
    by_source = {}
    for r in records:
        n = len(r["dense"])
        spans = candidates(r, cfg["proposal"], weights)
        row = {"candidate": score_candidates(r, spans)}
        for name in ("energy", "binary", "hybrid"):
            row[name] = score_candidates(
                r, candidates(r, cfg[name + "_control"], weights)
            )
        details = {}
        for kind in KINDS:
            row[kind], details[kind] = labeled_record(
                r, spans, kind, cfg["cutoffs"][kind], weights
            )
        cm = row["temporal"]["oracle"]
        prob = r["visual"].softmax(-1).cpu().numpy()
        pred = frame_spans(detect(prob, 0.65, 3), n)
        row["baseline"] = stream_stats(r["ref"], pred, n, np.array(cm))
        by_source[r["row"]["source"]] = dict(
            participant=r["row"]["participant"],
            metrics=row,
            reference=r["ref"],
            candidates=frame_spans([(1, a, b) for a, b in spans], n),
            predictions=details,
            baseline=pred,
        )
    summary = {k: aggregate([s["metrics"][k] for s in by_source.values()]) for k in row}
    return dict(summary=summary, by_source=by_source)


def paired_interval(result, kind):
    sources = list(result["by_source"].values())
    groups = sorted({r["participant"] for r in sources})

    def counts(person, kind):
        return np.sum(
            [
                s["metrics"][kind]["counts"][-1]
                for s in sources
                if s["participant"] == person
            ],
            axis=0,
        )

    idx = np.random.default_rng(731).integers(len(groups), size=(2000, len(groups)))

    def f1(kind):
        tp, fp, fn = np.array([counts(g, kind) for g in groups])[idx].sum(1).T
        return 2 * tp / (2 * tp + fp + fn).clip(1)

    return np.percentile(f1(kind) - f1("baseline"), [2.5, 97.5]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["fit", "test", "replay"])
    args = parser.parse_args()
    deterministic(17)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    if args.phase == "test":
        cfg = json.loads((OUT / "selection.json").read_text())
        if hashes(LOCK_FILES) != cfg["sha256"]:
            raise ValueError("Frozen inputs or heads differ from the selection")
        result = measure(
            prepare("test"),
            cfg,
            torch.load(OUT / "heads.pt", map_location="cuda", weights_only=True),
        )
        result["paired_primary_minus_baseline_f1_50_ci95"] = paired_interval(
            result, cfg["primary"]
        )
        old = json.loads((BASE / "evaluation.json").read_text())["temporal"]["primary"]
        baseline = result["summary"]["baseline"]
        assert (
            old["cm"] == baseline["oracle_cm"]
            and old["segment_f1"] == baseline["segment_f1"]
        )
        dump(OUT / "test.json", result)
        print(
            {
                k: (v["segment_f1"], v["false_per_min"])
                for k, v in result["summary"].items()
            }
        )
        return
    train, val = prepare("train"), prepare("val")
    fit_start = perf_counter()
    weights = training_heads(train)
    torch.cuda.synchronize()
    fit_s = perf_counter() - fit_start
    cfg, grid = select(train, val, weights)
    metrics = measure(val, cfg, weights)
    if args.phase == "replay":
        locked = json.loads((OUT / "selection.json").read_text())
        old = torch.load(OUT / "heads.pt", map_location="cuda", weights_only=True)
        assert all(
            torch.equal(t, old[k][n])
            for k, head in weights.items()
            for n, t in head.items()
        )
        assert all(locked[k] == v for k, v in cfg.items())
        assert json.loads(json.dumps(metrics)) == json.loads(
            (OUT / "validation.json").read_text()
        )
        print("Exact weight, validation and model-selection replay")
        return
    torch.save(
        {k: {n: v.cpu() for n, v in head.items()} for k, head in weights.items()},
        OUT / "heads.pt",
    )
    cfg["sha256"] = hashes(LOCK_FILES)
    dump(OUT / "selection.json", cfg)
    dump(OUT / "validation.json", metrics)
    dump(OUT / "search.json", grid)
    dump(
        OUT / "compute.json",
        dict(
            gpu=torch.cuda.get_device_name(),
            fit_seconds=fit_s,
            development_seconds=perf_counter() - start,
            development_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            binary_detector_parameters=289,
            hybrid_detector_parameters=299,
            fusion_parameters=84,
        ),
    )
    print(cfg)
    print(
        {
            k: (v["segment_f1"], v["false_per_min"])
            for k, v in metrics["summary"].items()
        }
    )


if __name__ == "__main__":
    main()

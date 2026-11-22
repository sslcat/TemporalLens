"""Held-out measurements. No fitting, threshold selection or checkpoint updates."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .media import HZ, LABELS, decode
from .model import MotionHead, segments
from .train import sync


def matrix_metrics(cm):
    totals = cm.sum(-1)
    predicted = cm.sum(-2)
    diagonal = cm.diagonal(axis1=-2, axis2=-1)
    f1 = 2 * diagonal / np.maximum(1, totals + predicted)
    return diagonal.sum(-1) / np.maximum(1, totals.sum(-1)), f1.mean(-1), f1


def score(logits, truth):
    predicted = segments(logits.argmax(-1))
    true_events = segments(truth, 1)
    cm = np.zeros((4, 4), dtype=int)
    overlaps, errors = [], []
    for label, a, b in true_events:
        guess = logits[a:b, 1:].mean(0).argmax()
        cm[label - 1, guess] += 1
        candidates = []
        for k, c, d in predicted:
            if k == label:
                intersection = max(0, min(b, d) - max(a, c))
                candidates.append((intersection / (max(b, d) - min(a, c)), c, d))
        iou, c, d = max(candidates, default=(0, 0, 0))
        overlaps.append(iou)
        if iou > 0:
            errors.extend([abs(c - a) / HZ, abs(d - b) / HZ])
    expected = [k for k, _, _ in true_events]
    actual = [k for k, _, _ in predicted]
    return dict(
        cm=cm,
        exact=actual == expected,
        bag=sorted(actual) == sorted(expected),
        iou=np.mean(overlaps),
        errors=errors,
        predicted=predicted,
    )


def evaluate(name, device, x, y, manifest, boot):
    net = MotionHead(name == "temporal").to(device).eval()
    net.load_state_dict(
        torch.load(f"artifacts/{name}.pt", map_location=device, weights_only=True)
    )

    def infer(array):
        with torch.inference_mode():
            return (
                net(torch.tensor(np.ascontiguousarray(array), device=device))
                .cpu()
                .numpy()
            )

    clean = infer(x)
    scores = [score(logit, target) for logit, target in zip(clean, y)]
    cms = np.stack([s["cm"] for s in scores])
    cm = cms.sum(0)
    accuracy, macro, f1 = matrix_metrics(cm)
    rep_acc, rep_macro, _ = matrix_metrics(cms[boot].sum(1))
    exact = np.array([s["exact"] for s in scores])
    bag = np.array([s["bag"] for s in scores])
    iou = np.array([s["iou"] for s in scores])
    errors = [e for s in scores for e in s["errors"]]
    inv = np.array([0, 2, 1, 4, 3])
    reversed_logits = infer(x[:, ::-1])
    perms = np.stack(
        [
            np.random.default_rng(909 + i).permutation(len(row))
            for i, row in enumerate(x)
        ]
    )
    idx = np.arange(len(x))[:, None]
    shuffled = infer(x[idx, perms])[idx, perms.argsort(1)]

    def oracle_acc(logits, labels):
        return float(
            matrix_metrics(sum(score(p, t)["cm"] for p, t in zip(logits, labels)))[0]
        )

    # Exchange the two halves in the stationary gap. Each bout retains its direction.
    swapped_x, swapped_y = [], []
    for features, targets in zip(x, y):
        (_, _, b), (_, c, _) = segments(targets, 1)
        cut = (b + c) // 2
        swapped_x.append(np.roll(features, -cut, axis=0))
        swapped_y.append(np.roll(targets, -cut))
    swapped = [score(p, t) for p, t in zip(infer(np.stack(swapped_x)), swapped_y)]
    sensitivity = dict(
        clean=float(accuracy),
        reversed_original=oracle_acc(reversed_logits[:, ::-1], y),
        reversed_semantic=oracle_acc(reversed_logits, inv[y[:, ::-1]]),
        shuffled_original=oracle_acc(shuffled, y),
        swapped_bout_exact=np.mean([s["exact"] for s in swapped]),
    )
    latencies, pipeline = [], []
    sample = torch.from_numpy(x[:1]).to(device)
    with torch.inference_mode():
        for i in range(110):
            sync(device)
            start = perf_counter()
            net(sample)
            sync(device)
            if i >= 10:
                latencies.append(1000 * (perf_counter() - start))
        for row in manifest[:24]:
            t0 = perf_counter()
            features, _, _ = decode(row["path"])
            segments(infer(features[None])[0].argmax(-1))
            pipeline.append(1000 * (perf_counter() - t0))
    boot_values = dict(
        accuracy=rep_acc,
        macro_f1=rep_macro,
        exact=exact[boot].mean(1),
        iou=iou[boot].mean(1),
    )
    summary = dict(
        accuracy=float(accuracy),
        macro_f1=float(macro),
        exact=exact.mean(),
        bag=bag.mean(),
        iou=iou.mean(),
        endpoint_mae_s=np.mean(errors) if errors else None,
        matched_events=len(errors) // 2,
        confusion=cm.tolist(),
        per_class_f1=dict(zip(LABELS[1:], f1)),
        sensitivity=sensitivity,
        model_latency_ms_p50_p95=np.percentile(latencies, [50, 95]).tolist(),
        decode_model_ms_p50_p95=np.percentile(pipeline, [50, 95]).tolist(),
        ci95={k: interval(v) for k, v in boot_values.items()},
    )
    examples = [
        dict(scene=r["scene"], iou=s["iou"], exact=s["exact"], predicted=s["predicted"])
        for r, s in zip(manifest, scores)
    ]
    return summary, boot_values, examples


def interval(a):
    return np.percentile(a, [2.5, 97.5]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    device = torch.device(parser.parse_args().device)
    torch.set_num_threads(1)
    with np.load("work/test.npz") as data:
        x, y = data["x"], data["y"]
    manifest = [
        r
        for line in Path("artifacts/manifest.jsonl").read_text().splitlines()
        if (r := json.loads(line))["split"] == "test"
    ]
    boot = np.random.default_rng(622).integers(len(x), size=(2000, len(x)))
    report, reps, examples = {}, {}, {}
    for name in ("frame", "temporal"):
        report[name], reps[name], examples[name] = evaluate(
            name, device, x, y, manifest, boot
        )
    report["paired_delta_ci95"] = {
        k: interval(v - reps["frame"][k]) for k, v in reps["temporal"].items()
    }
    Path("artifacts/metrics.json").write_text(json.dumps(report, indent=2))
    Path("artifacts/predictions.json").write_text(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()

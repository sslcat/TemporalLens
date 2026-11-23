"""Validation-only study. Run with --replay to compare tensors and every numeric report."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from .ipn_train import deterministic
from .stage_study import fit_linear
from .rejection_data import load_development, OUT
from .rejection_diagnostics import describe, shift, conditional_metrics
from .rejection_metrics import (
    METHODS,
    calibrate,
    apply_rule,
    evaluate,
    reliability,
    softmax,
)
from temporallens.two_stage import ROOT, linear_scores


def split_arrays(rows):
    return (
        np.array([r["logits"] for r in rows]),
        np.array([r["label"] for r in rows]),
        np.array([r["person"] for r in rows]),
    )


def leave_person_out(people):
    for person in sorted(set(people)):
        scored = people == person
        yield person, ~scored, scored


def fit_gate(train, val_rows):
    rows, _, features, _, _ = train
    verified = [r["stratum"] == "verified_other" for r in rows]
    true = [r["stratum"] == "matched" for r in rows]

    def enough(mask):
        return (
            sum(mask) >= 16 and len({r["person"] for r, s in zip(rows, mask) if s}) >= 4
        )

    if not (
        enough(verified)
        and enough(true)
        and any(r["stratum"] == "verified_other" for r in val_rows)
    ):
        return None
    used = np.array(true) | np.array(verified)
    return fit_linear(
        torch.tensor(features[used], device="cuda"),
        torch.tensor(np.array(true)[used], device="cuda"),
        1,
        0.05,
        balanced=True,
    )


def presence_scores(logits, gate, prior, y, verified, has_gate):
    mask = (y > 0) | verified
    truth = y[mask] > 0
    inputs = {
        "explicit_other": 1 - softmax(logits)[:, 0],
        "constant_gate": np.full(len(y), prior),
    }
    if has_gate:
        inputs["learned_gate"] = gate
    result = {}
    for k, values in inputs.items():
        probs = values[mask]
        diff = probs[truth, None] - probs[None, ~truth]
        result[k] = dict(
            roc_auc=float(((diff > 0) + 0.5 * (diff == 0)).mean()),
            reliability=reliability(probs, truth),
            positive_quantiles=np.percentile(probs[truth], [10, 50, 90]).tolist(),
            negative_quantiles=np.percentile(probs[~truth], [10, 50, 90]).tolist(),
        )
    return result


def passes(metrics, baseline):
    a, b = metrics["candidate"], baseline["candidate"]
    return dict(
        two_more_correct=a["tp"] - b["tp"] >= 2,
        f1_gain=a["f1"] - b["f1"] >= 0.05,
        two_people_gain=sum(
            s["tp"] > baseline["people"][p]["tp"] for p, s in metrics["people"].items()
        )
        >= 2,
        within_false_alarm_budget=metrics["stream"]["false_per_min"]
        <= baseline["stream"]["false_per_min"] + 0.25,
    )


def encode_report(obj):
    return (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf8")


def run(replay):
    deterministic(17)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    manifest = json.loads((ROOT / "artifacts/ipn/split.json").read_text())
    allowed = {r["source"] for r in manifest if r["split"] in ("train", "val")}
    accessed = set()

    def audit(event, args):
        if (
            event == "open"
            and isinstance(args[0], str)
            and args[0].lower().endswith(".npz")
        ):
            source = Path(args[0]).stem
            if source not in allowed:
                raise PermissionError("Cache outside the development allowlist")
            accessed.add(source)

    sys.addaudithook(audit)
    train, val = load_development("train"), load_development("val")
    train_rows, train_sources, train_features, _, train_hash = train
    val_rows, val_sources, val_features, _, val_hash = val
    assert accessed == set(train_sources) | set(val_sources) == allowed
    assert not {r["person"] for r in train_rows} & {r["person"] for r in val_rows}
    gate_state = fit_gate(train, val_rows)
    train_gate = val_gate = None
    if gate_state is not None:

        def predict_presence(features):
            return (
                linear_scores(torch.tensor(features, device="cuda"), gate_state)
                .sigmoid()[:, 0]
                .cpu()
                .numpy()
                .astype(float)
            )

        train_gate, val_gate = (
            predict_presence(train_features),
            predict_presence(val_features),
        )
    train_logits, train_labels, _ = split_arrays(train_rows)
    val_logits, val_labels, val_people = split_arrays(val_rows)
    prior = float(
        np.sum(train_labels > 0)
        / sum(r["stratum"] != "partial_or_duplicate" for r in train_rows)
    )
    val_metrics = {}
    decisions = {}
    folds = {}
    conditional = {}
    for method in METHODS:
        if method == "learned_gate" and gate_state is None:
            continue
        pred = np.zeros(len(val_labels), int)
        conf = np.zeros(len(val_labels))
        ranking = np.zeros(len(val_labels))
        eligible = np.zeros(len(val_labels), bool)
        temperatures = np.ones(len(val_labels))
        folds[method] = {}
        for person, fit, scored in leave_person_out(val_people):
            cfg = calibrate(
                method,
                val_logits[fit],
                val_labels[fit],
                val_gate[fit] if val_gate is not None else None,
                prior,
            )
            fold_out = apply_rule(
                method,
                cfg,
                val_logits[scored],
                val_gate[scored] if val_gate is not None else None,
                prior,
            )
            pred[scored], conf[scored], ranking[scored], eligible[scored] = fold_out
            folds[method][person] = cfg
            temperatures[scored] = cfg["temperature"]
        val_metrics[method] = evaluate(
            val_rows, val_sources, pred, conf, ranking, eligible
        )
        decisions[method] = dict(
            predicted=pred.tolist(),
            target_score=conf.tolist(),
            eligible=eligible.tolist(),
            ranking=ranking.tolist(),
        )
        if method in ("released", "temperature"):
            conditional[method] = conditional_metrics(
                val_logits, val_labels, temperatures
            )
    prev = json.loads((ROOT / "artifacts/two-stage/validation.json").read_text())[
        "by_source"
    ]
    for source in val_sources:
        computed = [
            [k, r["start"], r["end"]]
            for r, k in zip(val_rows, decisions["released"]["predicted"])
            if k and r["source"] == source
        ]
        assert computed == prev[source]["predictions"]["temporal"]
    criteria = {
        k: passes(m, val_metrics["released"])
        for k, m in val_metrics.items()
        if k != "released"
    }
    passing = [k for k, v in criteria.items() if all(v.values())]
    decision = dict(criteria=criteria, selected=None, status="retain_original")
    if passing:
        best = max(
            passing,
            key=lambda k: (
                val_metrics[k]["candidate"]["f1"],
                -val_metrics[k]["candidate"]["fp"],
            ),
        )
        decision.update(
            selected=best,
            status="freeze_for_independent_evaluation",
            full_validation_config=calibrate(
                best, val_logits, val_labels, val_gate, prior
            ),
        )
    train_metrics = {}
    for method in [
        "released",
        "no_rejection",
        "explicit_other",
        "constant_gate",
        "learned_gate",
    ]:
        if method == "learned_gate" and train_gate is None:
            continue
        cfg = dict(temperature=1.0, cutoff=0.0 if method == "no_rejection" else 0.60)
        outputs = apply_rule(method, cfg, train_logits, train_gate, prior)
        m = evaluate(train_rows, train_sources, *outputs)
        train_metrics[method] = {k: v for k, v in m.items() if k != "curve"}
    presence = {}
    for split, x, y, g, rows in [
        ("train", train_logits, train_labels, train_gate, train_rows),
        ("val", val_logits, val_labels, val_gate, val_rows),
    ]:
        presence[split] = presence_scores(
            x,
            g,
            prior,
            y,
            np.array([r["stratum"] == "verified_other" for r in rows]),
            gate_state is not None,
        )
    diagnostics = dict(
        train=describe(train_rows),
        val=describe(val_rows),
        shift=shift(train, val),
        presence=presence,
        conditional_validation=conditional,
        conditional_train=conditional_metrics(
            train_logits, train_labels, np.ones(len(train_labels))
        ),
    )
    results = dict(
        pipeline_base="e91a55fa1a930522a873410da2255805d588a75f",
        inputs=dict(train=train_hash, val=val_hash, cache_sources=sorted(accessed)),
        gate_fitted=gate_state is not None,
        training_prior=prior,
        fold_configs=folds,
        validation=val_metrics,
        train_fixed_cutoffs=train_metrics,
        decision=decision,
    )
    reports = {
        "results.json": results,
        "root-causes.json": diagnostics,
        "candidates.json": dict(
            train=train_rows, val=val_rows, validation_decisions=decisions
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in reports.items():
        path = OUT / name
        data = encode_report(value)
        if replay:
            assert path.read_bytes().replace(b"\r\n", b"\n") == data, name
        else:
            path.write_bytes(data)
    if gate_state is not None:
        path = OUT / "diagnostic-gate.pt"
        state = {k: v.cpu() for k, v in gate_state.items()}
        if replay:
            old = torch.load(path, weights_only=True, map_location="cpu")
            assert old.keys() == state.keys() and all(
                torch.equal(v, old[k]) for k, v in state.items()
            )
        else:
            torch.save(state, path)
    if not replay:
        compute = dict(
            seconds=perf_counter() - started,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            gpu=torch.cuda.get_device_name(),
        )
        (OUT / "compute.json").write_bytes(encode_report(compute))
    print(
        "replay" if replay else "run",
        decision["status"],
        "development caches",
        len(accessed),
        "validation proposal digest",
        val_hash,
    )
    for k, m in val_metrics.items():
        print(k, m["candidate"], m["stream"]["f1"], m["target_correctness"]["ece"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    run(parser.parse_args().replay)

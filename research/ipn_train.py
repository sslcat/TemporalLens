"""Fit on train/validation only. Held-out files are loaded by ipn_evaluate."""

import argparse
import copy
import json
import os
import subprocess
import threading
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ipn_ingest import OUT, WORK
from .ipn_metrics import source_score, summarize
from temporallens.gesture import GestureHead, HALO, padded

SEEDS = (17, 31, 73)
RATES = (0.0005, 0.0015)
KINDS = ("frame", "pooled", "temporal")
GRID = [(t, m) for t in (0.65, 0.8, 0.9) for m in (2, 3)]
FITS = WORK / "fits"


def deterministic(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def load_split(split):
    result = []
    for row in json.loads((OUT / "split.json").read_text()):
        if row["split"] != split:
            continue
        with np.load(WORK / "cache" / (row["source"] + ".npz")) as f:
            data = {
                k: torch.tensor(f[k], device="cuda", dtype=torch.float32)
                for k in ("x", "motion")
            }
            data.update(
                y=torch.tensor(f["y"], device="cuda"), dense=f["dense"], row=row
            )
        result.append(data)
    return result


class FlowLogit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(144, 5)
        self.register_buffer("mean", torch.zeros(144))
        self.register_buffer("scale", torch.ones(144))

    def forward(self, x):
        return self.linear((x - self.mean) / self.scale)


def predict(model, kind, record):
    with torch.inference_mode():
        logits = (
            model(record["motion"]) if kind == "flow" else model(padded(record["x"]))[0]
        )
        return logits


def validation(model, kind, records, weights):
    logits = [predict(model, kind, r) for r in records]
    loss = F.cross_entropy(
        torch.cat(logits), torch.cat([r["y"] for r in records]), weight=weights
    ).item()
    probs = [z.softmax(-1).cpu().numpy() for z in logits]
    return loss, probs


def choose_decoder(probs, records, grid=GRID):
    best = None
    for threshold, minimum in grid:
        s = summarize(
            [
                source_score(p, r["dense"], threshold, minimum)
                for p, r in zip(probs, records)
            ]
        )
        key = (s["segment_f1"][2], -s["false_alarms"], threshold)
        if best is None or key > best[0]:
            best = (key, threshold, minimum)
    return best[0][0], best[1], best[2]


def chunks(train):
    xs, ys = [], []
    for r in train:
        x, y = r["x"], r["y"]
        for i in range(0, len(y), 90):
            ids = torch.arange(i - HALO, i + 90 + HALO, device=x.device).clamp(
                0, len(y) - 1
            )
            xs.append(x[ids])
            ys.append(F.pad(y[i : i + 90], (0, 90 - len(y[i : i + 90])), value=-100))
    return torch.stack(xs), torch.stack(ys)


def fit(kind, seed, lr, train, val, weights, batches, output_dir=FITS):
    deterministic(seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    model = (FlowLogit() if kind == "flow" else GestureHead(kind)).cuda()
    feat = "motion" if kind == "flow" else "x"
    x = torch.cat([r[feat] for r in train])
    model.mean.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp_min(0.015))
    history = []
    if kind == "flow":
        y = torch.cat([r["y"] for r in train])
        opt = torch.optim.LBFGS(
            model.parameters(), lr=1, max_iter=60, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            loss = (
                F.cross_entropy(model(x), y, weight=weights)
                + lr * model.linear.weight.square().sum() / 2
            )
            loss.backward()
            return loss

        opt.step(closure)
        train_loss = F.cross_entropy(model(x), y, weight=weights).item()
        vl, probs = validation(model, kind, val, weights)
        quality, threshold, minimum = choose_decoder(probs, val)
        history.append([1, train_loss, vl, quality])
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 1
    else:
        del x
        xx, yy = batches
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        order = torch.Generator().manual_seed(seed)
        best = (-1, float("-inf"))
        for epoch in range(1, 31):
            loss_sum = weight_sum = 0.0
            model.train()
            for ix in torch.randperm(len(xx), generator=order).split(16):
                opt.zero_grad(set_to_none=True)
                logits = model(xx[ix]).reshape(-1, 5)
                target = yy[ix].reshape(-1)
                loss = F.cross_entropy(logits, target, weight=weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                mass = weights[target[target != -100]].sum().item()
                loss_sum += loss.item() * mass
                weight_sum += mass
            model.eval()
            vl, probs = validation(model, kind, val, weights)
            quality, _, _ = choose_decoder(probs, val, [(0.65, 2)])
            history.append([epoch, loss_sum / weight_sum, vl, quality])
            if (quality, -vl) > best:
                best = (quality, -vl)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= 7:
                break
        model.load_state_dict(best_state)
        _, probs = validation(model, kind, val, weights)
        quality, threshold, minimum = choose_decoder(probs, val)
    torch.cuda.synchronize()
    state = dict(
        kind=kind,
        weights={k: v.cpu() for k, v in best_state.items()},
        threshold=threshold,
        minimum=minimum,
    )
    path = output_dir / f"{kind}-{seed}-{lr}.pt"
    torch.save(state, path)
    return dict(
        kind=kind,
        seed=seed,
        lr=lr,
        val_f1=quality,
        threshold=threshold,
        minimum=minimum,
        best_epoch=best_epoch,
        history=history,
        seconds=perf_counter() - t0,
        peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        bytes=path.stat().st_size,
        parameters=sum(p.numel() for p in model.parameters()),
        path=path.as_posix(),
    )


def gpu_monitor(stop, samples):
    while not stop.is_set():
        data = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        samples.append(int(data.strip().splitlines()[0]))
        stop.wait(0.7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    FITS.mkdir(exist_ok=True)
    train, val = load_split("train"), load_split("val")
    labels = torch.cat([r["y"] for r in train])
    weights = (labels.numel() / labels.bincount()).sqrt()
    batches = chunks(train)
    if args.replay:
        report = json.loads((OUT / "training.json").read_text())
        output_dir = WORK / "replay"
        output_dir.mkdir(exist_ok=True)
        for kind, settings in report["selected"].items():
            for row in settings:
                expected = torch.load(row["path"], weights_only=True)
                rep = fit(
                    kind,
                    row["seed"],
                    row["lr"],
                    train,
                    val,
                    weights,
                    batches,
                    output_dir,
                )
                actual = torch.load(rep["path"], weights_only=True)
                assert all(
                    torch.equal(v, actual["weights"][k])
                    for k, v in expected["weights"].items()
                )
                assert rep["history"] == row["history"]
                assert (rep["threshold"], rep["minimum"]) == (
                    row["threshold"],
                    row["minimum"],
                )
                print(
                    "Exact tensor, decoder and history replay",
                    kind,
                    row["seed"],
                    flush=True,
                )
        return
    samples = []
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(stop, samples), daemon=True)
    monitor.start()
    selected = {}
    all_runs = []
    try:
        for kind in (*KINDS, "flow"):
            groups = []
            for lr in (0.001, 0.01) if kind == "flow" else RATES:
                group = []
                for seed in (17,) if kind == "flow" else SEEDS:
                    row = fit(kind, seed, lr, train, val, weights, batches)
                    group.append(row)
                    all_runs.append(row)
                    print(
                        kind,
                        lr,
                        seed,
                        "epoch",
                        row["best_epoch"],
                        "val",
                        round(row["val_f1"], 3),
                        flush=True,
                    )
                groups.append(group)
            chosen = max(groups, key=lambda g: np.mean([r["val_f1"] for r in g]))
            selected[kind] = chosen
            state = torch.load(chosen[0]["path"], weights_only=True)
            torch.save(
                state, OUT / ("gesture.pt" if kind == "temporal" else kind + ".pt")
            )
    finally:
        stop.set()
        monitor.join()
    report = dict(
        selected=selected,
        runs=all_runs,
        device=torch.cuda.get_device_name(),
        training_seconds=sum(r["seconds"] for r in all_runs),
        gpu_util_percent=samples,
        class_weights=weights.cpu().tolist(),
    )
    (OUT / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

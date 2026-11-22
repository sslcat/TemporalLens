import argparse
import json
import os
from pathlib import Path
import platform
import random
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from .model import MotionHead

SEED = 773


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def fit(name, device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    splits = []
    for split in ("train", "val"):
        with np.load(f"work/{split}.npz") as data:
            splits.append(
                tuple(torch.from_numpy(data[key]).to(device) for key in ("x", "y"))
            )
    (x, y), (vx, vy) = splits
    weights = y.numel() / (5 * y.flatten().bincount())
    model = MotionHead(name == "temporal").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    history = []
    best = float("inf")
    checkpoint = Path(f"artifacts/{name}.pt")
    sync(device)
    t0 = perf_counter()
    # Dedicated CPU RNG makes minibatch membership identical across heads.
    order = torch.Generator().manual_seed(SEED)
    for epoch in range(1, 141):
        model.train()
        for batch in torch.randperm(len(x), generator=order).split(32):
            opt.zero_grad(set_to_none=True)
            loss = cross_entropy(
                model(x[batch]).reshape(-1, 5), y[batch].reshape(-1), weight=weights
            )
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            train_loss = cross_entropy(
                model(x).reshape(-1, 5), y.reshape(-1), weight=weights
            ).item()
            val_loss = cross_entropy(
                model(vx).reshape(-1, 5), vy.reshape(-1), weight=weights
            ).item()
        history.append([epoch, train_loss, val_loss])
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.replace(checkpoint)
    sync(device)
    return dict(
        history=history,
        selected_epoch=best_epoch,
        seconds=perf_counter() - t0,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    device = torch.device(parser.parse_args().device)
    torch.set_num_threads(1)
    results = dict(
        device=torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        python=platform.python_version(),
        seed=SEED,
    )
    for name in ("frame", "temporal"):
        results[name] = fit(name, device)
        print(name, results[name]["seconds"], flush=True)
    Path("artifacts/training.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

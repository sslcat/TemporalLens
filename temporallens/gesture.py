"""Shared frozen image preprocessing, gesture heads and segment decoding."""

from pathlib import Path
import os

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

LABELS = ("Other", "Throw left", "Throw right", "Zoom in", "Zoom out")
RATE = 7.5
HALO = 7
ROOT = Path(__file__).resolve().parents[1]
ENCODER_REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def encoder_dir():
    return Path(
        os.environ.get(
            "TEMPORALLENS_ENCODER", ROOT / "work/real-feasibility/dinov2-small"
        )
    )


class FrozenEncoder:
    def __init__(self, device):
        from transformers import Dinov2Model

        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = (
            Dinov2Model.from_pretrained(
                encoder_dir(), local_files_only=True, use_safetensors=True
            )
            .eval()
            .requires_grad_(False)
            .to(device=self.device, dtype=self.dtype)
        )

    @torch.inference_mode()
    def __call__(self, frames):
        output = []
        for start in range(0, len(frames), 32):
            pixels = []
            for rgb in frames[start : start + 32]:
                image = ImageOps.pad(
                    Image.fromarray(rgb),
                    (224, 224),
                    method=Image.Resampling.BICUBIC,
                    color=(124, 116, 104),
                )
                pixels.append((np.asarray(image).astype(np.float32) / 255 - MEAN) / STD)
            batch = (
                torch.from_numpy(np.stack(pixels))
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(self.device, self.dtype)
            )
            patches = (
                self.model(batch)
                .last_hidden_state[:, 1:]
                .transpose(1, 2)
                .reshape(-1, 384, 16, 16)
            )
            output.append(
                F.adaptive_avg_pool2d(patches, (2, 2)).flatten(1).cpu().float().numpy()
            )
        return np.concatenate(output)


class GestureHead(nn.Module):
    """Input includes a seven-tick halo on either side of the requested range."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("mean", torch.zeros(1536))
        self.register_buffer("scale", torch.ones(1536))
        self.project = nn.Conv1d(1536, 64, 1)
        self.blocks = nn.ModuleList()
        for d in (1, 2, 4):
            if kind == "temporal":
                self.blocks.append(nn.Conv1d(64, 64, 3, dilation=d))
            else:
                self.blocks.append(
                    nn.Sequential(nn.Conv1d(64, 96, 1), nn.ReLU(), nn.Conv1d(96, 64, 1))
                )
        self.out = nn.Conv1d(64, 5, 1)

    def forward(self, x):
        x = ((x - self.mean) / self.scale).transpose(1, 2)
        if self.kind == "pooled":
            x = F.avg_pool1d(x, 15, stride=1)
        elif self.kind == "frame":
            x = x[:, :, HALO:-HALO]
        x = F.relu(self.project(x))
        for d, block in zip((1, 2, 4), self.blocks):
            center = x[:, :, d:-d] if self.kind == "temporal" else x
            x = F.relu(center + block(x))
        return self.out(x).transpose(1, 2)


def padded(x):
    return torch.cat((x[:1].expand(HALO, -1), x, x[-1:].expand(HALO, -1)))[None]


def runs(ids):
    cuts = np.flatnonzero(np.diff(ids)) + 1
    return [
        (int(ids[a]), int(a), int(b))
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ids)])
        if ids[a]
    ]


def detect(prob, threshold, minimum):
    ids = prob.argmax(1)
    ids[prob.max(1) < threshold] = 0
    return [(k, a, b) for k, a, b in runs(ids) if b - a >= minimum]


class GesturePredictor:
    def __init__(self, device):
        self.encoder = FrozenEncoder(device)
        state = torch.load(
            ROOT / "artifacts/ipn/gesture.pt", map_location=device, weights_only=True
        )
        self.head = GestureHead(state["kind"]).to(device).eval()
        self.head.load_state_dict(state["weights"])
        self.device = device
        self.threshold, self.minimum = state["threshold"], state["minimum"]

    @torch.inference_mode()
    def analyze(self, frames, duration):
        if all(np.array_equal(frames[0], rgb) for rgb in frames[1:]):
            return dict(
                duration=duration, device=self.device, supported=True, events=[]
            )
        x = torch.from_numpy(self.encoder(frames)).to(self.device)
        prob = self.head(padded(x))[0].softmax(-1).cpu().numpy()
        return dict(
            duration=duration,
            device=self.device,
            supported=True,
            events=[
                dict(
                    label=LABELS[k],
                    start=a / RATE,
                    end=min(b / RATE, duration),
                    score=float(prob[a:b, k].mean()),
                    uncertain=bool(prob[a:b, k].mean() < 0.7),
                )
                for k, a, b in detect(prob, self.threshold, self.minimum)
            ],
        )

"""Offline candidate generation and interval classification on sampled video."""

import json
import numpy as np
import torch
from torch.nn import functional as F

from .gesture import FrozenEncoder, GestureHead, LABELS, RATE, ROOT, padded, runs

OUT = ROOT / "artifacts/two-stage"


def smooth(x, width):
    padded_x = F.pad(x.T[None], (width // 2, width // 2), mode="replicate")
    return F.avg_pool1d(padded_x, width, stride=1)[0].T


def detector_input(motion):
    # Short- and long-window flow histograms and signed cell velocities.
    return torch.cat([smooth(motion, 5), smooth(motion, 15)], dim=1)


def energy(motion):
    return smooth(motion[:, 3::12].mean(1, keepdim=True), 5)[:, 0]


def linear_scores(x, state):
    return F.linear(
        (x - state["mean"]) / state["scale"], state["weight"], state["bias"]
    )


def proposal_input(motion, visual, method):
    features = detector_input(motion)
    if method == "hybrid":
        return torch.cat([features, smooth(visual, 5), smooth(visual, 15)], 1)
    return features


def proposal_trace(motion, cfg, weights, visual=None):
    if cfg["detector"] == "energy":
        return energy(motion)
    method = cfg["detector"]
    return linear_scores(
        proposal_input(motion, visual, method), weights[method]
    ).sigmoid()[:, 0]


def intervals(trace, threshold, gap, minimum):
    active = np.asarray(trace) >= threshold
    merged = []
    for _, a, b in runs(active.astype(int)):
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= minimum]


def fusion_input(visual, flow):
    t = len(flow)
    scores = torch.cat([visual, flow], dim=1)
    return torch.cat([scores[: (t + 1) // 2].mean(0), scores[t // 2 :].mean(0)])


def classify(kind, left, right, visual, flow, weights):
    if kind == "fusion":
        descriptor = fusion_input(visual[left:right], flow[left:right])
        return linear_scores(descriptor, weights["fusion"])
    logits = visual if kind == "temporal" else flow
    return logits[left:right, 1:].mean(0)


def labeled_intervals(candidates, kind, cutoff, visual, flow, weights):
    predictions = []
    for a, b in candidates:
        prob = classify(kind, a, b, visual, flow, weights).softmax(0)
        value, label = prob.max(0)
        if value.item() >= cutoff:
            predictions.append((int(label) + 1, a, b, value.item()))
    return predictions


def pilot_states(device, visual=True):
    pilot = ROOT / "artifacts/ipn"
    flow = torch.load(pilot / "flow.pt", map_location=device, weights_only=True)[
        "weights"
    ]
    flow = dict(
        mean=flow["mean"],
        scale=flow["scale"],
        weight=flow["linear.weight"],
        bias=flow["linear.bias"],
    )
    head = None
    if visual:
        head = GestureHead("temporal").to(device).eval()
        head.load_state_dict(
            torch.load(pilot / "gesture.pt", map_location=device, weights_only=True)[
                "weights"
            ]
        )
    return head, flow


class TwoStagePredictor:
    def __init__(self, device):
        self.device = device
        lock = json.loads((OUT / "selection.json").read_text())
        self.cfg = lock["proposal"]
        self.kind = lock["primary"]
        self.cutoff = lock["cutoffs"][self.kind]
        self.weights = torch.load(
            OUT / "heads.pt", weights_only=True, map_location=device
        )
        self.head, self.flow = pilot_states(
            device, self.kind != "flow" or self.cfg["detector"] == "hybrid"
        )
        self.encoder = FrozenEncoder(device) if self.head is not None else None

    @torch.inference_mode()
    def analyze(self, frames, duration):
        from .flow_features import flow_sequence, gray

        result = dict(
            duration=duration,
            device=self.device,
            supported=True,
            events=[],
            candidates=[],
        )
        if all(np.array_equal(frames[0], im) for im in frames[1:]):
            return result
        motion = torch.as_tensor(
            flow_sequence([gray(im) for im in frames]), device=self.device
        )
        visual = None
        if self.encoder is not None:
            x = torch.from_numpy(self.encoder(frames)).to(self.device)
            visual = self.head(padded(x))[0]
        trace = proposal_trace(motion, self.cfg, self.weights, visual).cpu().numpy()
        spans = intervals(
            trace, self.cfg["threshold"], self.cfg["gap"], self.cfg["minimum"]
        )
        result["candidates"] = [
            dict(start=a / RATE, end=min(b / RATE, duration)) for a, b in spans
        ]
        if not spans:
            return result
        flow = linear_scores(motion, self.flow)
        for k, a, b, score in labeled_intervals(
            spans, self.kind, self.cutoff, visual, flow, self.weights
        ):
            result["events"].append(
                dict(
                    label=LABELS[k],
                    start=a / RATE,
                    end=min(b / RATE, duration),
                    score=score,
                    uncertain=score < 0.7,
                )
            )
        return result

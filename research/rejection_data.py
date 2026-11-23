"""Extract proposals and diagnostic labels from the two development splits only."""

import csv
from collections import Counter, defaultdict
import hashlib
import json

import numpy as np
import torch

from temporallens.gesture import ROOT
from temporallens.two_stage import OUT as PRIOR, fusion_input
from .ipn_ingest import META
from .ipn_metrics import match, overlap
from .stage_study import candidates, prepare, sample_bounds, hashes, LOCK_FILES

OUT = ROOT / "artifacts/rejection"
TARGET_CODES = {"G05": 1, "G06": 2, "G10": 3, "G11": 4}


def official_labels(allowed):
    annotations = defaultdict(list)
    with (META / "Annot_List.txt").open(newline="") as file:
        for row in csv.DictReader(file):
            if row["video"] in allowed:
                annotations[row["video"]].append(
                    (int(row["t_start"]) - 1, int(row["t_end"]), row["label"])
                )
    return annotations


def verify_timeline(annotations, dense):
    end = 0
    for a, b, code in annotations:
        if (
            a != end
            or b <= a
            or not code
            or not np.all(dense[a:b] == TARGET_CODES.get(code, 0))
        ):
            raise ValueError("Original annotations disagree with cached frame labels")
        end = b
    if end != len(dense):
        raise ValueError("Original annotation coverage is incomplete")


def proposal_labels(ref, spans):
    matched = match([(1, a, b) for _, a, b in ref], [(1, a, b) for a, b in spans], 0.5)
    return {j: i for i, j in matched}


def load_development(split):
    if split not in ("train", "val"):
        raise ValueError("The rejection study accepts only train and val")
    lock = json.loads((PRIOR / "selection.json").read_text())
    if hashes(LOCK_FILES) != lock["sha256"]:
        raise ValueError("Frozen detector inputs have changed")
    weights = torch.load(PRIOR / "heads.pt", weights_only=True, map_location="cuda")
    records = prepare(split)
    annotations = official_labels({r["row"]["source"] for r in records})
    rows = []
    sources = {}
    descriptors = []
    embeddings = []
    for r in records:
        row = r["row"]
        source = row["source"]
        person = row["participant"]
        dense = r["dense"]
        ann = annotations[source]
        verify_timeline(ann, dense)
        ticks = candidates(r, lock["proposal"], weights)
        spans = [(4 * i, min(4 * j, len(dense))) for i, j in ticks]
        ref = r["ref"]
        assignment = proposal_labels(ref, spans)
        sources[source] = dict(person=person, frames=len(dense), ref=ref, spans=spans)
        for idx, ((i, j), (a, b)) in enumerate(zip(ticks, spans)):
            best = max(overlap((0, a, b), e) for e in ref)
            ref_index = assignment.get(idx)
            k = ref[ref_index][0] if ref_index is not None else 0
            codes = Counter()
            for u, v, code in ann:
                count = max(0, min(b, v) - max(a, u))
                if count:
                    codes[code] += count
            if sum(codes.values()) != b - a:
                raise ValueError("Incomplete candidate annotation")
            verified = all(code not in TARGET_CODES for code in codes)
            if verified != (best == 0):
                raise ValueError("Original interval and proposal mismatch")
            group = (
                "matched"
                if k
                else "verified_other"
                if verified
                else "partial_or_duplicate"
            )
            visual = r["visual"][i:j]
            flow = r["flow"][i:j]
            entry = dict(
                source=source,
                person=person,
                start=a,
                end=b,
                label=k,
                stratum=group,
                best_iou=best,
                original_annotation_frames=dict(codes),
                logits=visual.mean(0).tolist(),
                flow_logits=flow.mean(0).tolist(),
            )
            if k:
                _, u, v = ref[ref_index]
                left, right = sample_bounds(u, v)
                entry.update(
                    oracle=r["visual"][left:right].mean(0).tolist(),
                    reference=[u, v],
                    iou=overlap((0, a, b), (0, u, v)),
                )
            rows.append(entry)
            descriptors.append(fusion_input(visual, flow).cpu().numpy())
            embeddings.append(r["x"][i:j].mean(0).cpu().numpy())
    if split == "val":
        old = json.loads((PRIOR / "validation.json").read_text())["by_source"]
        for s, record in sources.items():
            assert [[1, a, b] for a, b in record["spans"]] == old[s]["candidates"]
    signature = hashlib.sha256(
        json.dumps({s: v["spans"] for s, v in sources.items()}, sort_keys=True).encode()
    ).hexdigest()
    return rows, sources, np.stack(descriptors), np.stack(embeddings), signature

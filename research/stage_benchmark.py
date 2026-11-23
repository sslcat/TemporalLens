"""Paired local decode + inference timing on the same prior 10-second QA upload."""

import gc
import json
from time import perf_counter
import numpy as np
import torch
from .ipn_ingest import WORK
from .stage_study import dump
from temporallens.gesture import GesturePredictor
from temporallens.media import decode
from temporallens.two_stage import OUT, TwoStagePredictor


def main():
    path = WORK / "ui/gesture.mp4"
    report = {}
    for name, cls in (("baseline", GesturePredictor), ("two_stage", TwoStagePredictor)):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        model = cls("cuda")
        torch.cuda.synchronize()
        init = perf_counter() - start
        samples = []
        for repeat in range(11):
            start = perf_counter()
            frames, _, duration = decode(path, gesture_frames=True)
            response = model.analyze(frames, duration)
            torch.cuda.synchronize()
            if repeat:
                samples.append(1000 * (perf_counter() - start))
        report[name] = dict(
            duration=duration,
            events=len(response["events"]),
            model_init_seconds=init,
            warm_decode_infer_ms=samples,
            median_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)),
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
        del model
    dump(OUT / "inference.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

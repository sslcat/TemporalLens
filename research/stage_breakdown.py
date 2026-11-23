"""Count what happens to correctly localized proposals with a fixed assignment."""

import json
from collections import Counter
from .ipn_metrics import match
from .stage_metrics import aggregate, stream_stats
from .stage_study import dump
from temporallens.two_stage import OUT


def main():
    report = {}
    old_candidates = {}
    for split in ("validation", "test"):
        sources = json.loads((OUT / (split + ".json")).read_text())["by_source"]
        stats = {
            k: Counter(matched=0, correct=0, wrong=0, rejected=0)
            for k in ("temporal", "flow", "fusion")
        }
        blind = []
        for s in sources.values():
            truth = s["reference"]
            proposals = s["candidates"]
            blind.append(
                stream_stats(
                    [(1, a, b) for _, a, b in truth],
                    [(1, a, b) for _, a, b in s["baseline"]],
                    s["metrics"]["baseline"]["frames"],
                )
            )
            pairs = match([(1, a, b) for _, a, b in truth], proposals, 0.50)
            for kind, count in stats.items():
                retained = {(a, b): k for k, a, b in s["predictions"][kind]}
                for i, j in pairs:
                    _, a, b = proposals[j]
                    label = retained.get((a, b))
                    key = (
                        "rejected"
                        if label is None
                        else "correct"
                        if label == truth[i][0]
                        else "wrong"
                    )
                    count[key] += 1
                    count["matched"] += 1
        report[split] = stats
        summary = aggregate(blind)
        old_candidates[split] = {
            k: summary[k]
            for k in (
                "counts",
                "recall",
                "false_per_min",
                "endpoint_mae_frames",
                "segment_f1",
            )
        }
    dump(OUT / "candidate-classification.json", report)
    dump(OUT / "old-candidates.json", old_candidates)
    print(report, old_candidates)


if __name__ == "__main__":
    main()

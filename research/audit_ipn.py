"""Summarize original IPN Hand annotations without loading any image data."""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

TARGETS = {"G05", "G06", "G10", "G11"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(folder):
    for name in ("Annot_List.txt", "metadata.csv", "Video_TrainList.txt", "Video_TestList.txt"):
        print(name, hashlib.sha256((folder / name).read_bytes()).hexdigest())

    metadata = read_csv(folder / "metadata.csv")
    meta = {r["Video Name"]: r for r in metadata}
    if len(meta) != len(metadata):
        raise ValueError("Duplicate metadata key")
    events = defaultdict(list)
    for row in read_csv(folder / "Annot_List.txt"):
        start, end = int(row["t_start"]), int(row["t_end"])
        if end < start or end - start + 1 != int(row["frames"]):
            raise ValueError(f"Invalid interval in {row['video']}")
        events[row["video"]].append((start, end, row["label"]))
    if set(events) != set(meta):
        raise ValueError("Annotation and metadata video IDs differ")

    print("Total annotated frames", sum(v[-1][1] for v in events.values()))
    print("Metadata minus final annotated frame", Counter(
        int(meta[v]["Frames"]) - intervals[-1][1] for v, intervals in events.items()
    ))
    listed = set()
    for split in ("train", "test"):
        names = []
        for line in (folder / f"Video_{split.title()}List.txt").read_text().splitlines():
            video, count = line.split()
            if video in listed or meta[video]["Set"] != split or meta[video]["Frames"] != count:
                raise ValueError(f"Split or length mismatch for {video}")
            listed.add(video)
            names.append(video)
        label_counts, pairs, tiles = Counter(), Counter(), Counter()
        eligible_videos = set()
        for name in names:
            timeline = events[name]
            if timeline[0][0] != 1 or any(b + 1 != c for (_, b, _), (c, _, _) in pairwise(timeline)):
                raise ValueError(f"Gap, overlap or unsorted rows for {name}")
            targets = [e for e in timeline if e[2] in TARGETS]
            label_counts.update(e[2] for e in targets)
            for left, right in pairwise(targets):
                if left[2] != right[2] and right[1] - left[0] + 1 <= 450:
                    eligible_videos.add(name)
                    pairs[left[2], right[2]] += 1
            for last in range(450, int(meta[name]["Frames"]) + 1, 450):
                n = sum(last - 450 < start <= end <= last for start, end, _ in targets)
                tiles[n] += 1
        print(split, "recordings", len(names), "target counts", dict(sorted(label_counts.items())))
        print("Candidate pairs", sum(pairs.values()), "in", len(eligible_videos), "sources")
        print("Ordered pair counts", dict(sorted(pairs.items())))
        print("Blind complete tiles by number of complete targets", dict(sorted(tiles.items())))
    if listed != set(meta):
        raise ValueError("Official lists do not cover every source")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder with the four original metadata files")
    summarize(parser.parse_args().folder)

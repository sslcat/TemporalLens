"""Index original RGB shards. Keep provenance and previews private by default."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

from PIL import Image, ImageDraw

WORK = Path("work/ipn")
META = Path("work/real-feasibility/original-release")
OUT = Path("artifacts/ipn")
MEMBER = re.compile(r"frames/([^/]+)/([^/]+)_(\d{6})\.jpg")


def participants(source):
    return "_".join(source.split("_")[:2])


def read_meta():
    with (META / "metadata.csv").open(newline="") as f:
        return {r["Video Name"]: r for r in csv.DictReader(f)}


def scan(args):
    shard, selected = args
    metadata = read_meta()
    previews = WORK / "previews"
    previews.mkdir(exist_ok=True)
    seen = defaultdict(set)
    handles, hashes = {}, defaultdict(dict)
    try:
        with tarfile.open(shard, mode="r|gz", bufsize=1024 * 1024) as archive:
            for member in archive:
                match = MEMBER.fullmatch(member.name.removeprefix("./"))
                if not match or not member.isfile():
                    continue
                source, prefix, digits = match.groups()
                assert source == prefix and source in metadata
                index = int(digits)
                assert index not in seen[source] and 0 < member.size < 512000
                seen[source].add(index)
                if selected is None:
                    if index in (1, 1801):
                        (previews / f"{source}-{index}.jpg").write_bytes(
                            archive.extractfile(member).read()
                        )
                elif source in selected:
                    b = archive.extractfile(member).read()
                    hashes[source][index] = hashlib.sha256(b).digest()
                    if (index - 1) % 4 == 0:
                        if source not in handles:
                            handles[source] = zipfile.ZipFile(
                                WORK / f"{source}.zip", "w"
                            )
                        handles[source].writestr(f"{index}.jpg", b)
        result = {}
        for source, numbers in seen.items():
            total = int(metadata[source]["Frames"])
            if selected is not None and source in selected:
                digest = b"".join(hashes[source][i] for i in sorted(numbers))
                (WORK / f"{source}.hash").write_bytes(digest)
            result[source] = dict(
                shard=shard.name,
                files=len(numbers),
                minimum=min(numbers),
                maximum=max(numbers),
                missing=sorted(set(range(1, total + 1)) - numbers),
                extra=sorted(numbers - set(range(1, total + 1))),
            )
        print(shard.name, len(result), "sources inspected", flush=True)
        return result
    finally:
        for h in handles.values():
            h.close()


def sheets():
    groups = defaultdict(list)
    for v, r in read_meta().items():
        groups[(r["Set"], participants(v))].append(v)
    ordered = sorted(groups)
    for start in range(0, len(ordered), 10):
        page = Image.new("RGB", (800, 10 * 114), "white")
        painter = ImageDraw.Draw(page)
        for row, (split, group) in enumerate(ordered[start : start + 10]):
            y = row * 114
            painter.text((3, y + 4), f"{split} {group}", fill="black")
            for k, v in enumerate(sorted(groups[split, group])):
                for j, i in enumerate((1, 1801)):
                    p = WORK / "previews" / f"{v}-{i}.jpg"
                    if p.exists():
                        with Image.open(p) as im:
                            page.paste(
                                im.resize((88, 66)), (90 + k * 176 + j * 88, y + 21)
                            )
                painter.text((90 + k * 176, y + 88), v.split("#")[-1], fill="black")
        page.save(WORK / f"rooms-{start // 10}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["index", "extract"])
    mode = parser.parse_args().mode
    selected = (
        None
        if mode == "index"
        else {r["source"] for r in json.loads((OUT / "split.json").read_text())}
    )
    shards = sorted(WORK.glob("frames*.tgz"))
    assert len(shards) == 5
    with ProcessPoolExecutor(5) as pool:
        combined = {}
        for entries in pool.map(scan, [(s, selected) for s in shards]):
            assert not combined.keys() & entries.keys()
            combined.update(entries)
    assert set(combined) == set(read_meta())
    (WORK / "index.json").write_text(json.dumps(combined))
    if mode == "index":
        sheets()

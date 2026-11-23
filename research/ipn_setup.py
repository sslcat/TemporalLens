"""Explicit download of pinned encoder files. Add --metadata for research inputs."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
BASE = Path("work/real-feasibility")
ENCODER = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
}
FILES = {
    "original-release/Annot_List.txt": (
        "1zFRngxukYnVPWqAm2isrOrSUfSe-Sfp3",
        "40298deb8151fe6d4223cb31ffde040d73a729b6ed5c63f12e71fd9d3f753bb5",
    ),
    "original-release/metadata.csv": (
        "17d2lsFOJZ-R74PrkS83iO7xSV8ZDiwwT",
        "aa32627e9066e6345e27969d605338be6bc45b0a2bb12b9f45e021d5416c8885",
    ),
    "original-release/Video_TrainList.txt": (
        "1hj8ERCfjzStMzNxfWnS1ywl0ExKvjT00",
        "ce104d51fee138f1aefe9fade7245d874e2dcfa007a6057be37f077c002f5d01",
    ),
    "original-release/Video_TestList.txt": (
        "1CK_wnCZz-O2nZ3IcXYo3uifxEQCtwYbU",
        "1dcdca30132070776e5cd68ed5e300f4abf65cfec02d80894606f25827a60ffb",
    ),
    "classes.txt": (
        "1-3Ue2pMBQuEG26ggzDjdFuoW3akKS6Qq",
        "b2078ebd5c356eb16f4b78d3b4b9a4b1f320ad202671bceae1ba5d4741da3a2b",
    ),
}


def verified_file(path, url, digest, limit):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    with urllib.request.urlopen(url, timeout=60) as src:
        data = src.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected download: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_bytes(data)
    tmp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    for name, digest in ENCODER.items():
        verified_file(
            BASE / "dinov2-small" / name,
            f"https://huggingface.co/facebook/dinov2-small/resolve/{REVISION}/{name}",
            digest,
            88249960,
        )
    if args.metadata:
        for name, (file_id, digest) in FILES.items():
            verified_file(
                BASE / name,
                "https://drive.usercontent.google.com/download?export=download&confirm=t&id="
                + file_id,
                digest,
                200000,
            )
    print("Verified requested local files")

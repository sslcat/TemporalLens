"""Fetch only the five original IPN Hand RGB shards into ignored local storage."""

import concurrent.futures
import os
from pathlib import Path
import time
import urllib.request

SHARDS = [
    ("17yn_1n3LrMHLVSCT4zAbN6sbGXKy1GlA", 1578962498),
    ("1OBlvjl-Z0Wr6xXnLaCGGwFUEXJQLCkDw", 1873001149),
    ("1YCDW763mlXQlfuydFHX_EHG_hyBWC48F", 2053343841),
    ("1I90cK_4gyyQQRkLjFn2p90CCIPV4q41Y", 1659001027),
    ("1SY_TEQX80MtjygS8RqASqRMDDPyiENlZ", 1827928894),
]
WORK = Path("work/ipn")


def fetch(i, file_id, size):
    path = WORK / f"frames{i:02}.tgz"
    if path.exists() and path.stat().st_size == size:
        return
    part = path.with_suffix(".part")
    url = f"https://drive.usercontent.google.com/download?export=download&confirm=t&id={file_id}"
    for attempt in range(5):
        start = part.stat().st_size if part.exists() else 0
        if start == size:
            break
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{size - 1}"}
            )
            with urllib.request.urlopen(request, timeout=60) as src:
                if src.status == 206:
                    if (
                        src.headers.get("Content-Range")
                        != f"bytes {start}-{size - 1}/{size}"
                    ):
                        raise ValueError(f"Unexpected range for {path}")
                elif start or int(src.headers.get("Content-Length", 0)) != size:
                    raise ValueError("Archive server did not honor resume request")
                with part.open("ab") as dst:
                    while block := src.read(1024 * 1024):
                        if dst.tell() + len(block) > size:
                            raise ValueError("Archive exceeds declared length")
                        dst.write(block)
        except (OSError, ValueError) as exc:
            print(path.name, repr(exc), flush=True)
            time.sleep(1 + attempt)
    if part.stat().st_size != size:
        raise IOError(f"Incomplete download: {path}")
    os.replace(part, path)
    print("Retrieved", path, flush=True)


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        futures = [pool.submit(fetch, i, *shard) for i, shard in enumerate(SHARDS, 1)]
        for future in futures:
            future.result()

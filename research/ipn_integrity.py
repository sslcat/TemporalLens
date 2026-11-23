"""Verify all model-input pixels and record a digest per original source."""

import hashlib
import io
import json
import zipfile

from PIL import Image
from .ipn_ingest import OUT, WORK


def main():
    seen = {}
    sources = {}
    for r in json.loads((OUT / "split.json").read_text()):
        v = r["source"]
        sources[v] = hashlib.sha256((WORK / (v + ".hash")).read_bytes()).hexdigest()
        with zipfile.ZipFile(WORK / (v + ".zip")) as z:
            for name in z.namelist():
                with Image.open(io.BytesIO(z.read(name))) as im:
                    digest = hashlib.sha256(im.convert("RGB").tobytes()).digest()
                if seen.setdefault(digest, r["split"]) != r["split"]:
                    raise ValueError(f"Identical sampled pixels across splits: {v}")
    (OUT / "source_integrity.json").write_text(
        json.dumps(
            dict(
                distinct_sampled_pixel_hashes=len(seen),
                cross_split_pixel_duplicates=0,
                sha256_of_ordered_original_jpeg_digests=sources,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

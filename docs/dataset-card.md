# Controlled marker motion v1

I needed a dataset that I could redistribute together with timestamped examples. Jester contains hand gestures and supplies clip labels. The 22.8 GB download and its internal research license made it a poor match. Section 3 prohibits redistribution of the licensed data. EPIC-KITCHENS-100 offers action segments across 100 hours of head-mounted video. Its CC BY-NC 4.0 license permits sharing subject to attribution, change notices and noncommercial use. I did not download either collection. [Jester dataset](https://www.qualcomm.com/developer/software/jester-dataset) · [Jester agreement](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf) · [EPIC-KITCHENS terms](https://epic-kitchens.github.io/2020-100)

## Source and unit

I rendered 396 original, eight-second MP4 clips in Pillow and PyAV, each with 80 frames at 10 Hz and 240 × 160 pixels. A bright cyan disc moves twice across a dark grid. Its radius, color, initial location, travel speeds, rest periods and background vary by scene seed. Ground truth is the motion command that advances the renderer at each tick. Integer rasterization and compression can obscure subpixel changes. See [the generator](../temporallens/data.py).

| Split | Scene seeds | Clips | Events | Events per direction |
|---|---|---:|---:|---:|
| Training | 10000–10239 | 240 | 480 | 120 |
| Validation | 20000–20059 | 60 | 120 | 30 |
| Held-out test | 30000–30095 | 96 | 192 | 48 |

I assigned an entire scene to one split before generating any frames. Each seed defines its own source clip. There are no human subjects, cropped derivatives or reused source recordings. Within each split all twelve ordered pairs of distinct directions appear equally often. A filename and a seed can reveal the pair, so the inference and training paths consume only decoded pixel features. The shared generator intentionally makes the held-out scenes in-distribution.

## Annotation format

Each JSONL record contains the split, integer scene seed, relative clip path, SHA-256 of the encoded file, and two events. The first moving sample defines the inclusive start. The first subsequent stationary sample defines the exclusive end. Left/right and up/down refer to image coordinates. Any interval outside those two events is still. The manifest records the actual generation run. [Release manifest](../artifacts/manifest.jsonl)

The generated imagery and annotations use the repository’s MIT license. The release includes the generator, manifest and one held-out example. Running the generator reconstructs the complete collection under `work/`. See [notices](../NOTICE.md).

The split has no people, camera motion, diagonal action class, occlusion or simultaneous actors. A model succeeding on it has learned a property of these generated trajectories. Generalization to filmed movement requires a new dataset and a new evaluation.

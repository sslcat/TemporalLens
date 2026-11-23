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

## Real-video extension · IPN Hand

I trained the gesture pilot on the original IPN Hand 320 × 240 RGB JPEG release and its matching original temporal annotations. I obtained the five original shards and checked the entire frame index across 200 sources. I used 56 complete recordings: two takes from each of 28 groups, 32/8/16 recordings across train/validation/test. Every selected recording's available frame numbers ran from 1 through its final annotated frame. On the full release, the 14 one-frame differences in the metadata corresponded to absent trailing files. I used the observed numbered images. No refined HandS timelines or derived landmark annotations enter this experiment. [Access audit](real-video-access.md) · [Source manifest](../artifacts/ipn/split.json)

The official class table and publisher descriptions identify ID 8/G05 as throw left, 9/G06 as throw right, 13/G10 as zoom in and 14/G11 as zoom out. I collapsed all other labels to index zero. Each target has 32 training events, eight validation events and 16 held-out events. I checked development thumbnails at the start, middle and end of twenty annotated gestures. Endpoint hand poses can look similar between classes. This was an alignment spot-check by one reviewer. Independent human boundary agreement was not measured. [Publisher vocabulary](https://gibranbenitez.github.io/IPN_Hand/)

The authors explicitly split the corpus by subject. The 50 four-take filename groups are consistent with that split. I kept each selected group and all its derivative windows together. I inspected two frames in each of the four takes from each group and kept visibly matching setups on one side. Several takes visibly changed background within a participant group. Three test groups have ambiguous plain-wall setups. The dataset lacks definitive physical-room IDs. The results support held-out-participant evaluation with inspected backgrounds. Unseen-room generalization remains unverified. [Collection and subject split](https://gibranbenitez.github.io/2021_ICPR_IpnHand.pdf)

I checked all 222,987 selected original JPEG byte hashes and the RGB pixel hashes of all 55,768 sampled inputs for equality across splits. Neither check found a cross-split match. I also compared four low-resolution grayscale observations from overlapping eight-tick windows, striding four ticks. The closest cross-split RMS was .0889 on [0,1] intensity against a .025 review threshold. I visually inspected the closest train/test and validation/test pairs: their backgrounds, clothing and poses differed. This check can miss transformed, cropped or time-shifted copies. The archives, thumbnails, clips and feature caches stay in ignored local storage. [Integrity evidence](../artifacts/ipn/source_integrity.json) · [Feature audit](../artifacts/ipn/feature_audit.json) · [Data rights](../artifacts/ipn/NOTICE.md)

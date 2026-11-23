# Real-video source and access audit

I checked the following first-party distributions on 2026-11-23. Directory listings, successful byte reads and complete files are recorded separately below. I used anonymous requests. I kept third-party media, metadata and model weights outside Git.

## IPN Hand — recommended original release

Benitez-Garcia, Olivares-Mercado, Sanchez-Perez and Yanai recorded 200 webcam sessions from 50 people across 28 scenes. Participants performed touchless-interface gestures in four prompted sessions each. The authors apply **CC BY 4.0** to both the data and annotations. It permits copying, adaptation and commercial use with attribution. A redistributed subset should identify the authors and title, link the source and license, preserve supplied notices and state edits such as cropping, sampling or relabeling. Identifiable people and bystanders introduce privacy and publicity considerations outside that copyright grant. My proposed release contains code and manifests by default. The project's MIT notice covers its own work. [Dataset and citation](https://gibranbenitez.github.io/IPN_Hand/) · [Collection paper](https://mm.cs.uec.ac.jp/e/pub/conf20/210110gibran_1.pdf) · [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)

| Original distribution | Access check | Measured size |
|---|---|---:|
| Resized RGB JPEGs, five independent gzip tar archives | All five returned HTTP 206. I decoded a 320 × 240 JPEG from each 128 KiB prefix. I also decoded eight numerically identified images from an official training source in shard 02. | 8,992,237,409 bytes across five files (8.37 GiB). Smallest 1,578,962,498 bytes. |
| Original recordings, five archives | All five returned HTTP 206, gzip magic and a readable first tar header on a 64 KiB range. The publisher labels these MP4 videos. The first member of each archive actually has an AVI filename. I did not verify the full containers. | 4,943,719,526 bytes (4.60 GiB). |
| Original temporal labels, metadata, class key and train/test lists | Complete public files read. Label CSV contains source ID, class code and ID, first and last frame, and duration in frames. | Label file 197,443 bytes, metadata CSV 12,584 bytes. |
| Refined IPN HandS temporal ZIP | Returned HTTP 200, 82,274 bytes. ZIP member CRCs passed. The publisher checksum and retrieved digest disagree. | Separate release, excluded from this protocol. |

The byte totals sum the denominators of the five HTTP Content-Range headers in each folder. A successful prefix proves partial access at the time checked. Full transfer, archive-wide integrity and future Drive quota are unverified. [Original JPEG archives](https://drive.google.com/drive/folders/1GJ3ZzidMngAK5lbcLtX5aENfXelIFAQe) · [Original video archives](https://drive.google.com/drive/folders/1O4Fn_jbAEcKIksXHmQIMcHTyaDw4wt9x) · [Original metadata folder](https://drive.google.com/drive/folders/1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD)

### Which timeline I can use

I paired the **original 320 × 240 frame release with the original Annot_List.txt**. Its 5,649 inclusive intervals start at frame 1 and form contiguous runs within each recording. They cover 800,491 frames. The metadata and split lists report 800,505. In exactly 14 recordings there is one additional trailing frame. Mask that unannotated tick if present. Before sampling, enumerate each retained source's actual numbered JPEGs, verify the prefix, numbering and final frame, and quarantine unexpected discrepancies. The original extraction removed near-duplicate frames. Nominal seconds on this index are suitable for a preview assembled at 30 Hz. Actual wall-clock boundaries on the independent raw-video timeline require a verified map. Score boundaries primarily in released-frame units. [Original annotations](https://drive.google.com/file/d/1zFRngxukYnVPWqAm2isrOrSUfSe-Sfp3/view) · [Authors' timeline explanation](https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md)

The refined HandS release supplies different class IDs and inclusive 1-based labels for frames extracted at 30 fps from raw recordings. Its frame-count CSV differs between timelines for 95/200 videos. For example, the first video in frame shard 01 has 3,855 original frames versus 3,987 on the refined timeline. I observed SHA-256 `520dfbf3d52ecab6535d8841bba55e94d6b512596ed1cff4505d2c1345bcbaec` for the refined ZIP. The authors list `4686a2ebf70a6036166c261bc6992d050cfba08c3b18edcdd460b3fa9f30ec71`. The cause is unknown. Reconcile the version with the publisher before any future use of refined labels. The original pairing is independently accessible. [Refined release notes and checksums](https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md) · [Refined ZIP listing](https://drive.google.com/file/d/1HybcqZqnNKZGgJJXCMjj6tLFbpsNP_a1/view)

### Rechecking the metadata

For the original release I recorded the following locally observed SHA-256 digests. Place the four files from the original metadata folder in one directory and run `python research/audit_ipn.py PATH_TO_DIRECTORY` from the repository root. This standard-library check reads only local tables and prints the pair and tiling counts used in the design. [Audit code](../research/audit_ipn.py)

| Original file | SHA-256 |
|---|---|
| Annot_List.txt | `40298deb8151fe6d4223cb31ffde040d73a729b6ed5c63f12e71fd9d3f753bb5` |
| metadata.csv | `aa32627e9066e6345e27969d605338be6bc45b0a2bb12b9f45e021d5416c8885` |
| Video_TrainList.txt | `ce104d51fee138f1aefe9fade7245d874e2dcfa007a6057be37f077c002f5d01` |
| Video_TestList.txt | `1dcdca30132070776e5cd68ed5e300f4abf65cfec02d80894606f25827a60ffb` |

## Object-interaction alternatives

| Dataset | Events and size | Actual access and rights | Fit |
|---|---|---|---|
| KIT Bimanual Actions | 540 recordings, six people, 2 h 18 min. Frame-indexed half-open intervals per hand include approach, retreat, lift and place. Six RGB-D subject ZIPs total about 93.2 GiB. | The first RGB-D ZIP returned HTTP 200 and ZIP bytes on a capped read. The action ZIP also returned ZIP bytes. KIT restricts RGB/depth images to scientific purposes. Annotations and derived data offered by KIT have a separate CC0 dedication. | Relevant lifting/placing task, but one subject costs at least 12.3 GiB at the archive level. The image grant does not establish commercial redistribution rights. [Files and schema](https://bimanual-actions.humanoids.kit.edu/original_dataset) · [Publisher terms](https://bimanual-actions.humanoids.kit.edu/terms-of-use) |
| Assembly101 | 4,321 videos, 362 multiview recordings of toy assembly/disassembly, over 100k coarse and 1M fine temporal segments. The Hub inventory reported 3.89 TB of repository storage across assets. One listed camera MP4 was 824,653,792 bytes. | Public inventory accessible. The primary Hub reports an automatic access gate. An anonymous range GET on the listed C10095_rgb.mp4 returned HTTP 401. The authors use CC BY-NC 4.0, allowing attributed noncommercial sharing and adaptation. | Multiple views of the same physical event must stay in one split. The size, gate and noncommercial terms work against this release. [Publisher and terms](https://assembly-101.github.io/) · [Official file distribution](https://huggingface.co/datasets/cvml-nus/assembly101/tree/main) |

The original synthetic study also considered EPIC-KITCHENS and Jester. Its licensing review is available in the [existing dataset card](dataset-card.md). For an object-manipulation release with broad redistribution rights, my fallback is a consented original capture described in the [pilot design](real-video-feasibility.md).

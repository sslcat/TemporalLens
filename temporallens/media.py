"""Bounded decoding and the fixed appearance encoder shared by both heads."""

from pathlib import Path
import math
import time

import av
import numpy as np

HZ = 10
MAX_BYTES = 24 * 1024 * 1024
LABELS = ("Still", "Left", "Right", "Up", "Down")


class VideoError(ValueError):
    pass


def embedding(rgb):
    # One bright cyan component against a dark scene. Coordinates in image units.
    r, g, b = rgb.transpose(2, 0, 1)
    y, x = np.nonzero((r < 130) & (g > 150) & (b > 145))
    h, w = r.shape
    if not 4 <= len(x) <= 0.12 * w * h:
        return (0.0, 0.0), False
    compact = len(x) / ((np.ptp(x) + 1) * (np.ptp(y) + 1))
    return (float(x.mean() / w), float(y.mean() / h)), compact >= 0.40


def decode(path, *, gesture_frames=False):
    rate = 7.5 if gesture_frames else HZ
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise VideoError("The file exceeds 24 MiB.")
    with path.open("rb") as file:
        header = file.read(16)
    if header[4:8] == b"ftyp":
        fmt = "mov"
    elif header[:4] == bytes.fromhex("1a45dfa3"):
        fmt = "matroska"
    else:
        raise VideoError("Choose an MP4 or WebM video.")
    try:
        # Force a local container demuxer. Disable secondary MOV data references.
        with (
            path.open("rb") as src,
            av.open(
                src,
                format=fmt,
                options={
                    "protocol_whitelist": "file",
                    "enable_drefs": "0",
                    "use_absolute_path": "0",
                    "max_pixels": str(1920 * 1080),
                },
            ) as movie,
        ):
            if len(movie.streams.video) != 1:
                raise VideoError("Choose a file with one video stream.")
            stream = movie.streams.video[0]
            if stream.codec_context.name not in {"h264", "vp8", "vp9", "mpeg4"}:
                raise VideoError("Use H.264, MPEG-4, VP8 or VP9 video.")
            if not 0 < stream.width * stream.height <= 1920 * 1080:
                raise VideoError("Resolution exceeds the 1080p pixel limit.")
            stream.codec_context.options = {"max_pixels": str(1920 * 1080)}
            stream.codec_context.thread_count = 1
            start = last = None
            previous = None
            features, valid = [], []
            started = time.monotonic()
            for index, frame in enumerate(movie.decode(stream)):
                if index >= 900 or time.monotonic() - started > 12:
                    raise VideoError("Video exceeds the decoding budget.")
                if frame.width * frame.height > 1920 * 1080 or frame.pts is None:
                    raise VideoError("Invalid frame geometry or timestamp.")
                stamp = float(frame.pts * frame.time_base)
                if start is None:
                    start = stamp
                t = stamp - start
                if (
                    not math.isfinite(t)
                    or t >= 15.001
                    or (last is not None and t <= last)
                ):
                    raise VideoError("Use 5–15 seconds with increasing timestamps.")
                # Sample the frame displayed at each model tick (also for VFR).
                while previous is not None and len(features) / rate < t - 1e-6:
                    features.append(previous[0])
                    valid.append(previous[1])
                if gesture_frames:
                    scale = 320 / max(frame.width, frame.height)
                    previous = (
                        frame.reformat(
                            width=max(1, round(frame.width * scale)),
                            height=max(1, round(frame.height * scale)),
                            format="rgb24",
                        ).to_ndarray(),
                        True,
                    )
                else:
                    rgb = frame.reformat(
                        width=96, height=64, format="rgb24"
                    ).to_ndarray()
                    previous = embedding(rgb)
                last = t
                period = (
                    float(frame.duration * frame.time_base)
                    if frame.duration
                    else 1 / float(stream.average_rate or HZ)
                )
            if last is None:
                raise VideoError("No decodable video frames.")
            duration = last + period
            if not 4.999 <= duration <= 15.001:
                raise VideoError("Choose a video lasting 5–15 seconds.")
            while len(features) / rate < duration - 1e-6:
                features.append(previous[0])
                valid.append(previous[1])
            return (
                (features if gesture_frames else np.asarray(features, np.float32)),
                np.asarray(valid),
                duration,
            )
    except (av.error.FFmpegError, OverflowError, ZeroDivisionError) as error:
        raise VideoError("The video is damaged or unsupported.") from error

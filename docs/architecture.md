# Architecture

I use the same Python decoder, encoder and segment extraction code in training and serving. The browser sends the file bytes to a single FastAPI process on loopback. The server writes them to a random temporary directory, decodes on a worker thread and returns a small JSON response. The browser previews its own object URL and seeks locally. The example takes the same upload route as a user file.

```text
MP4 / WebM → decoded frames with timestamps → 10 Hz display-time samples
           → cyan pixel centroid → frame head OR temporal head
           → five logits per tick → sustained label runs → timestamped events
```

| Module | Responsibility |
|---|---|
| `media.py` | Container and codec checks, time sampling, RGB conversion and marker observation |
| `data.py` | Seeded scene rendering, split manifest and decoded feature cache |
| `model.py` | One-frame and five-frame heads, contiguous run extraction |
| `train.py` | Seeded optimization and validation checkpoint selection |
| `evaluate.py` | Oracle recognition, full-clip prediction, interventions and bootstrap |
| `figures.py` | Plots generated from saved predictions and decoded examples |
| `api.py` | Model lifecycle, bounded uploads, inference and local static UI |

I chose a fixed marker encoder because the experiment isolates direction and timing, and rendered coordinates can be recovered from color alone. Training consumes the observations recovered from compressed pixels. I never feed the simulator’s trajectory into a model. Both heads see identical normalized coordinates. The temporal head has access to a centered five-frame neighborhood, so offline predictions can use up to 200 ms of future context.

## Serving

`GET /api/status` reports the loaded model, sampling rate, classes and device. `POST /api/analyze` accepts raw MP4/WebM bytes with a video MIME type or `application/octet-stream`. Results include seconds, half-open intervals, direction and the mean uncalibrated softmax score for each detected run. Any failed marker observation marks the clip unsupported. Confidence below 0.70 receives an uncertainty tag in the UI. This display threshold has no calibration guarantee.

Uploads are limited to 24 MiB, 5–15 seconds, 2,073,600 pixels per frame, 900 decoded frames and one active request per process. The server checks body size while streaming, file signature, stream count, codec, dimensions and decoded timestamps. It uses a fixed local filename, forces the demuxer and reads a local file object. Temporary storage uses a context manager on success and failure. Install one worker and bind to loopback. The 12-second decode check runs between frames and the native codec runs inside the server process. This design targets local use. All page assets and model weights are local.

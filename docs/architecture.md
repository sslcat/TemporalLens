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

## Real-video mode

The mode selector sends a raw upload to `POST /api/analyze?mode=gesture`. The default remains `marker`. Both paths share the existing byte, container, codec, duration and pixel checks, one-request limit and temporary-file lifecycle. Gesture mode samples the displayed frame at 7.5 Hz, preserves aspect ratio and bounds the longest edge at 320 pixels. On first use it loads the verified local encoder and head. An incomplete gesture install reports HTTP 503. The browser reuses the event timeline and seeks the local object URL. It identifies the checkpoint, uncalibrated scores, supported gestures and observed reliability limits. The gesture results link displays the real pilot measurements.

I use DINOv2-S/14's patch tokens, pooled into four spatial cells. The independent-frame and order-invariant heads apply three residual pointwise blocks of width 64→96→64 after projection. The latter averages a local window before projection. The temporal head uses three width-three convolutions with dilations 1/2/4 and residual connections. Each primary head emits five logits per tick. The classical reference fits 725 multinomial logistic coefficients and biases to 144 optical-flow features. Private RGB assets and cache paths are outside the web static directory. [Shared gesture inference](../temporallens/gesture.py) · [Protocol](ipn-protocol.md)

## Proposing an interval before assigning a gesture

For `POST /api/analyze?mode=two-stage`, I use the same decoded frames, frozen vision encoder and pilot CNN. At each tick I concatenate 5- and 15-tick means of the 144 motion features and five CNN logits. A 299-parameter binary logistic model estimates target presence. At a score of .40 I form runs, bridge gaps of up to three ticks and retain spans of at least six ticks. All windows can use future frames.

For each proposal the selected classifier averages the CNN's four target logits and takes a softmax. At .60 or above, the response includes a labeled event with the candidate's original endpoints. The response also includes the full candidate list, available for seeking. Both real modes shortcut a perfectly static decoded clip to an empty result. The 84-parameter fusion comparator and 21-parameter diagnostic background gate are research artifacts. Validation selected the frozen CNN for serving. [Inference implementation](../temporallens/two_stage.py) · [Selection record](../artifacts/two-stage/selection.json) · [Rejection diagnostic](candidate-rejection.md)

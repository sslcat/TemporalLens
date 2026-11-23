import asyncio
from contextlib import asynccontextmanager
from importlib.util import find_spec
from typing import Literal
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
import torch

from .media import HZ, LABELS, MAX_BYTES, VideoError, decode
from .model import MotionHead, segments
from .gesture import GesturePredictor, encoder_dir, LABELS as GESTURE_LABELS

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


@asynccontextmanager
async def lifespan(app):
    preferred = os.getenv("TEMPORALLENS_DEVICE", "auto")
    if preferred not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("TEMPORALLENS_DEVICE must be auto, cpu or cuda")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if preferred == "auto"
        else preferred
    )
    torch.set_num_threads(1)
    app.state.device = device
    app.state.model = MotionHead(True).to(device).eval()
    state = torch.load(
        ROOT / "artifacts/temporal.pt", map_location=device, weights_only=True
    )
    app.state.model.load_state_dict(state)
    app.state.busy = False
    app.state.gesture_model = None
    app.state.two_stage_model = None
    yield


app = FastAPI(title="TemporalLens", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
)


def analyze(path, model, device):
    x, valid, duration = decode(path)
    response = dict(
        duration=duration, device=device, events=[], supported=bool(valid.all())
    )
    if not response["supported"]:
        return response
    with torch.inference_mode():
        probs = model(torch.from_numpy(x)[None].to(device))[0].softmax(-1).cpu().numpy()
    for k, a, b in segments(probs.argmax(-1)):
        score = float(probs[a:b, k].mean())
        response["events"].append(
            dict(
                label=LABELS[k],
                start=a / HZ,
                end=min(duration, b / HZ),
                score=score,
                uncertain=score < 0.7,
            )
        )
    return response


def gesture_ready():
    return (
        find_spec("transformers") is not None
        and (encoder_dir() / "model.safetensors").is_file()
        and (ROOT / "artifacts/ipn/gesture.pt").is_file()
    )


def two_stage_config():
    path = ROOT / "artifacts/two-stage/selection.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def two_stage_ready():
    cfg = two_stage_config()
    needs_visual = cfg and (
        cfg["primary"] != "flow" or cfg["proposal"]["detector"] == "hybrid"
    )
    return bool(
        cfg
        and find_spec("cv2")
        and (ROOT / "artifacts/two-stage/heads.pt").is_file()
        and (ROOT / "artifacts/ipn/flow.pt").is_file()
        and (not needs_visual or gesture_ready())
    )


def analyze_two_stage(path):
    from .two_stage import TwoStagePredictor

    frames, _, duration = decode(path, gesture_frames=True)
    if app.state.two_stage_model is None:
        try:
            app.state.two_stage_model = TwoStagePredictor(app.state.device)
        except (ImportError, OSError, RuntimeError) as exc:
            raise HTTPException(
                503,
                "The local two-stage model could not be loaded. Check the pilot setup.",
            ) from exc
    return app.state.two_stage_model.analyze(frames, duration)


def analyze_gesture(path):
    frames, _, duration = decode(path, gesture_frames=True)
    if app.state.gesture_model is None:
        try:
            app.state.gesture_model = GesturePredictor(app.state.device)
        except (ImportError, OSError, RuntimeError) as exc:
            raise HTTPException(
                503,
                "The local gesture checkpoint or encoder could not be loaded. Run the gesture setup in the reproduction guide.",
            ) from exc
    return app.state.gesture_model.analyze(frames, duration)


@app.get("/api/status")
def status():
    return dict(
        model="Temporal CNN",
        device=app.state.device,
        frequency=HZ,
        actions=LABELS[1:],
        gesture_ready=gesture_ready(),
        gesture_actions=GESTURE_LABELS[1:],
        two_stage_ready=two_stage_ready(),
        two_stage_classifier=two_stage_config().get("primary"),
    )


@app.post("/api/analyze")
async def upload(
    request: Request, mode: Literal["marker", "gesture", "two-stage"] = "marker"
):
    if mode == "two-stage" and not two_stage_ready():
        raise HTTPException(
            503,
            "Two-stage mode needs the local pilot weights and optional dependencies. See the reproduction guide.",
        )
    if mode == "gesture" and not gesture_ready():
        raise HTTPException(
            503,
            "Gesture mode needs its local encoder and pilot dependencies. See the reproduction guide.",
        )
    mime = request.headers.get("content-type", "").split(";")[0]
    if mime not in {"video/mp4", "video/webm", "application/octet-stream"}:
        raise HTTPException(415, "Send the video as an MP4 or WebM request body.")
    size = request.headers.get("content-length")
    if size and (not size.isdecimal() or len(size) > 10 or int(size) > MAX_BYTES):
        raise HTTPException(413, "The upload exceeds 24 MiB.")
    if app.state.busy:
        raise HTTPException(429, "An analysis is already running.")
    app.state.busy = True
    try:
        with TemporaryDirectory(prefix="temporallens-") as directory:
            path = Path(directory) / "input.bin"
            received = 0
            async with asyncio.timeout(30):
                with path.open("wb") as dest:
                    async for chunk in request.stream():
                        received += len(chunk)
                        if received > MAX_BYTES:
                            raise HTTPException(413, "The upload exceeds 24 MiB.")
                        dest.write(chunk)
            try:
                if mode == "two-stage":
                    return await asyncio.to_thread(analyze_two_stage, path)
                if mode == "gesture":
                    return await asyncio.to_thread(analyze_gesture, path)
                return await asyncio.to_thread(
                    analyze, path, app.state.model, app.state.device
                )
            except VideoError as error:
                raise HTTPException(422, str(error)) from error
    except TimeoutError as error:
        raise HTTPException(408, "The upload timed out.") from error
    finally:
        app.state.busy = False


@app.get("/two-stage-results.png")
def two_stage_results():
    return FileResponse(ROOT / "artifacts/two-stage/comparison.png")


@app.get("/gesture-results.png")
def gesture_results():
    return FileResponse(ROOT / "artifacts/ipn/comparisons.png")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/assets", StaticFiles(directory=WEB), name="assets")

# Reproducing the local release

I ran the experiment from the repository root on Windows with CPython 3.13, an RTX 4070 SUPER and the official PyTorch 2.9.1 CUDA 12.8 wheel. The code also supports a CPU-only run. The exact release environment is recorded in [the package snapshot](../artifacts/environment.txt). The setup below uses the portable direct dependency pins. [Official wheel index and versions](https://pytorch.org/get-started/previous-versions/)

```powershell
cd D:\TemporalLens
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
```

For CPU use the PyTorch index ending in `/cpu` and pass `--device cpu` to the two commands below. On Linux use `python3.13 -m venv .venv` and `source .venv/bin/activate`.

## Rebuild all measurements

```powershell
python -m temporallens.data
python -m temporallens.train --device cuda
python -m temporallens.evaluate --device cuda
python -m temporallens.figures
python -m temporallens.controls
python -m pytest -q
```

I run these commands in order. They replace the generated dataset, checkpoints and metrics in the working copy. Use a fresh local clone to preserve the shipped run. The seed ranges and pair schedules fix the split. Codec versions can change encoded hashes and pixel rounding. GPU/software versions and other desktop workloads can change both timings and floating-point results. For one summary figure the script chooses the highest and lowest temporal IoU, breaking ties by manifest order. The app’s example is the first held-out clip.

## Start the app and check the browser

```powershell
$env:TEMPORALLENS_DEVICE = 'auto'
python -m uvicorn temporallens.api:app --host 127.0.0.1 --port 18731
```

Open [the workspace](http://127.0.0.1:18731). The packaged example runs through the model at startup. Click an event to seek or upload an MP4/WebM. In a second activated shell, run the browser check using a local Edge binary.

```powershell
$env:TL_BROWSER = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
python tests/browser_smoke.py
```

Alternatively install Playwright Chromium with `python -m playwright install chromium` and leave `TL_BROWSER` unset. The smoke test saves actual desktop and mobile captures under the ignored `work/` directory. Stop Uvicorn with Ctrl+C. The CI definition runs CPU tests and a browser smoke test on Linux. The local release does not require a cloud account.


## Local release checks

I regenerated all 396 videos in a second working directory. Every encoded SHA-256, annotation record, sampled feature and target matched the release. A second RTX 4070 SUPER training run reproduced every checkpoint tensor and both complete loss histories exactly.

All 18 local pytest cases passed on the release suite. They include MP4 and WebM, sampling above and below 10 Hz, both duration bounds, an unsupported scene, a stationary marker, malformed and oversized uploads, temporary-file cleanup, host filtering and CPU inference. Starlette's test client emitted one dependency deprecation warning. Ruff reported no findings. I also exercised CUDA inference through the live FastAPI app in Edge, including playback, upload, error recovery, seeking through both controls and a 390-pixel mobile layout. [Desktop capture](../artifacts/workspace.png)

The GitHub Actions workflow is prepared for a Linux CPU runner. Remote CI execution awaits publication approval.

## Run or reproduce the IPN Hand gesture pilot

For app inference, activate the project environment configured above, install the gesture extras, and explicitly fetch the hash-checked public DINOv2 encoder. These commands run from the repository root. The application loads that encoder from local disk on first gesture use. The shipped `artifacts/ipn/gesture.pt` contains the head and train-only normalization. The synthetic demo works without the optional encoder.

```powershell
python -m pip install -r requirements-gesture.txt
python -m research.ipn_setup
python -m uvicorn temporallens.api:app --host 127.0.0.1 --port 18731
```

Select **Real hand gestures · IPN pilot** and upload a continuous 5–15 s clip. The four supported classes and the detector's limitations appear in the workspace. The API accepts the same bounded video body at `/api/analyze?mode=gesture`.

For research reproduction use a fresh local checkout. The next commands download 8.99 GB of official RGB archives and regenerate pilot outputs. Allow at least 20 GB of disk and use CUDA for the pilot. The metadata/encoder setup verifies pinned digests. The split manifest is fixed at commit `5237f80`.

```powershell
python -m pip install -r requirements-ipn-research.txt
python -m research.ipn_setup --metadata
python -m research.ipn_fetch
python -m research.ipn_ingest index
python -m research.ipn_ingest extract
python -m research.ipn_integrity
python -m research.ipn_features
python -m research.ipn_train
python -m research.ipn_train --replay
python -m research.ipn_evaluate
python -m research.ipn_diagnostics
python -m research.ipn_figures
python -m pytest -q
```

With the updated Uvicorn instance running, use a second shell to create private QA uploads from validation data, then verify desktop and mobile behavior. The fixture search selects a clip with a detectable event so the test can operate the timeline. The held-out evaluation uses every designated source.

```powershell
python -m research.ipn_ui_fixture
$env:TL_BROWSER = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
python tests/ipn_browser_smoke.py
python -m research.ipn_ui_fixture --latency
```

Original archives, derived JPEGs, feature caches, private browser captures and per-candidate weights stay under ignored `work/`. The public release includes the four primary heads and numeric evidence. Software and GPU differences may change timing and floating-point output. [Pilot environment](../artifacts/ipn/environment.json) · [Data attribution](../artifacts/ipn/NOTICE.md)

The two-stage follow-up reuses this feature cache and both frozen component heads. Its separate local training, replay, testing and browser commands are in the [two-stage study](two-stage.md#local-execution-and-reproduction).

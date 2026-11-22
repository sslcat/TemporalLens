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

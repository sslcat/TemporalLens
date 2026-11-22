# Licensing and attribution

I release the project-authored source, documentation, renderer, rendered dataset, annotation manifest, original experiment figures and learned weights under the accompanying MIT license. Please preserve its copyright and permission notice when redistributing them. All training imagery comes from the renderer. The scope of this grant is the material authored for TemporalLens.

PyTorch supplies tensor operations and CUDA kernels. PyAV and its FFmpeg libraries supply video decoding and encoding. Pillow draws the marker, NumPy handles coordinates and Matplotlib renders the plots. FastAPI, Starlette and Uvicorn serve the local app. Pytest, HTTPX and Playwright support the checks. Each dependency has its own distribution terms. A source checkout installs these packages separately. Anyone redistributing a bundled environment should review the notices in the wheels, including FFmpeg and codec components.

I reviewed EPIC-KITCHENS-100 and Jester during dataset selection. Neither contributes footage, annotations or weights to this release. The dataset card links to the publishers’ terms.

I used OpenAI Codex for assistance with the research design, implementation, local experiment execution, tests and documentation. The numeric results come from the saved local runs.

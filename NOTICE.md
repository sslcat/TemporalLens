# Licensing and attribution

I release the project-authored source, documentation, renderer, rendered dataset, synthetic annotation manifest, marker experiment figures and marker weights under the accompanying MIT license. Please preserve its copyright and permission notice when redistributing them. The marker experiment uses imagery from the renderer. The gesture experiment uses original IPN Hand RGB frames under CC BY 4.0 and the separately downloaded Apache-2.0 DINOv2 encoder. The pilot artifacts carry their own [attribution and license](artifacts/ipn/NOTICE.md). The scope of this grant is the material authored for TemporalLens.

PyTorch supplies tensor operations and CUDA kernels. PyAV and its FFmpeg libraries supply video decoding and encoding. Pillow draws the marker, NumPy handles coordinates and Matplotlib renders the plots. FastAPI, Starlette and Uvicorn serve the local app. Pytest, HTTPX and Playwright support the checks. Each dependency has its own distribution terms. A source checkout installs these packages separately. Anyone redistributing a bundled environment should review the notices in the wheels, including FFmpeg and codec components.

I reviewed EPIC-KITCHENS-100 and Jester during dataset selection. Neither contributes footage, annotations or weights to this release. The dataset card links to the publishers’ terms.

I used OpenAI Codex for assistance with the research design, implementation, local experiment execution, tests and documentation. The numeric results come from the saved local runs.

The desktop screenshot shows the TemporalLens UI and the generated marker example, released under MIT. For the mobile screenshot I cropped the live results panels after analyzing an IPN Hand validation clip. The displayed intervals and labels are my derived predictions. I release that screenshot under CC BY 4.0 with the same IPN Hand attribution. The public images contain no participant frames. [Original pilot notice](artifacts/ipn/NOTICE.md) · [Two-stage artifacts](artifacts/two-stage/NOTICE.md) · [Validation diagnostic artifacts](artifacts/rejection/NOTICE.md)

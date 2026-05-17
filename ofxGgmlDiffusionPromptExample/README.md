# ofxGgmlDiffusionPromptExample

Root-level text-to-image example for `ofxGgmlDiffusion`.

The native runtime is opt-in. Build it first:

```powershell
..\scripts\build-stable-diffusion.bat
```

After build, regenerate the example project (projectGenerator) and rebuild so the
addon config macro is picked up.

Then point the example at a local model with `OFXGGML_DIFFUSION_MODEL`, or place
one at `bin/data/models/model.safetensors`.

For PhotoMaker, set `OFXGGML_PHOTOMAKER_MODEL` and
`OFXGGML_PHOTOMAKER_REFS` before launch. `OFXGGML_PHOTOMAKER_REFS` is a
semicolon-separated list of image paths. Run `..\scripts\doctor-diffusion.bat`
from the addon root if the example does not find the expected local assets.

Press `R` to run one text-to-image request on a worker thread. Press `C` to
cancel the pending result. Output is saved under `bin/data/outputs`.

```powershell
..\scripts\run-diffusion-example.bat -DryRun
..\scripts\run-diffusion-example.bat -Build -Model C:\path\to\model.safetensors
```

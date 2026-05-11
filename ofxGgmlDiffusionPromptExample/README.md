# ofxGgmlDiffusionPromptExample

Root-level text-to-image example for `ofxGgmlDiffusion`.

The native runtime is opt-in. Build it first:

```powershell
..\scripts\build-stable-diffusion.bat
```

Then point the example at a local model with `OFXGGML_DIFFUSION_MODEL`, or place
one at `bin/data/models/model.safetensors`.

Press `R` to run one text-to-image request on a worker thread. Press `C` to
cancel the pending result. Output is saved under `bin/data/outputs`.

```powershell
..\scripts\run-diffusion-example.bat -DryRun
..\scripts\run-diffusion-example.bat -Build -Model C:\path\to\model.safetensors
```

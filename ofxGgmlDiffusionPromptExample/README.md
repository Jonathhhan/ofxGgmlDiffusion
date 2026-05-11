# ofxGgmlDiffusionPromptExample

Root-level text-to-image example for `ofxGgmlDiffusion`.

The native runtime is opt-in. Build it first:

```powershell
..\scripts\build-stable-diffusion.bat
```

Then point the example at a local model with `OFXGGML_DIFFUSION_MODEL`, or place
one at `bin/data/models/model.safetensors`.

Press `R` to run one blocking text-to-image request. Output is saved under
`bin/data/outputs`.

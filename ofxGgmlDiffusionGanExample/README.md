# ofxGgmlDiffusionGanExample

Root-level GAN image generation request example for `ofxGgmlDiffusion`.

This example does not pretend to generate images until a real GAN backend is
installed. It builds and validates a GAN image request, then runs it through the
unavailable backend so the next implementation step has a clear plug-in point.

Set `OFXGGML_GAN_GENERATOR` to a local exported generator file, or place one at
`bin/data/models/generator.gguf`.

```powershell
..\scripts\run-gan-example.bat -DryRun
..\scripts\run-gan-example.bat -Build -Generator C:\path\to\generator.gguf
```

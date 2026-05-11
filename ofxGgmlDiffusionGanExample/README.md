# ofxGgmlDiffusionGanExample

Root-level GAN image generation request example for `ofxGgmlDiffusion`.

This example uses `ofxGgmlDiffusionTinyGanBackend`, a fixed tiny MLP generator
proof. It produces pixels when ggml headers/libs are available through
`ofxGgmlCore`; otherwise it reports a clear unavailable-backend message.

The built-in `builtin:tiny-mlp` generator is deterministic and not trained. It
is only a proof that a GAN-style generator can run through the addon boundary.
Set `OFXGGML_GAN_GENERATOR` later to test an exported generator file once that
format is defined.

```powershell
..\scripts\run-gan-example.bat -DryRun
..\scripts\run-gan-example.bat -Build -Generator C:\path\to\generator.gguf
```

# ofxGgmlDiffusionGanExample

Root-level GAN image generation request example for `ofxGgmlDiffusion`.

This example uses `ofxGgmlDiffusionTinyGanBackend`, a fixed tiny MLP generator
proof. It produces pixels when ggml headers/libs are available through
`ofxGgmlCore`; otherwise it reports a clear unavailable-backend message.

The built-in `builtin:tiny-mlp` generator is deterministic and not trained. You
can also generate a tiny preset file that changes the deterministic seeds and
shape used by the proof backend:

```powershell
..\scripts\create-tiny-gan-preset.bat
..\scripts\run-gan-example.bat -DryRun
..\scripts\run-gan-example.bat -Build -Generator bin\data\models\tiny-mlp.ofxggmlgan
```

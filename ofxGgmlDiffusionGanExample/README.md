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
..\scripts\create-tiny-gan-fixtures.bat -OutputPath bin\data\datasets\tiny-fixtures -Count 8
..\scripts\train-tiny-gan.bat -DryRun -Dataset C:\path\to\images -Epochs 2 -DryRunBatchesPerEpoch 3
..\scripts\train-tiny-gan.bat -DryRun -Dataset bin\data\datasets\tiny-fixtures -OutputPreset bin\data\models\tiny-preview-trained.ofxggmlgan -WritePreviewPreset -Force
..\scripts\run-gan-example.bat -DryRun
..\scripts\run-gan-example.bat -DryRun -PreviewPreset -ForcePreviewPreset
..\scripts\run-gan-example.bat -Build -Generator bin\data\models\tiny-mlp.ofxggmlgan
```

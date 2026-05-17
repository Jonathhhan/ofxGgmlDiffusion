# ofxGgmlDiffusionGanExample

Root-level GAN image generation request example for `ofxGgmlDiffusion`.

This example has two lanes:

- **Proof** uses `ofxGgmlDiffusionTinyGanBackend` (deterministic tiny ggml graph,
  not trained) to validate request plumbing and image postprocessing.
- **Production** uses `ofxGgmlDiffusionGgufGanBackend` with an exported `.gguf`
  generator checkpoint.

The built-in `builtin:tiny-mlp` generator is deterministic and **not trained**.
It always runs a small ggml graph in-process and writes synthetic image output.
You can write and load tiny preset files (`.ofxggmlgan`) to change the deterministic
seed/shape/scale parameters used by this proof path.

`ofxGgmlDiffusionTinyGanBackend` proves:
- ggml-based inference wiring for image outputs
- request/build wiring in examples and async UX
- dry-run planning for tiny training settings

`ofxGgmlDiffusionTinyGanBackend` intentionally does **not** provide:
- real pretrained GAN checkpoint loading
- real discriminator/generator weight updates in this release
- trained quality benchmarking
- real production-GAN inference

`ofxGgmlDiffusionGgufGanBackend` is the explicit production lane marker for
exported `.gguf` checkpoints and is intentionally split from the proof workflow so
training tools and presets cannot mask production path requirements.

At this step, the lane validates GGUF checkpoint loading prerequisites and keeps the
request/run contract clear; the backend runtime call is intentionally isolated so a native
generator runtime can be dropped in directly next.

Inside the example, use the ImGui panel to:

- edit the prompt,
- switch backend lane (proof/production),
- in Proof: create fixture images, write a preview preset, and pick built-in/preset
  generator mode,
- in Production: browse for a `.gguf` generator checkpoint and run inference.

The right side shows the generated image, fixture thumbnails and dry-run loss curve
in proof mode. The script path is still useful for repeatable setup and dry-run
validation:

```powershell
..\scripts\create-tiny-gan-preset.bat
..\scripts\create-tiny-gan-fixtures.bat -OutputPath bin\data\datasets\tiny-fixtures -Count 8
..\scripts\train-tiny-gan.bat -DryRun -Dataset C:\path\to\images -Epochs 2 -DryRunBatchesPerEpoch 3
..\scripts\train-tiny-gan.bat -DryRun -Dataset bin\data\datasets\tiny-fixtures -OutputPreset bin\data\models\tiny-preview-trained.ofxggmlgan -WritePreviewPreset -Force
..\scripts\run-gan-example.bat -DryRun
..\scripts\run-gan-example.bat -DryRun -PreviewPreset -ForcePreviewPreset
..\scripts\run-gan-example.bat -Build -Generator bin\data\models\tiny-mlp.ofxggmlgan
..\scripts\run-gan-example.bat -Build -Generator C:\path\to\generator.gguf
```

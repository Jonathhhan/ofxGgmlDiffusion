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

`ofxGgmlDiffusionGgufGanBackend` is the explicit production lane for supported
exported `.gguf` checkpoints and is intentionally split from the proof workflow so
training tools and presets cannot mask production path requirements. The first
supported external checkpoint is the small
[`gguf-org/pixel`](https://huggingface.co/gguf-org/pixel) Pixel/DCGAN model.
Other `.gguf` files are rejected until their architecture has a matching loader.
Pixel/DCGAN is unconditional, so prompt text is used as deterministic sample
variation rather than semantic text guidance.

Inside the example, use the ImGui panel to:

- edit the prompt,
- randomize each run by default or lock a seed for repeatable output,
- enable animation, which walks the seed at the selected FPS,
- switch backend lane (proof/production),
- in Proof: create fixture images, write a preview preset, and pick built-in/preset
  generator mode,
- in Production: browse for a `.gguf` generator checkpoint and run inference.

When `OFXGGML_GAN_GENERATOR` points at a `.gguf` file, the example starts in
Production automatically. When it points at a `.ofxggmlgan` preset, the example
stays in Proof and selects the preset lane. If Production is selected without a
GGUF generator and a proof preset exists, Run falls back to that proof preset
instead of stopping at a missing-model warning.

To try the public Pixel/DCGAN checkpoint without manually downloading model files:

```powershell
..\scripts\download-pixel-gan-model.bat
..\scripts\run-gan-example.bat -Build -Generator bin\data\models\pixel-model-f16.gguf
```

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

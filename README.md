# ofxGgmlDiffusion

`ofxGgmlDiffusion` is the companion addon for Stable Diffusion, SDXL, Flux-style
diffusion, GAN-style image generation, image-to-image, inpainting, and
control/image conditioning workflows on top of `ofxGgmlCore`.

`ofxGgmlCore` stays the dependency. This addon owns diffusion-specific workflow code so core can stay small and boring.

Family map: https://jonathhhan.github.io/ofxGgmlCore/

## First Milestone

- define small request/result/config types
- keep one root-level smoke example
- keep generated models, media, builds, and IDE files out of git
- validate the addon with local headless tests

The first migration from `ofxStableDiffusion` keeps the useful typed surface:
context settings, model-family labels, text/image/inpaint/upscale modes,
LoRA/control image descriptors, identity adapter descriptors, prompt cleanup,
and request validation. The large native wrapper, vendored
stable-diffusion.cpp tree, generated media, and experimental app workflows stay
out until they are deliberately reintroduced.

Native `stable-diffusion.cpp` runtime files are generated locally:

```powershell
scripts\build-stable-diffusion.bat
scripts\build-stable-diffusion.bat -DryRun
scripts\build-stable-diffusion.bat -CpuOnly
scripts\build-stable-diffusion.bat -Cuda
scripts\build-stable-diffusion.bat -BundledGgml
```

The script defaults to `-Auto`, reuses ggml from sibling addon
`../ofxGgmlCore`, enables only backends that are available on the current
machine, and installs generated files under `libs/stable-diffusion`. Pass
`-BundledGgml` only when you intentionally want stable-diffusion.cpp to build
against its own vendored ggml copy.

Generated source, build trees, libraries, executables, models, and output media
stay out of git.

## Native Backend

`ofxGgmlDiffusionNativeBackend` is the first bridge to the generated
`stable-diffusion.cpp` runtime. It compiles as a clear unavailable stub until
the native runtime is generated and the app build defines
`OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION`.

The first bridge intentionally supports text-to-image only and returns generated
pixels in `ofxGgmlDiffusionResult::images`. Use
`ofxGgmlDiffusionImageUtils::toPixels()` or `saveFirstImage()` to display or
save those pixels from an openFrameworks app. Image-to-image, inpainting, and
deeper native progress integration should be layered on after this boundary
stays boring.

`ofxGgmlDiffusionAsyncRunner` wraps the native backend in a worker thread for
openFrameworks examples. `cancel()` marks the pending result as cancelled once
stable-diffusion.cpp returns control; the current C API path does not interrupt
an in-flight sampling step.

## Identity Adapters

PhotoMaker belongs in `ofxGgmlDiffusion`, not a separate addon for now. It is a
diffusion identity-personalization adapter: configure it on the request, keep
the model and reference images local, and let the native bridge wire it once the
stable-diffusion.cpp C API surface is settled.

```cpp
auto request = ofxGgmlDiffusionUtils::makeTextToImageRequest("portrait of img");
request.modelFamily = ofxGgmlDiffusionModelFamily::SDXL;
request.identityAdapter = ofxGgmlDiffusionUtils::makePhotoMakerAdapter(
	"models/photomaker.safetensors",
	{"references/person-01.jpg", "references/person-02.jpg"},
	"img");
```

The current native bridge returns a clear error instead of silently ignoring an
identity adapter. The typed request surface and validation are ready first; the
actual PhotoMaker call should be wired after confirming the installed
stable-diffusion.cpp header.

## Diffusers Reference

Hugging Face Diffusers is a useful design reference for this addon, but not a
runtime dependency. Borrow the shape: pipelines compose schedulers, model
families, adapters, and media helpers. Keep the implementation local C++ around
generated stable-diffusion.cpp binaries and explicit user-provided assets.

Good ideas to mirror carefully:

- pipeline names should describe workflows, not upstream implementation details
- schedulers should stay explicit request/config choices
- adapters such as LoRA, ControlNet, and PhotoMaker should be typed request
  data
- examples should show one complete workflow without becoming a model zoo

## GAN Image Generation

GANs for image generation belong here, not in `ofxGgmlMusic`. Treat them as a
separate image-generation backend family beside diffusion: useful for exported
generator inference, style-specific image synthesis, super-resolution, or
image-to-image experiments.

The public API now has the first boring boundary for that work:

- `ofxGgmlDiffusionBackendFamily::GAN` marks the request as a GAN image workflow
- `ofxGgmlDiffusionGanSettings` carries generator path, latent settings,
  optional conditioning image, and class label metadata
- `makeGanImageRequest()` creates a validated text-to-image GAN request
- `ofxGgmlDiffusionImageGenerationBackend` is the shared backend interface for
  future diffusion, GAN, or external image runtimes
- `ofxGgmlDiffusionTinyGanBackend` is the first real ggml proof: a fixed tiny
  MLP generator that runs when Core's ggml headers/libs are installed
- `scripts\create-tiny-gan-preset.*` writes a small `.ofxggmlgan` preset file
  for testing exported-generator loading without committing model binaries
- `ofxGgmlDiffusionTinyGanTraining.*` and `scripts\train-tiny-gan.*` define the
  first dry-run-only training loop plan, including paired discriminator and
  generator update counts, so dataset/output/optimizer assumptions are checked
  before real adversarial training is attempted
- the training dry-run scans a dataset directory recursively and reports
  supported image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tga`) plus ignored
  files before any real training work starts
- `ofxGgmlDiffusionUnavailableImageGenerationBackend` lets examples fail
  clearly until a real generator backend is installed

The tiny backend is deterministic and not trained; it proves the graph and pixel
path, not model quality. The stable path should continue with inference from a
known local generator graph or an external bridge. Full in-addon adversarial training should remain
experimental until ggml training/autograd support is proven locally with focused
tests.

## Example

`ofxGgmlDiffusionPromptExample` is a root-level text-to-image example. Generate
it with the openFrameworks projectGenerator using addons `ofxGgmlDiffusion`, `ofxGgmlCore`, and `ofxImGui`, set `OFXGGML_DIFFUSION_MODEL` or place a model at
`bin/data/models/model.safetensors`, then press `R` to run one generation.
The example runs generation on a worker thread; press `C` to cancel the pending
result.

`ofxGgmlDiffusionGanExample` is a root-level GAN request example. Generate it
with the same addons, set `OFXGGML_GAN_GENERATOR` or place a generator at
`bin/data/models/generator.gguf`, then press `R` to validate and run the request.
It reports a clear unavailable-backend result until a real GAN runtime is wired.

```powershell
scripts\run-diffusion-example.bat -DryRun
scripts\run-diffusion-example.bat -Build -Model C:\path\to\model.safetensors
scripts\create-tiny-gan-preset.bat
scripts\train-tiny-gan.bat -DryRun -Dataset C:\path\to\images -OutputPreset ofxGgmlDiffusionGanExample\bin\data\models\tiny-trained.ofxggmlgan -Epochs 2 -DryRunBatchesPerEpoch 3
scripts\run-gan-example.bat -DryRun
scripts\run-gan-example.bat -Build -Generator ofxGgmlDiffusionGanExample\bin\data\models\tiny-mlp.ofxggmlgan
```

## Dependencies

- openFrameworks
- `ofxGgmlCore`
- `ofxImGui` for examples

## Validate

```powershell
scripts\validate-local.bat
```

On macOS/Linux:

```sh
./scripts/validate-local.sh
```

## Boundary

Keep diffusion-specific preprocessing, postprocessing, model launch, identity
adapter integration, media handling, and examples here. Move code down into
`ofxGgmlCore` only when it becomes a stable, domain-neutral primitive with
focused tests.

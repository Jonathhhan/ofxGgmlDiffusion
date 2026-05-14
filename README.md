# ofxGgmlDiffusion

`ofxGgmlDiffusion` is the companion addon for Stable Diffusion, SDXL, Flux-style
diffusion, GAN-style image generation, image-to-image, inpainting, and
control/image conditioning workflows on top of `ofxGgmlCore`.

`ofxGgmlCore` stays the dependency. This addon owns diffusion-specific workflow code so core can stay small and boring.

Family map: https://jonathhhan.github.io/ofxGgmlCore/

Current addon API version: `1.0.1`.

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

For diffusion-lane planning and future backend boundaries, see
[docs/DIFFUSION_WORKFLOWS.md](docs/DIFFUSION_WORKFLOWS.md).

Native `stable-diffusion.cpp` runtime files are generated locally:

```powershell
scripts\build-stable-diffusion.bat
scripts\build-stable-diffusion.bat -DryRun
scripts\build-stable-diffusion.bat -CpuOnly
scripts\build-stable-diffusion.bat -Cuda
scripts\build-stable-diffusion.bat -BundledGgml
scripts\test-stable-diffusion-native.bat
scripts\doctor-diffusion.bat
```

The script defaults to `-Auto`, reuses ggml from sibling addon
`../ofxGgmlCore`, enables only backends that are available on the current
machine, and installs generated files under `libs/stable-diffusion`. Pass
`-BundledGgml` only when you intentionally want stable-diffusion.cpp to build
against its own vendored ggml copy.

Generated source, build trees, libraries, executables, models, and output media
stay out of git.

`scripts\test-stable-diffusion-native.bat` is a model-free smoke test. It
compiles and links the addon bridge against the generated native header/lib,
confirms the PhotoMaker C API fields are present, then checks the explicit
missing-model and missing-context error paths. Use it after
`scripts\build-stable-diffusion.bat` to verify the local runtime boundary before
downloading large diffusion models.

`scripts\doctor-diffusion.bat` checks the local runtime, generated example
project, diffusion model env var, and optional PhotoMaker env vars without
starting generation. Run it before a heavy model smoke.

## Native Backend

`ofxGgmlDiffusionNativeBackend` is the first bridge to the generated
`stable-diffusion.cpp` runtime. It compiles as a clear unavailable stub until
the native runtime is generated and the app build defines
`OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION`. It implements the shared
`ofxGgmlDiffusionImageGenerationBackend` interface and can be constructed with
`ofxGgmlMakeNativeDiffusionImageGenerationBackend()`, so examples and future
pipeline helpers can choose between diffusion, GAN, and external image backends
through one boundary.

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

ofxGgmlDiffusionContextSettings context;
context.modelPath = "models/sdxl.safetensors";
context.photoMakerPath = request.identityAdapter.modelPath;

ofxGgmlDiffusionImage image;
if (ofxGgmlDiffusionImageUtils::loadImage("references/person-01.jpg", image)) {
	request.identityAdapter.referenceImages.push_back(image);
}
```

The current native bridge returns a clear error instead of silently ignoring an
identity adapter. The typed request surface and validation are ready first; the
native capability smoke now confirms that the installed stable-diffusion.cpp
header exposes the required PhotoMaker context and image parameter fields. The
native bridge now passes decoded `referenceImages` into
`sd_img_gen_params_t::pm_params`. Path-only references stay as metadata until an
example or app loads them with `ofxGgmlDiffusionImageUtils::loadImage()`.

The prompt example can opt into this path with environment variables:

```powershell
$env:OFXGGML_PHOTOMAKER_MODEL="C:\models\photomaker.safetensors"
$env:OFXGGML_PHOTOMAKER_REFS="C:\refs\person-01.jpg;C:\refs\person-02.jpg"
scripts\run-diffusion-example.bat -Build
```

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
  stable-diffusion.cpp, GAN, or external image runtimes
- `ofxGgmlDiffusionTinyGanBackend` is the first real ggml proof: a fixed tiny
  MLP generator that runs when Core's ggml headers/libs are installed
- `scripts\create-tiny-gan-preset.*` writes a small `.ofxggmlgan` preset file
  for testing exported-generator loading without committing model binaries
- `ofxGgmlDiffusionTinyGanTraining.*` and `scripts\train-tiny-gan.*` define the
  first dry-run-only training loop plan, including paired discriminator and
  generator update counts, so dataset/output/optimizer assumptions are checked
  before real adversarial training is attempted
- the training dry-run scans a dataset directory recursively and reports
  supported image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tga`, `.ppm`) plus ignored
  files before any real training work starts
- `scripts\create-tiny-gan-fixtures.*` writes deterministic 64x64 ASCII PPM
  fixture images for scanner tests and future toy training experiments
- `ofxGgmlDiffusionLoadTinyGanPpmImage()` loads those fixture images and creates
  normalized float pixels in `[-1, 1]` for the future discriminator path
- `ofxGgmlDiffusionRunTinyGanDiscriminatorForward()` is a deterministic tiny
  classifier preview over normalized fixture pixels; it is not trained yet
- `ofxGgmlDiffusionTinyGanBinaryCrossEntropy()` and
  `ofxGgmlDiffusionComputeTinyGanLossPair()` define the first real/fake loss
  helpers used by the dry-run trace
- `ofxGgmlDiffusionPreviewTinyGanWeightUpdate()` previews one deterministic
  scalar update from those losses without writing trained weights
- `ofxGgmlDiffusionApplyTinyGanPresetPreviewUpdate()` and
  `scripts\train-tiny-gan.bat -WritePreviewPreset` can write an explicit
  preview `.ofxggmlgan` preset while still keeping training dry-run-only
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
scripts\create-tiny-gan-fixtures.bat -OutputPath ofxGgmlDiffusionGanExample\bin\data\datasets\tiny-fixtures -Count 8
scripts\train-tiny-gan.bat -DryRun -Dataset C:\path\to\images -OutputPreset ofxGgmlDiffusionGanExample\bin\data\models\tiny-trained.ofxggmlgan -Epochs 2 -DryRunBatchesPerEpoch 3
scripts\train-tiny-gan.bat -DryRun -Dataset ofxGgmlDiffusionGanExample\bin\data\datasets\tiny-fixtures -OutputPreset ofxGgmlDiffusionGanExample\bin\data\models\tiny-preview-trained.ofxggmlgan -WritePreviewPreset -Force
scripts\run-gan-example.bat -DryRun
scripts\run-gan-example.bat -DryRun -PreviewPreset -ForcePreviewPreset
scripts\run-gan-example.bat -Build -Generator ofxGgmlDiffusionGanExample\bin\data\models\tiny-mlp.ofxggmlgan
```

## Dependencies

- openFrameworks
- `ofxGgmlCore`
- `ofxImGui` for examples

## Validate

```powershell
scripts\doctor-diffusion.bat
scripts\validate-local.bat
```

On macOS/Linux:

```sh
./scripts/doctor-diffusion.sh
./scripts/validate-local.sh
```

## Boundary

Keep diffusion-specific preprocessing, postprocessing, model launch, identity
adapter integration, media handling, and examples here. Move code down into
`ofxGgmlCore` only when it becomes a stable, domain-neutral primitive with
focused tests.

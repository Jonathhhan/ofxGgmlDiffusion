# ofxGgmlDiffusion

`ofxGgmlDiffusion` is the companion addon for Stable Diffusion, SDXL, Flux-style diffusion, image-to-image, inpainting, and control/image conditioning workflows on top of `ofxGgmlCore`.

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

## Example

`ofxGgmlDiffusionPromptExample` is a root-level text-to-image example. Generate
it with the openFrameworks projectGenerator using addons `ofxGgmlDiffusion`, `ofxGgmlCore`, and `ofxImGui`, set `OFXGGML_DIFFUSION_MODEL` or place a model at
`bin/data/models/model.safetensors`, then press `R` to run one generation.
The example runs generation on a worker thread; press `C` to cancel the pending
result.

```powershell
scripts\run-diffusion-example.bat -DryRun
scripts\run-diffusion-example.bat -Build -Model C:\path\to\model.safetensors
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

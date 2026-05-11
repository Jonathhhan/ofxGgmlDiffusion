# ofxGgmlDiffusion

`ofxGgmlDiffusion` is the companion addon for Stable Diffusion, SDXL, Flux-style diffusion, image-to-image, inpainting, and control/image conditioning workflows on top of `ofxGgmlCore`.

`ofxGgmlCore` stays the dependency. This addon owns diffusion-specific workflow code so core can stay small and boring.

## First Milestone

- define small request/result/config types
- keep one root-level smoke example
- keep generated models, media, builds, and IDE files out of git
- validate the addon with local headless tests

The first migration from `ofxStableDiffusion` keeps the useful typed surface:
context settings, model-family labels, text/image/inpaint/upscale modes,
LoRA/control image descriptors, prompt cleanup, and request validation. The
large native wrapper, vendored stable-diffusion.cpp tree, generated media, and
experimental app workflows stay out until they are deliberately reintroduced.

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
threaded/cancellable generation should be layered on after this boundary stays
boring.

## Example

`ofxGgmlDiffusionPromptExample` is a root-level text-to-image example. Generate
it with the openFrameworks projectGenerator using addons `ofxGgmlDiffusion` and
`ofxGgmlCore`, set `OFXGGML_DIFFUSION_MODEL` or place a model at
`bin/data/models/model.safetensors`, then press `R` to run one generation.

```powershell
scripts\run-diffusion-example.bat -DryRun
scripts\run-diffusion-example.bat -Build -Model C:\path\to\model.safetensors
```

## Dependencies

- openFrameworks
- `ofxGgmlCore`

## Validate

```powershell
scripts\validate-local.bat
```

On macOS/Linux:

```sh
./scripts/validate-local.sh
```

## Boundary

Keep diffusion-specific preprocessing, postprocessing, model launch, media handling, and examples here. Move code down into `ofxGgmlCore` only when it becomes a stable, domain-neutral primitive with focused tests.

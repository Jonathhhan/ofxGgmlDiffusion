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

## Example

`ofxGgmlDiffusionPromptExample` is a root-level prompt conditioning smoke test. Generate it with the openFrameworks projectGenerator using addons `ofxGgmlDiffusion` and `ofxGgmlCore`.

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

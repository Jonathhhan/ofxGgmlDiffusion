# ofxGgmlDiffusion

`ofxGgmlDiffusion` is the companion addon for Stable Diffusion, SDXL, Flux-style diffusion, image-to-image, inpainting, and control/image conditioning workflows on top of `ofxGgmlCore`.

`ofxGgmlCore` stays the dependency. This addon owns diffusion-specific workflow code so core can stay small and boring.

## First Milestone

- define small request/result types
- keep one root-level smoke example
- keep generated models, media, builds, and IDE files out of git
- validate the addon with local headless tests

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
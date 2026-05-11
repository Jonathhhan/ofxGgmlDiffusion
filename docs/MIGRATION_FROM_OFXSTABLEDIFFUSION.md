# Migration From ofxStableDiffusion

`ofxStableDiffusion` proved several useful ideas, but it also mixed the addon
API, native runtime, generated binaries, sample media, and large workflow
experiments in one repository.

`ofxGgmlDiffusion` starts smaller:

- keep typed context/request/result objects
- keep prompt cleanup and request validation helpers
- keep LoRA and control-image descriptors
- keep root-level examples only
- generate native stable-diffusion.cpp binaries locally while reusing
  `ofxGgmlCore` ggml by default
- do not commit model files, generated images/videos, native build output, or
  vendored upstream source

## Old To New

| Old `ofxStableDiffusion` idea | New home |
| --- | --- |
| `ofxStableDiffusionContextSettings` | `ofxGgmlDiffusionContextSettings` |
| `ofxStableDiffusionImageRequest` | `ofxGgmlDiffusionRequest` |
| prompt cleanup helpers | `ofxGgmlDiffusionUtils::cleanPrompt()` |
| request validation helpers | `ofxGgmlDiffusionUtils::validate()` |
| LoRA descriptors | `ofxGgmlDiffusionLora` |
| ControlNet descriptors | `ofxGgmlDiffusionControlImage` |
| native wrapper | `ofxGgmlDiffusionNativeBackend` first text-to-image bridge |
| worker thread and cancellation | future backend layer |
| large GUI and video workflows | later focused companion examples |

The first native bridge now accepts `ofxGgmlDiffusionRequest` for text-to-image.
The next migration step is image saving, then threaded/cancellable generation.

`scripts/build-stable-diffusion.*` now owns the generated runtime setup path.
It clones/builds/installs upstream locally only when the user runs it, points
stable-diffusion.cpp at sibling `ofxGgmlCore` ggml by default, and its dry-run
mode is covered by local validation.

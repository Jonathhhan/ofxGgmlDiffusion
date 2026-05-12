# Changelog

## Unreleased

- Constrained `scripts\build-stable-diffusion.* -Auto` to enable GPU backends
  only when both the host toolchain and the sibling `ofxGgmlCore` ggml backend
  library are present.
- Added an optional headless native bridge smoke test for compiling and linking
  against the generated `stable-diffusion.cpp` header/lib without requiring a
  model download.
- Connected `ofxGgmlDiffusionNativeBackend` to the shared image generation
  backend interface and factory path used by other image backends.
- Added native capability reporting for installed stable-diffusion.cpp
  PhotoMaker fields and exposed `photoMakerPath` on context settings.
- Added decoded PhotoMaker reference images, native `pm_params` mapping, and
  optional PhotoMaker environment setup in the prompt example.
- Added `scripts\doctor-diffusion.*` to report local runtime, model,
  generated-project, and PhotoMaker setup state before heavy generation.

## 1.0.1 - 2026-05-12

- Added independent Diffusion addon version metadata.
- Exposed version metadata through the public umbrella header.
- Documented the release checklist, release policy, and `v1.0.1` scope.
- Kept generated stable-diffusion.cpp runtime files, model files, and generated
  media as local-only state.

## 1.0.0

- Started `ofxGgmlDiffusion` as the companion addon for Stable Diffusion,
  SDXL, Flux-style diffusion, GAN-style image generation, image-to-image,
  inpainting, and conditioning workflows on top of `ofxGgmlCore`.
- Added typed diffusion request/result/config helpers, native
  stable-diffusion.cpp setup scripts, a text-to-image bridge boundary, async
  runner, PhotoMaker request surface, GAN request types, tiny GAN backend,
  fixture generation, and dry-run tiny GAN training contracts.

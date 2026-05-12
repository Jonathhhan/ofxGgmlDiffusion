# Changelog

## Unreleased

- Constrained `scripts\build-stable-diffusion.* -Auto` to enable GPU backends
  only when both the host toolchain and the sibling `ofxGgmlCore` ggml backend
  library are present.

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

# Diffusion Workflow Boundaries

`ofxGgmlDiffusion` owns local image-generation workflows for the ofxGgml
ecosystem. This document is for Codex, GitHub Copilot, Hermes Agent, and human
contributors planning diffusion-lane work before changing runtime behavior.

This guide intentionally follows the split rule from the legacy/reference
`ofxGgml` docs: broad domain workflows, model-specific preprocessing,
generated media, and heavy optional dependencies belong in companion addons.
Shared code should move down only when it is stable, domain-neutral,
dependency-light, and covered by focused tests.

## Owned workflow surface

This addon may define:

- Stable Diffusion, SDXL, Flux-style, and related image-generation request
  shapes
- `stable-diffusion.cpp` setup, native bridge, capability checks, and smoke
  tests
- text-to-image, image-to-image, inpainting, upscale, and control-image
  workflow planning
- LoRA, ControlNet, PhotoMaker, and identity-adapter request descriptors
- GAN image-generation boundaries and tiny deterministic proof backends
- image conversion, saving, and openFrameworks example handoff helpers
- model-local setup docs and generated-artifact cleanup rules

## Not owned here

Keep these responsibilities out of `ofxGgmlDiffusion`:

- ggml setup, backend selection, and runtime discovery owned by `ofxGgmlCore`
- text, audio, video, segmentation, music, RAG, or agent orchestration UX
- committed model weights, generated images, generated videos, native build
  trees, upstream source clones, or media caches
- reusable GitHub Actions policy owned by `ofxGgmlWorkflows`
- generic image understanding workflows owned by `ofxGgmlVision`

## Legacy reference use

Use the old `ofxGgml` repository as a reference archive, not as a managed
automation target. Good material to reuse includes:

- core-versus-companion split rules
- generated-artifact hygiene patterns
- focused example expectations
- release and validation discipline

Do not copy broad all-in-one workflow assumptions from the reference repo into
this lane. Translate useful ideas into small diffusion-specific docs,
validation checks, or examples.

## Planning handoff

Before changing diffusion behavior, write down:

```text
Workflow:
Backend family:
Model files:
Adapter files:
Generated local artifacts:
User-visible output:
Out of scope:
Validation:
```

Runtime changes should name the backend family involved, whether the work uses
diffusion, GAN, or external image generation, and which local artifacts must be
generated before the workflow can run.

## Validation ladder

Use the smallest command that proves the changed layer:

| Change type | Suggested validation |
| --- | --- |
| Docs or planning only | `scripts\validate-local.bat` |
| Native setup script changes | `scripts\test-stable-diffusion-setup-dry-run.bat` |
| Native bridge boundary | `scripts\test-stable-diffusion-native.bat -DryRun` |
| Ecosystem runtime smoke evidence | `scripts\run-diffusion-runtime-smoke.bat -Json -SummaryOnly` |
| Example launch path | `scripts\test-launch-dry-run.bat` |
| Local setup diagnosis | `scripts\doctor-diffusion.bat` |
| Request, image, or GAN helper changes | `scripts\test-addon.bat` |

`scripts\run-diffusion-runtime-smoke.*` is intentionally model-free for now. It
proves the installed `stable-diffusion.cpp` bridge can compile, link, expose the
expected capability surface, and fail cleanly without a model or loaded context.
Real text-to-image and image-to-video generation smokes should build on this
entrypoint once model paths, output locations, runtime backends, and media
cleanup are explicit.

## Safe first tasks

Good early diffusion-lane tasks are:

- documenting backend and artifact assumptions
- adding dry-run validation around generated native runtime paths
- clarifying how identity adapters and reference images are loaded
- improving example handoff docs without adding model zoo behavior
- keeping GAN work bounded to deterministic fixtures and explicit presets

Avoid broadening runtime behavior until model files, generated artifacts,
output media, and the validation command are explicit.

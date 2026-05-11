# Roadmap

## Current Milestone

- Seed the companion addon skeleton.
- Keep `ofxGgmlDiffusionPromptExample` as the first root-level smoke example.
- Keep `ofxGgmlCore` as the only required library dependency; examples may depend on `ofxImGui`.
- Add local validation and headless tests.
- Migrate the typed request/result/config shape from the old
  `ofxStableDiffusion` addon without copying its native runtime or generated
  artifacts.
- Add generated `stable-diffusion.cpp` setup/build scripts with dry-run
  validation, defaulting to ggml from sibling `ofxGgmlCore`.
- Connect the first opt-in stable-diffusion.cpp bridge adapter for
  text-to-image, including unavailable-runtime test coverage.
- Add image/pixel saving helpers for generated `ofxGgmlDiffusionImage` data.
- Convert the root prompt smoke example into a text-to-image example that
  accepts a user-provided local model.
- Add generated-project build/run scripts plus launch dry-run validation for
  the text-to-image example.
- Add threaded generation with result-level cancellation around the native
  bridge.
- Add a typed identity adapter request surface for PhotoMaker without creating
  a separate addon.
- Adopt Diffusers-inspired vocabulary for future pipelines, schedulers, model
  families, and adapters without taking a Python runtime dependency.
- Define GAN-style image generation as an image backend family that belongs in
  this addon, separate from music/audio generation.
- Add the first `ofxGgmlDiffusionImageGenerationBackend` interface, GAN request
  settings, and unavailable fallback backend.
- Add `ofxGgmlDiffusionGanExample` as a root-level GAN request smoke example
  with dry-run launch script coverage.
- Add `ofxGgmlDiffusionTinyGanBackend`, a fixed tiny ggml MLP generator proof
  that produces image pixels when Core's ggml runtime is installed.
- Add `.ofxggmlgan` tiny preset loading plus scripts to create local preset
  files without committing generated model assets.
- Add a dry-run-only tiny GAN training contract and script with paired
  discriminator/generator update planning so the future dataset/output/optimizer
  path is explicit before real training code lands.
- Add a recursive dry-run dataset scanner for tiny GAN training that reports
  supported image counts and ignored files without requiring real training.
- Add generated 64x64 PPM fixture images for tiny GAN scanner tests and future
  toy training experiments without committing binary assets.
- Add a tiny PPM loader and normalization path so fixture images can feed a
  discriminator forward pass.
- Add a deterministic tiny discriminator forward pass over normalized fixture
  pixels.
- Add binary classification loss helpers around discriminator probabilities.
- Add the first deterministic toy weight-update preview without enabling
  non-dry-run training.
- Connect the toy update preview to a tiny preset/checkpoint mutation path while
  keeping file writes opt-in and tested.
- Use the preview preset output in the GAN example run path as a selectable
  local generator descriptor.

## Next Milestones

- Wire PhotoMaker into the native stable-diffusion.cpp bridge after confirming
  the installed C API fields.
- Connect the existing stable-diffusion.cpp bridge to the shared image backend
  interface.
- Choose a small exported generator format and fixture model for the first
  non-builtin GAN inference backend.
- Implement a toy discriminator/training loop behind the tiny GAN training
  contract with real weight updates, then require repeatable fixture outputs
  before exposing non-dry-run training.
- Add a tiny GAN example UI panel that can switch between built-in, preset, and
  preview-preset generator descriptors.
- Sketch a small C++ pipeline layer after text-to-image and PhotoMaker are
  proven, so examples can compose request setup without duplicating glue.
- Add native progress reporting once there is a reliable stable-diffusion.cpp
  callback path.
- Add focused tests around request/result helpers.
- Document the `clone -> setup -> run` path from a new user's point of view.

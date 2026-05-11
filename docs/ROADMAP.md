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

## Next Milestones

- Wire PhotoMaker into the native stable-diffusion.cpp bridge after confirming
  the installed C API fields.
- Sketch a backend-family interface that can host diffusion, GAN, and external
  image generation backends without changing `ofxGgmlCore`.
- Sketch a small C++ pipeline layer after text-to-image and PhotoMaker are
  proven, so examples can compose request setup without duplicating glue.
- Add native progress reporting once there is a reliable stable-diffusion.cpp
  callback path.
- Add focused tests around request/result helpers.
- Document the `clone -> setup -> run` path from a new user's point of view.

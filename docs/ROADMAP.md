# Roadmap

## Current Milestone

- Seed the companion addon skeleton.
- Keep `ofxGgmlDiffusionPromptExample` as the first root-level smoke example.
- Keep `ofxGgmlCore` as the only required addon dependency.
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

## Next Milestones

- Add threaded/cancellable generation around the native bridge.
- Add a generated-project build script for the text-to-image example.
- Add focused tests around request/result helpers.
- Document the `clone -> setup -> run` path from a new user's point of view.

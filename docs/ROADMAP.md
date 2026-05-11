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

## Next Milestones

- Add one useful openFrameworks example that runs text-to-image with
  user-provided model assets.
- Add image saving helpers for generated `ofxGgmlDiffusionImage` data.
- Add threaded/cancellable generation around the native bridge.
- Add focused tests around request/result helpers.
- Document the `clone -> setup -> run` path from a new user's point of view.

# Roadmap

## Current Milestone

- Seed the companion addon skeleton.
- Keep `ofxGgmlDiffusionPromptExample` as the first root-level smoke example.
- Keep `ofxGgmlCore` as the only required addon dependency.
- Add local validation and headless tests.
- Migrate the typed request/result/config shape from the old
  `ofxStableDiffusion` addon without copying its native runtime or generated
  artifacts.

## Next Milestones

- Add `scripts/build-stable-diffusion.*` for a generated local native runtime.
- Connect the first stable-diffusion.cpp bridge adapter.
- Add one useful openFrameworks example that runs with user-provided assets.
- Add focused tests around request/result helpers.
- Document the `clone -> setup -> run` path from a new user's point of view.

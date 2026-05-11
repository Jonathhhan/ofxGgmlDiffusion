# Architecture

`ofxGgmlDiffusion` owns diffusion-specific workflow code. It should use `ofxGgmlCore` for stable runtime primitives and keep model-family workflow details out of core.

## Dependency Direction

```text
openFrameworks app
  -> ofxGgmlDiffusion
      -> ofxGgmlCore
```

No dependency should point from `ofxGgmlCore` back to `ofxGgmlDiffusion`.

## Owned Here

- diffusion-specific request/result helpers
- model-specific preprocessing and postprocessing
- focused root-level examples
- local media/model workflow documentation

## Not Owned Here

- ggml runtime setup and backend selection
- generic tensor, graph, model metadata, and result types
- unrelated companion workflows
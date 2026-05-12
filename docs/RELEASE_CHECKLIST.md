# Release Checklist

Use this before tagging or announcing an `ofxGgmlDiffusion` release. The goal is
to prove the typed image-generation surface and keep generated runtime/media
state local.

## Fresh Clone Layout

From the openFrameworks `addons` folder:

```powershell
git clone https://github.com/Jonathhhan/ofxGgmlCore.git
git clone https://github.com/Jonathhhan/ofxGgmlDiffusion.git
cd ofxGgmlDiffusion
```

Expected layout:

```text
addons/
  ofxGgmlCore/
  ofxGgmlDiffusion/
  ofxImGui/
```

## Optional stable-diffusion.cpp Runtime

The native diffusion runtime is opt-in. Runtime files are generated locally:

```powershell
scripts\build-stable-diffusion.bat
scripts\setup-stable-diffusion.bat
```

macOS/Linux:

```sh
./scripts/build-stable-diffusion.sh
./scripts/setup-stable-diffusion.sh
```

Expected generated local paths:

```text
libs/stable-diffusion/
models/
data/models/
```

These paths must not be staged for release.

## GAN Smoke

Run dry-run coverage for the GAN lane:

```powershell
scripts\create-tiny-gan-preset.bat
scripts\create-tiny-gan-fixtures.bat -OutputPath C:\temp\gan-fixtures -Count 8
scripts\train-tiny-gan.bat -DryRun -Dataset C:\temp\gan-fixtures -OutputPreset C:\temp\tiny-preview.ofxggmlgan
scripts\run-gan-example.bat -DryRun -PreviewPreset
```

Generated fixture images, presets, checkpoints, and preview outputs must not be
staged.

## Local Validation

Run:

```powershell
scripts\validate-local.bat
```

macOS/Linux:

```sh
./scripts/validate-local.sh
```

For a pre-tag release candidate gate:

```powershell
scripts\release-candidate.bat
```

macOS/Linux:

```sh
./scripts/release-candidate.sh
```

## Before Tagging

- `git status --short --ignored` shows no unexpected generated outputs
- no native runtime binaries, model files, generated media, generated OF project
  files, fixtures, presets, or build outputs are staged
- `CHANGELOG.md` has an entry for the release
- `docs/releases/vX.Y.Z.md` matches the release scope
- `docs/MIGRATION_FROM_OFXSTABLEDIFFUSION.md` still describes the old addon
  boundary accurately
- release notes distinguish working bridge/proof paths from future full runtime
  claims

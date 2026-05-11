param()

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Assert-Contains {
	param(
		[string[]]$Output,
		[string]$Needle,
		[string]$Label
	)
	$text = $Output -join "`n"
	if (!$text.Contains($Needle)) {
		throw "$Label did not contain expected text: $Needle`n$text"
	}
}

function Assert-NotContains {
	param(
		[string[]]$Output,
		[string]$Needle,
		[string]$Label
	)
	$text = $Output -join "`n"
	if ($text.Contains($Needle)) {
		throw "$Label contained unexpected text: $Needle`n$text"
	}
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$scratchDir = Join-Path ([System.IO.Path]::GetTempPath()) "ofxGgmlDiffusion-launch-dry-run"
New-Item -ItemType Directory -Force -Path $scratchDir | Out-Null
$modelPath = Join-Path $scratchDir "dry-run-model.safetensors"
if (!(Test-Path -LiteralPath $modelPath -PathType Leaf)) {
	New-Item -ItemType File -Path $modelPath | Out-Null
}

Write-Step "Diffusion example dry-run"
$output = & (Join-Path $scriptRoot "run-diffusion-example.ps1") -DryRun -Model $modelPath *>&1 |
	ForEach-Object { $_.ToString() }

Assert-Contains $output "Using diffusion model: $modelPath" "Diffusion example dry-run"
Assert-Contains $output "Executable:" "Diffusion example dry-run"
Assert-Contains $output "Output:" "Diffusion example dry-run"
Assert-NotContains $output "Starting ofxGgmlDiffusionPromptExample" "Diffusion example dry-run"

$generatorPath = Join-Path $scratchDir "dry-run-generator.gguf"
if (!(Test-Path -LiteralPath $generatorPath -PathType Leaf)) {
	New-Item -ItemType File -Path $generatorPath | Out-Null
}

Write-Step "GAN example dry-run"
$output = & (Join-Path $scriptRoot "run-gan-example.ps1") -DryRun -Generator $generatorPath *>&1 |
	ForEach-Object { $_.ToString() }

Assert-Contains $output "Using GAN generator: $generatorPath" "GAN example dry-run"
Assert-Contains $output "Executable:" "GAN example dry-run"
Assert-Contains $output "Output:" "GAN example dry-run"
Assert-NotContains $output "Starting ofxGgmlDiffusionGanExample" "GAN example dry-run"

Write-Step "Tiny GAN preset dry-run"
$presetPath = Join-Path $scratchDir "tiny-mlp.ofxggmlgan"
$output = & (Join-Path $scriptRoot "create-tiny-gan-preset.ps1") -DryRun -OutputPath $presetPath *>&1 |
	ForEach-Object { $_.ToString() }

Assert-Contains $output "Tiny GAN preset output: $presetPath" "Tiny GAN preset dry-run"
Assert-Contains $output "Dry run complete" "Tiny GAN preset dry-run"

Write-Step "Tiny GAN fixture dry-run"
$fixturePath = Join-Path $scratchDir "tiny-gan-fixtures-dry-run"
$output = & (Join-Path $scriptRoot "create-tiny-gan-fixtures.ps1") -DryRun -OutputPath $fixturePath -Count 3 *>&1 |
	ForEach-Object { $_.ToString() }

Assert-Contains $output "Tiny GAN fixture dataset output: $fixturePath" "Tiny GAN fixture dry-run"
Assert-Contains $output "Image count: 3" "Tiny GAN fixture dry-run"
Assert-Contains $output "Dry run complete" "Tiny GAN fixture dry-run"

Write-Step "Tiny GAN training dry-run"
$trainedPresetPath = Join-Path $scratchDir "tiny-trained.ofxggmlgan"
$datasetPath = Join-Path $scratchDir "tiny-gan-dataset"
& (Join-Path $scriptRoot "create-tiny-gan-fixtures.ps1") -OutputPath $datasetPath -Count 3 -Force | Out-Null
Set-Content -LiteralPath (Join-Path $datasetPath "notes.txt") -Value "dry-run"
$output = & (Join-Path $scriptRoot "train-tiny-gan.ps1") -DryRun -Dataset $datasetPath -OutputPreset $trainedPresetPath -Epochs 2 -BatchSize 8 -DryRunBatchesPerEpoch 3 *>&1 |
	ForEach-Object { $_.ToString() }

Assert-Contains $output "Tiny GAN training dry-run" "Tiny GAN training dry-run"
Assert-Contains $output "Dataset: $datasetPath" "Tiny GAN training dry-run"
Assert-Contains $output "Dataset exists: yes" "Tiny GAN training dry-run"
Assert-Contains $output "Dataset images: 3" "Tiny GAN training dry-run"
Assert-Contains $output "Dataset unsupported files: 1" "Tiny GAN training dry-run"
Assert-Contains $output "Output preset: $trainedPresetPath" "Tiny GAN training dry-run"
Assert-Contains $output "Discriminator architecture: tiny-mlp-binary-classifier" "Tiny GAN training dry-run"
Assert-Contains $output "Planned discriminator updates: 6" "Tiny GAN training dry-run"
Assert-Contains $output "Planned generator updates: 6" "Tiny GAN training dry-run"
Assert-Contains $output "Dry run complete" "Tiny GAN training dry-run"

Write-Step "Tiny GAN preview preset write"
$previewPresetPath = Join-Path $scratchDir "tiny-preview-trained.ofxggmlgan"
$output = & (Join-Path $scriptRoot "train-tiny-gan.ps1") -DryRun -Dataset $datasetPath -OutputPreset $previewPresetPath -Epochs 2 -BatchSize 8 -DryRunBatchesPerEpoch 3 -WritePreviewPreset -Force *>&1 |
	ForEach-Object { $_.ToString() }

Assert-Contains $output "Wrote preview preset: $previewPresetPath" "Tiny GAN preview preset write"
Assert-Contains $output "Dry run complete; no weights were trained" "Tiny GAN preview preset write"
if (!(Test-Path -LiteralPath $previewPresetPath -PathType Leaf)) {
	throw "Tiny GAN preview preset was not written: $previewPresetPath"
}
Assert-Contains (Get-Content -LiteralPath $previewPresetPath) "architecture=tiny-mlp" "Tiny GAN preview preset file"
Assert-Contains (Get-Content -LiteralPath $previewPresetPath) "w1Seed=" "Tiny GAN preview preset file"

Write-Step "Launch dry-run smoke coverage passed"

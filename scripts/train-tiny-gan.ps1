param(
	[string]$Dataset = "",
	[string]$OutputPreset = "",
	[int]$LatentSize = 512,
	[int]$HiddenSize = 96,
	[int]$Epochs = 1,
	[int]$BatchSize = 4,
	[float]$LearningRate = 0.001,
	[switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot

if ([string]::IsNullOrWhiteSpace($OutputPreset)) {
	$OutputPreset = Join-Path $addonRoot "ofxGgmlDiffusionGanExample\bin\data\models\tiny-trained.ofxggmlgan"
}

$OutputPreset = [System.IO.Path]::GetFullPath($OutputPreset)

if (!$DryRun) {
	throw "Tiny GAN training is experimental and currently supports -DryRun only."
}
if ($LatentSize -lt 8 -or $LatentSize -gt 1024) {
	throw "LatentSize must be between 8 and 1024."
}
if ($HiddenSize -lt 8 -or $HiddenSize -gt 512) {
	throw "HiddenSize must be between 8 and 512."
}
if ($Epochs -lt 1) {
	throw "Epochs must be at least 1."
}
if ($BatchSize -lt 1) {
	throw "BatchSize must be at least 1."
}
if ($LearningRate -le 0) {
	throw "LearningRate must be greater than zero."
}

Write-Step "Tiny GAN training dry-run"
Write-Step "Dataset: $(if ([string]::IsNullOrWhiteSpace($Dataset)) { '(synthetic placeholder)' } else { $Dataset })"
Write-Step "Output preset: $OutputPreset"
Write-Step "Architecture: tiny-mlp"
Write-Step "Image size: 64x64"
Write-Step "LatentSize: $LatentSize"
Write-Step "HiddenSize: $HiddenSize"
Write-Step "Epochs: $Epochs"
Write-Step "BatchSize: $BatchSize"
Write-Step "LearningRate: $LearningRate"
Write-Step "Dry run complete; no files were written and no weights were trained"

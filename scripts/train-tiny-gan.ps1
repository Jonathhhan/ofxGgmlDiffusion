param(
	[string]$Dataset = "",
	[string]$OutputPreset = "",
	[int]$LatentSize = 512,
	[int]$HiddenSize = 96,
	[int]$DiscriminatorHiddenSize = 96,
	[int]$Epochs = 1,
	[int]$BatchSize = 4,
	[int]$DryRunBatchesPerEpoch = 1,
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
if ($DiscriminatorHiddenSize -lt 8 -or $DiscriminatorHiddenSize -gt 512) {
	throw "DiscriminatorHiddenSize must be between 8 and 512."
}
if ($Epochs -lt 1) {
	throw "Epochs must be at least 1."
}
if ($BatchSize -lt 1) {
	throw "BatchSize must be at least 1."
}
if ($DryRunBatchesPerEpoch -lt 1 -or $DryRunBatchesPerEpoch -gt 128) {
	throw "DryRunBatchesPerEpoch must be between 1 and 128."
}
if ($LearningRate -le 0) {
	throw "LearningRate must be greater than zero."
}

$supportedExtensions = @(".png", ".jpg", ".jpeg", ".bmp", ".tga", ".ppm")
$datasetLabel = if ([string]::IsNullOrWhiteSpace($Dataset)) { "(synthetic placeholder)" } else { $Dataset }
$datasetExists = $false
$datasetIsDirectory = $false
$datasetImageCount = 0
$datasetUnsupportedCount = 0
if (![string]::IsNullOrWhiteSpace($Dataset)) {
	$datasetExists = Test-Path -LiteralPath $Dataset
	if ($datasetExists) {
		$datasetIsDirectory = Test-Path -LiteralPath $Dataset -PathType Container
		if ($datasetIsDirectory) {
			$files = Get-ChildItem -LiteralPath $Dataset -Recurse -File
			foreach ($file in $files) {
				if ($supportedExtensions -contains $file.Extension.ToLowerInvariant()) {
					$datasetImageCount += 1
				} else {
					$datasetUnsupportedCount += 1
				}
			}
		}
	}
}

$plannedUpdates = $Epochs * $DryRunBatchesPerEpoch
$finalDiscriminatorLoss = [Math]::Round(1.3862944 - 0.18, 6)
$finalGeneratorLoss = [Math]::Round(0.6931472 - 0.07, 6)

Write-Step "Tiny GAN training dry-run"
Write-Step "Dataset: $datasetLabel"
Write-Step "Dataset exists: $(if ($datasetExists) { 'yes' } else { 'no' })"
Write-Step "Dataset directory: $(if ($datasetIsDirectory) { 'yes' } else { 'no' })"
Write-Step "Dataset images: $datasetImageCount"
Write-Step "Dataset unsupported files: $datasetUnsupportedCount"
Write-Step "Output preset: $OutputPreset"
Write-Step "Generator architecture: tiny-mlp"
Write-Step "Discriminator architecture: tiny-mlp-binary-classifier"
Write-Step "Image size: 64x64"
Write-Step "LatentSize: $LatentSize"
Write-Step "GeneratorHiddenSize: $HiddenSize"
Write-Step "DiscriminatorHiddenSize: $DiscriminatorHiddenSize"
Write-Step "Epochs: $Epochs"
Write-Step "BatchSize: $BatchSize"
Write-Step "DryRunBatchesPerEpoch: $DryRunBatchesPerEpoch"
Write-Step "Planned discriminator updates: $plannedUpdates"
Write-Step "Planned generator updates: $plannedUpdates"
Write-Step "LearningRate: $LearningRate"
Write-Step "Final dry-run discriminator loss: $finalDiscriminatorLoss"
Write-Step "Final dry-run generator loss: $finalGeneratorLoss"
Write-Step "Dry run complete; no files were written and no weights were trained"

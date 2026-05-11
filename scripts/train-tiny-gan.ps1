param(
	[string]$Dataset = "",
	[string]$InputPreset = "",
	[string]$OutputPreset = "",
	[int]$LatentSize = 512,
	[int]$HiddenSize = 96,
	[int]$DiscriminatorHiddenSize = 96,
	[int]$Epochs = 1,
	[int]$BatchSize = 4,
	[int]$DryRunBatchesPerEpoch = 1,
	[float]$LearningRate = 0.001,
	[switch]$WritePreviewPreset,
	[switch]$Force,
	[switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Get-Bce {
	param(
		[float]$Probability,
		[bool]$TargetReal
	)
	if ([float]::IsNaN($Probability) -or [float]::IsInfinity($Probability)) {
		$Probability = 0.5
	}
	$clamped = [Math]::Min(0.999999, [Math]::Max(0.000001, $Probability))
	if ($TargetReal) {
		return -[Math]::Log($clamped)
	}
	return -[Math]::Log(1.0 - $clamped)
}

function Get-Preset {
	param([string]$Path)
	$preset = @{
		version = 1
		architecture = "tiny-mlp"
		latentSize = $LatentSize
		hiddenSize = $HiddenSize
		w1Seed = 17
		b1Seed = 29
		w2Seed = 43
		b2Seed = 71
		latentScale = 1.0
		w1Scale = 0.18
		b1Scale = 0.08
		w2Scale = 0.09
		b2Scale = 0.03
	}
	if ([string]::IsNullOrWhiteSpace($Path)) {
		return $preset
	}
	$resolved = [System.IO.Path]::GetFullPath($Path)
	if (!(Test-Path -LiteralPath $resolved -PathType Leaf)) {
		throw "InputPreset was not found: $resolved"
	}
	foreach ($line in Get-Content -LiteralPath $resolved) {
		$value = ($line -replace "#.*$", "").Trim()
		if ([string]::IsNullOrWhiteSpace($value)) {
			continue
		}
		$parts = $value.Split("=", 2)
		if ($parts.Count -ne 2) {
			throw "Invalid tiny GAN preset line: $line"
		}
		$key = $parts[0].Trim()
		$preset[$key] = $parts[1].Trim()
	}
	if ([int]$preset.version -ne 1 -or [string]$preset.architecture -ne "tiny-mlp") {
		throw "InputPreset must be a tiny-mlp version 1 preset."
	}
	return $preset
}

function Get-SeedOffset {
	param(
		[float]$Gradient,
		[int]$Fallback
	)
	if ([float]::IsNaN($Gradient) -or [float]::IsInfinity($Gradient)) {
		return $Fallback
	}
	return [Math]::Max(1, [int]([Math]::Abs($Gradient) * 100000.0) + $Fallback)
}

function Get-ClampedScale {
	param([float]$Value)
	if ([float]::IsNaN($Value) -or [float]::IsInfinity($Value)) {
		return 0.01
	}
	return [Math]::Min(2.0, [Math]::Max(0.001, $Value))
}

function Write-Preset {
	param(
		[hashtable]$Preset,
		[string]$Path
	)
	$content = @"
# ofxGgmlDiffusion tiny GAN preset v1
version=$($Preset.version)
architecture=$($Preset.architecture)
latentSize=$($Preset.latentSize)
hiddenSize=$($Preset.hiddenSize)
w1Seed=$($Preset.w1Seed)
b1Seed=$($Preset.b1Seed)
w2Seed=$($Preset.w2Seed)
b2Seed=$($Preset.b2Seed)
latentScale=$($Preset.latentScale)
w1Scale=$($Preset.w1Scale)
b1Scale=$($Preset.b1Scale)
w2Scale=$($Preset.w2Scale)
b2Scale=$($Preset.b2Scale)
"@
	$outputDir = Split-Path -Parent $Path
	New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
	Set-Content -LiteralPath $Path -Value $content -Encoding ASCII
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
$realProbability = 0.68
$fakeProbability = 0.38
$generatorFakeProbability = 0.57
$realLoss = Get-Bce -Probability $realProbability -TargetReal $true
$fakeLoss = Get-Bce -Probability $fakeProbability -TargetReal $false
$finalDiscriminatorLoss = [Math]::Round(0.5 * ($realLoss + $fakeLoss), 6)
$finalGeneratorLoss = [Math]::Round((Get-Bce -Probability $generatorFakeProbability -TargetReal $true), 6)
$discriminatorGradient = $fakeLoss - $realLoss
$generatorGradient = -$finalGeneratorLoss
$initialDiscriminatorWeight = 0.125
$initialGeneratorWeight = -0.075
$updatedDiscriminatorWeight = $initialDiscriminatorWeight - ($LearningRate * $discriminatorGradient)
$updatedGeneratorWeight = $initialGeneratorWeight - ($LearningRate * $generatorGradient)

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
Write-Step "Preview discriminator weight: $updatedDiscriminatorWeight"
Write-Step "Preview generator weight: $updatedGeneratorWeight"

if ($WritePreviewPreset) {
	if ((Test-Path -LiteralPath $OutputPreset -PathType Leaf) -and !$Force) {
		throw "Preview preset already exists: $OutputPreset. Pass -Force to overwrite."
	}
	$preset = Get-Preset -Path $InputPreset
	$generatorDelta = $updatedGeneratorWeight - $initialGeneratorWeight
	$discriminatorDelta = $updatedDiscriminatorWeight - $initialDiscriminatorWeight
	$preset.latentScale = Get-ClampedScale ([float]$preset.latentScale + ($generatorDelta * 0.10))
	$preset.w1Scale = Get-ClampedScale ([float]$preset.w1Scale + $generatorDelta)
	$preset.b1Scale = Get-ClampedScale ([float]$preset.b1Scale + ($generatorDelta * 0.25))
	$preset.w2Scale = Get-ClampedScale ([float]$preset.w2Scale + ($generatorDelta * 0.75))
	$preset.b2Scale = Get-ClampedScale ([float]$preset.b2Scale + ($generatorDelta * 0.25) + ($discriminatorDelta * 0.10))
	$preset.w1Seed = [uint32]([uint32]$preset.w1Seed + (Get-SeedOffset -Gradient $generatorGradient -Fallback 11))
	$preset.b1Seed = [uint32]([uint32]$preset.b1Seed + (Get-SeedOffset -Gradient $generatorGradient -Fallback 17))
	$preset.w2Seed = [uint32]([uint32]$preset.w2Seed + (Get-SeedOffset -Gradient $discriminatorGradient -Fallback 23))
	$preset.b2Seed = [uint32]([uint32]$preset.b2Seed + (Get-SeedOffset -Gradient $discriminatorGradient -Fallback 29))
	Write-Preset -Preset $preset -Path $OutputPreset
	Write-Step "Wrote preview preset: $OutputPreset"
	Write-Step "Dry run complete; no weights were trained"
} else {
	Write-Step "Dry run complete; no files were written and no weights were trained"
}

param(
	[string]$Generator = $env:OFXGGML_GAN_GENERATOR,
	[switch]$PreviewPreset,
	[string]$PreviewDataset = "",
	[string]$PreviewPresetPath = "",
	[switch]$ForcePreviewPreset,
	[switch]$Build,
	[switch]$DryRun,
	[string]$Configuration = "Release",
	[string]$Platform = "x64"
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Normalize-PathText {
	param([string]$Path)
	if ([string]::IsNullOrWhiteSpace($Path)) {
		return ""
	}
	return [System.IO.Path]::GetFullPath($Path)
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot
$exampleName = "ofxGgmlDiffusionGanExample"
$exampleRoot = Join-Path $addonRoot $exampleName
$exampleExe = Join-Path $exampleRoot "bin\$exampleName.exe"
$outputPath = Join-Path $exampleRoot "bin\data\outputs\$exampleName.png"
$defaultPreviewDataset = Join-Path $exampleRoot "bin\data\datasets\tiny-fixtures"
$defaultPreviewPresetPath = Join-Path $exampleRoot "bin\data\models\tiny-preview-trained.ofxggmlgan"

if ($env:OFXGGML_LAUNCH_DRY_RUN_ONLY -eq "1") {
	$Build = $false
	$DryRun = $true
}

if ($Build) {
	& (Join-Path $scriptRoot "build-diffusion-example.ps1") -Configuration $Configuration -Platform $Platform -Example $exampleName
	if ($LASTEXITCODE -ne 0) {
		exit $LASTEXITCODE
	}
}

if ($PreviewPreset) {
	if ([string]::IsNullOrWhiteSpace($PreviewDataset)) {
		$PreviewDataset = $defaultPreviewDataset
	}
	if ([string]::IsNullOrWhiteSpace($PreviewPresetPath)) {
		$PreviewPresetPath = $defaultPreviewPresetPath
	}
	$PreviewDataset = Normalize-PathText $PreviewDataset
	$PreviewPresetPath = Normalize-PathText $PreviewPresetPath

	if (!(Test-Path -LiteralPath $PreviewDataset -PathType Container)) {
		Write-Step "Creating tiny GAN fixtures: $PreviewDataset"
		& (Join-Path $scriptRoot "create-tiny-gan-fixtures.ps1") -OutputPath $PreviewDataset -Count 8 -Force
	}

	Write-Step "Writing tiny GAN preview preset: $PreviewPresetPath"
	if ($ForcePreviewPreset) {
		& (Join-Path $scriptRoot "train-tiny-gan.ps1") -DryRun -Dataset $PreviewDataset -OutputPreset $PreviewPresetPath -Epochs 2 -DryRunBatchesPerEpoch 3 -WritePreviewPreset -Force
	} else {
		& (Join-Path $scriptRoot "train-tiny-gan.ps1") -DryRun -Dataset $PreviewDataset -OutputPreset $PreviewPresetPath -Epochs 2 -DryRunBatchesPerEpoch 3 -WritePreviewPreset
	}
	$Generator = $PreviewPresetPath
}

if ((Test-Path -LiteralPath $exampleExe -PathType Leaf)) {
	$exampleExeExists = $true
} elseif ($DryRun) {
	$exampleExeExists = $false
	Write-Warning "GAN example executable was not found: $exampleExe"
} else {
	throw "GAN example executable was not found: $exampleExe. Run scripts\run-gan-example.bat -Build or generate the project first."
}

$Generator = Normalize-PathText $Generator
if (![string]::IsNullOrWhiteSpace($Generator)) {
	$env:OFXGGML_GAN_GENERATOR = $Generator
	Write-Step "Using GAN generator: $Generator"
} else {
	Write-Warning "No GAN generator was provided. Set OFXGGML_GAN_GENERATOR, place a supported GGUF at ofxGgmlDiffusionGanExample\bin\data\models\generator.gguf, or run scripts\download-pixel-gan-model.bat."
}

if ($DryRun) {
	Write-Step "Executable: $exampleExe"
	Write-Step "Executable exists: $(if ($exampleExeExists) { 'yes' } else { 'no' })"
	Write-Step "Output: $outputPath"
	return
}

Write-Step "Starting $exampleName"
& $exampleExe
exit $LASTEXITCODE

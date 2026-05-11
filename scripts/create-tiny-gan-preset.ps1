param(
	[string]$OutputPath = "",
	[int]$LatentSize = 512,
	[int]$HiddenSize = 96,
	[int]$W1Seed = 17,
	[int]$B1Seed = 29,
	[int]$W2Seed = 43,
	[int]$B2Seed = 71,
	[float]$LatentScale = 1.0,
	[float]$W1Scale = 0.18,
	[float]$B1Scale = 0.08,
	[float]$W2Scale = 0.09,
	[float]$B2Scale = 0.03,
	[switch]$Force,
	[switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
	$OutputPath = Join-Path $addonRoot "ofxGgmlDiffusionGanExample\bin\data\models\tiny-mlp.ofxggmlgan"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)

if ($LatentSize -lt 8 -or $LatentSize -gt 1024) {
	throw "LatentSize must be between 8 and 1024."
}
if ($HiddenSize -lt 8 -or $HiddenSize -gt 512) {
	throw "HiddenSize must be between 8 and 512."
}

$content = @"
# ofxGgmlDiffusion tiny GAN preset v1
version=1
architecture=tiny-mlp
latentSize=$LatentSize
hiddenSize=$HiddenSize
w1Seed=$W1Seed
b1Seed=$B1Seed
w2Seed=$W2Seed
b2Seed=$B2Seed
latentScale=$LatentScale
w1Scale=$W1Scale
b1Scale=$B1Scale
w2Scale=$W2Scale
b2Scale=$B2Scale
"@

if ($DryRun) {
	Write-Step "Tiny GAN preset output: $OutputPath"
	Write-Step "LatentSize: $LatentSize"
	Write-Step "HiddenSize: $HiddenSize"
	Write-Step "Dry run complete; no files were written"
	return
}

if ((Test-Path -LiteralPath $OutputPath -PathType Leaf) -and !$Force) {
	throw "Tiny GAN preset already exists: $OutputPath. Pass -Force to overwrite."
}

$outputDir = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Set-Content -LiteralPath $OutputPath -Value $content -Encoding ASCII
Write-Step "Wrote tiny GAN preset: $OutputPath"

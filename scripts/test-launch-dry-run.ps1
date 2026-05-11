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

Write-Step "Launch dry-run smoke coverage passed"

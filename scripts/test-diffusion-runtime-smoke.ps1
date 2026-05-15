param()

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Assert-Contains {
	param(
		[string]$Text,
		[string]$Needle,
		[string]$Label
	)
	if (!$Text.Contains($Needle)) {
		throw "$Label did not contain expected text: $Needle`n$Text"
	}
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $scriptRoot "run-diffusion-runtime-smoke.ps1"

Write-Step "Diffusion runtime smoke dry-run"
$textOutput = & $script -DryRun 2>&1 6>&1 | Out-String
Assert-Contains $textOutput "ofxGgmlDiffusion runtime smoke plan" "runtime smoke dry-run"
Assert-Contains $textOutput "Backend: stable-diffusion.cpp" "runtime smoke dry-run"
Assert-Contains $textOutput "Ready:" "runtime smoke dry-run"
Assert-Contains $textOutput "Dry run complete; no files were changed" "runtime smoke dry-run"

Write-Step "Diffusion runtime smoke JSON dry-run"
$jsonOutput = & $script -DryRun -Json -SummaryOnly 2>&1 6>&1 | Out-String
$summary = $jsonOutput | ConvertFrom-Json
if ($summary.Name -ne "ofxGgmlDiffusion runtime smoke") {
	throw "Unexpected runtime smoke name: $($summary.Name)"
}
if ($summary.Backend -ne "stable-diffusion.cpp") {
	throw "Unexpected runtime smoke backend: $($summary.Backend)"
}
if (!($summary.NextCommands -contains "scripts\run-diffusion-runtime-smoke.bat -Json -SummaryOnly")) {
	throw "JSON dry-run did not include the runtime smoke command."
}

Write-Step "Diffusion runtime smoke contract passed"

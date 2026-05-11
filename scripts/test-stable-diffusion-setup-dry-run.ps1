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
$script = Join-Path $scriptRoot "build-stable-diffusion.ps1"

Write-Step "stable-diffusion default dry-run"
$defaultOutput = & $script -DryRun 2>&1 6>&1 | Out-String
Assert-Contains $defaultOutput "Dry run: stable-diffusion.cpp setup plan" "default dry-run"
Assert-Contains $defaultOutput "mode: Auto" "default dry-run"
Assert-Contains $defaultOutput "ggml: ofxGgmlCore" "default dry-run"
Assert-Contains $defaultOutput "-DSD_USE_SYSTEM_GGML=ON" "default dry-run"
Assert-Contains $defaultOutput "-DSD_BUILD_EXAMPLES=OFF" "default dry-run"
Assert-Contains $defaultOutput "Dry run complete; no files were changed" "default dry-run"

Write-Step "stable-diffusion CPU-only dry-run"
$cpuOutput = & $script -DryRun -CpuOnly 2>&1 6>&1 | Out-String
Assert-Contains $cpuOutput "mode: CpuOnly" "CPU-only dry-run"
Assert-Contains $cpuOutput "CUDA=OFF" "CPU-only dry-run"
Assert-Contains $cpuOutput "Vulkan=OFF" "CPU-only dry-run"

Write-Step "stable-diffusion bundled ggml dry-run"
$bundledOutput = & $script -DryRun -CpuOnly -BundledGgml 2>&1 6>&1 | Out-String
Assert-Contains $bundledOutput "ggml: Bundled" "bundled ggml dry-run"

Write-Step "stable-diffusion examples dry-run"
$examplesOutput = & $script -DryRun -CpuOnly -BuildExamples 2>&1 6>&1 | Out-String
Assert-Contains $examplesOutput "-DSD_BUILD_EXAMPLES=ON" "examples dry-run"

Write-Step "stable-diffusion setup dry-run coverage passed"

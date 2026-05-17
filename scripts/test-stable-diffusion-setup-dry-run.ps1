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

function Assert-ContainsCollapsed {
	param(
		[string]$Text,
		[string]$Needle,
		[string]$Label
	)
	$collapsedText = $Text -replace "\s+", " "
	$collapsedNeedle = $Needle -replace "\s+", " "
	Assert-Contains $collapsedText $collapsedNeedle $Label
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
Assert-Contains $defaultOutput "source update: OFF" "default dry-run"
Assert-Contains $defaultOutput "git clone --recursive --depth 1 --branch master" "default dry-run"
Assert-Contains $defaultOutput "Dry run complete; no files were changed" "default dry-run"

Write-Step "stable-diffusion Core-constrained auto dry-run"
$fakeCore = Join-Path ([System.IO.Path]::GetTempPath()) "ofxGgmlDiffusion-fake-core"
$fakeInclude = Join-Path $fakeCore "libs\ggml\include"
$fakeLib = Join-Path $fakeCore "libs\ggml\lib"
New-Item -ItemType Directory -Force -Path $fakeInclude | Out-Null
New-Item -ItemType Directory -Force -Path $fakeLib | Out-Null
foreach ($file in @(
	(Join-Path $fakeInclude "ggml.h"),
	(Join-Path $fakeLib "ggml.lib"),
	(Join-Path $fakeLib "ggml-base.lib"),
	(Join-Path $fakeLib "ggml-cpu.lib")
)) {
	if (!(Test-Path -LiteralPath $file -PathType Leaf)) {
		New-Item -ItemType File -Path $file | Out-Null
	}
}
$coreConstrainedOutput = & $script -DryRun -OfxGgmlCorePath $fakeCore 2>&1 6>&1 | Out-String
Assert-Contains $coreConstrainedOutput "mode: Auto" "Core-constrained dry-run"
Assert-Contains $coreConstrainedOutput "CUDA=OFF" "Core-constrained dry-run"
Assert-Contains $coreConstrainedOutput "Vulkan=OFF" "Core-constrained dry-run"

Write-Step "stable-diffusion CPU-only dry-run"
$cpuOutput = & $script -DryRun -CpuOnly 2>&1 6>&1 | Out-String
Assert-Contains $cpuOutput "mode: CpuOnly" "CPU-only dry-run"
Assert-Contains $cpuOutput "CUDA=OFF" "CPU-only dry-run"
Assert-Contains $cpuOutput "Vulkan=OFF" "CPU-only dry-run"

Write-Step "stable-diffusion bundled ggml request prefers Core dry-run"
$bundledPreferredCoreOutput = & $script -DryRun -CpuOnly -BundledGgml -OfxGgmlCorePath $fakeCore 2>&1 6>&1 | Out-String
Assert-Contains $bundledPreferredCoreOutput "ggml: ofxGgmlCore" "bundled request with Core dry-run"
Assert-Contains $bundledPreferredCoreOutput "BundledGgml requested but ignored because ofxGgmlCore ggml is available" "bundled request with Core dry-run"

Write-Step "stable-diffusion bundled ggml fallback dry-run"
$missingCore = Join-Path ([System.IO.Path]::GetTempPath()) "ofxGgmlDiffusion-missing-core"
if (Test-Path -LiteralPath $missingCore) {
	Remove-Item -LiteralPath $missingCore -Recurse -Force
}
$bundledOutput = & $script -DryRun -CpuOnly -BundledGgml -OfxGgmlCorePath $missingCore 2>&1 6>&1 | Out-String
Assert-Contains $bundledOutput "ggml: Bundled" "bundled fallback dry-run"
Assert-Contains $bundledOutput "BundledGgml requested and ofxGgmlCore ggml was not found" "bundled fallback dry-run"

Write-Step "stable-diffusion examples dry-run"
$examplesOutput = & $script -DryRun -CpuOnly -BuildExamples 2>&1 6>&1 | Out-String
Assert-Contains $examplesOutput "-DSD_BUILD_EXAMPLES=ON" "examples dry-run"

Write-Step "stable-diffusion source update dry-run"
$updateOutput = & $script -DryRun -CpuOnly -Update 2>&1 6>&1 | Out-String
Assert-Contains $updateOutput "source update: ON" "source update dry-run"
Assert-Contains $updateOutput "git -C" "source update dry-run"
Assert-ContainsCollapsed $updateOutput "fetch --depth 1 origin master" "source update dry-run"
Assert-ContainsCollapsed $updateOutput "checkout --detach FETCH_HEAD" "source update dry-run"
Assert-ContainsCollapsed $updateOutput "submodule update --init --recursive --depth 1" "source update dry-run"

Write-Step "stable-diffusion setup dry-run coverage passed"

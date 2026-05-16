param(
	[string]$Model = $env:OFXGGML_DIFFUSION_MODEL,
	[switch]$Build,
	[switch]$DryRun,
	[string]$Configuration = "Release",
	[string]$Platform = "x64",
	[int]$Jobs = 1
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
$exampleRoot = Join-Path $addonRoot "ofxGgmlDiffusionPromptExample"
$exampleExe = Join-Path $exampleRoot "bin\ofxGgmlDiffusionPromptExample.exe"
$outputPath = Join-Path $exampleRoot "bin\data\outputs\ofxGgmlDiffusionPromptExample.png"

if ($env:OFXGGML_LAUNCH_DRY_RUN_ONLY -eq "1") {
	$Build = $false
	$DryRun = $true
}

if ($Build) {
	& (Join-Path $scriptRoot "build-diffusion-example.ps1") -Configuration $Configuration -Platform $Platform -Jobs $Jobs
	if ($LASTEXITCODE -ne 0) {
		exit $LASTEXITCODE
	}
}

if ((Test-Path -LiteralPath $exampleExe -PathType Leaf)) {
	$exampleExeExists = $true
} elseif ($DryRun) {
	$exampleExeExists = $false
	Write-Warning "Diffusion example executable was not found: $exampleExe"
} else {
	throw "Diffusion example executable was not found: $exampleExe. Run scripts\run-diffusion-example.bat -Build or scripts\build-diffusion-example.bat first."
}

$Model = Normalize-PathText $Model
if (![string]::IsNullOrWhiteSpace($Model)) {
	$env:OFXGGML_DIFFUSION_MODEL = $Model
	Write-Step "Using diffusion model: $Model"
} else {
	Write-Warning "No diffusion model was provided. Set OFXGGML_DIFFUSION_MODEL or place one at ofxGgmlDiffusionPromptExample\bin\data\models\model.safetensors."
}

if ($DryRun) {
	Write-Step "Executable: $exampleExe"
	Write-Step "Executable exists: $(if ($exampleExeExists) { 'yes' } else { 'no' })"
	Write-Step "Output: $outputPath"
	return
}

Write-Step "Starting ofxGgmlDiffusionPromptExample"
& $exampleExe
exit $LASTEXITCODE

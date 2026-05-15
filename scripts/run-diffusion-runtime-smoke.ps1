param(
	[string]$Configuration = "Release",
	[string]$BuildDir = "",
	[switch]$Clean,
	[switch]$DryRun,
	[switch]$Json,
	[switch]$SummaryOnly
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Test-WindowsHost {
	return !($IsLinux -or $IsMacOS)
}

function Get-PowerShellExecutable {
	$pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
	if ($pwsh) {
		return $pwsh.Source
	}

	$windowsPowerShell = Get-Command powershell -ErrorAction SilentlyContinue
	if ($windowsPowerShell) {
		return $windowsPowerShell.Source
	}

	throw "Could not find pwsh or powershell."
}

function Test-RuntimeSmokeReady {
	$stableHeader = Join-Path $addonRoot "libs\stable-diffusion\include\stable-diffusion.h"
	$stableLibrary = if (Test-WindowsHost) {
		Join-Path $addonRoot "libs\stable-diffusion\lib\stable-diffusion.lib"
	} else {
		Join-Path $addonRoot "libs/stable-diffusion/lib/libstable-diffusion.a"
	}
	$ggmlHeader = Join-Path $addonsRoot "ofxGgmlCore\libs\ggml\include\ggml.h"

	return (Test-Path -LiteralPath $stableHeader -PathType Leaf) -and
		(Test-Path -LiteralPath $stableLibrary -PathType Leaf) -and
		(Test-Path -LiteralPath $ggmlHeader -PathType Leaf)
}

function New-DryRunSummary {
	$ready = Test-RuntimeSmokeReady
	return [ordered]@{
		Name = "ofxGgmlDiffusion runtime smoke"
		Root = [string]$addonRoot
		Backend = "stable-diffusion.cpp"
		BuildDir = $BuildDir
		Ready = $ready
		NativeSmokeScript = $nativeScript
		NextCommands = @(
			"scripts\run-diffusion-runtime-smoke.bat -Json -SummaryOnly",
			"scripts\test-stable-diffusion-native.bat",
			"scripts\doctor-diffusion.bat"
		)
	}
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Resolve-Path (Join-Path $scriptRoot "..")
$addonsRoot = Split-Path -Parent $addonRoot
$nativeScript = Join-Path $scriptRoot "test-stable-diffusion-native.ps1"

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
	$BuildDir = Join-Path ([System.IO.Path]::GetTempPath()) "ofxGgmlDiffusion-runtime-smoke"
}

if ($DryRun) {
	$summary = New-DryRunSummary
	if ($Json) {
		$summary | ConvertTo-Json -Depth 5
		return
	}

	Write-Step "ofxGgmlDiffusion runtime smoke plan"
	Write-Host "  Backend: $($summary.Backend)"
	Write-Host "  BuildDir: $($summary.BuildDir)"
	Write-Host "  Ready: $($summary.Ready)"
	Write-Host "  Native smoke: $($summary.NativeSmokeScript)"
	Write-Host "  Next: $($summary.NextCommands[0])"
	Write-Step "Dry run complete; no files were changed"
	return
}

$started = Get-Date
$powerShell = Get-PowerShellExecutable
$arguments = @(
	"-NoProfile",
	"-ExecutionPolicy",
	"Bypass",
	"-File",
	$nativeScript,
	"-Configuration",
	$Configuration,
	"-BuildDir",
	$BuildDir
)
if ($Clean) {
	$arguments += "-Clean"
}

$output = @()
$exitCode = 0
try {
	$output = & $powerShell @arguments 2>&1 | ForEach-Object { "$_" }
	$exitCode = $LASTEXITCODE
} catch {
	$output += "$_"
	$exitCode = 1
}

$elapsedMs = [int]((Get-Date) - $started).TotalMilliseconds
$passed = ($exitCode -eq 0)
$summary = [ordered]@{
	Name = "ofxGgmlDiffusion runtime smoke"
	Passed = $passed
	Backend = "stable-diffusion.cpp"
	Configuration = $Configuration
	BuildDir = $BuildDir
	ResultCount = 1
	FailedCount = $(if ($passed) { 0 } else { 1 })
	ElapsedMs = $elapsedMs
	Error = $(if ($passed) { "" } else { ($output -join "`n") })
}

if ($Json) {
	if ($SummaryOnly) {
		$summary | ConvertTo-Json -Depth 5
	} else {
		[ordered]@{
			Summary = $summary
			Output = $output
		} | ConvertTo-Json -Depth 5
	}
} else {
	foreach ($line in $output) {
		Write-Host $line
	}
	Write-Step "ofxGgmlDiffusion runtime smoke summary"
	Write-Host "  Backend: $($summary.Backend)"
	Write-Host "  Passed: $($summary.Passed)"
	Write-Host "  ElapsedMs: $($summary.ElapsedMs)"
}

if (!$passed) {
	exit 1
}

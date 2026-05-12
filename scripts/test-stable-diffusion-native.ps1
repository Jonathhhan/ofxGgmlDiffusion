param(
	[string]$Configuration = "Release",
	[string]$BuildDir = "",
	[switch]$Clean,
	[switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Test-WindowsHost {
	return !($IsLinux -or $IsMacOS)
}

function Convert-ToCmdArgument {
	param([string]$Value)
	return '"' + ($Value -replace '"', '""') + '"'
}

function Invoke-CheckedNative {
	param(
		[string]$Step,
		[scriptblock]$Command
	)
	& $Command
	if ($LASTEXITCODE -ne 0) {
		throw "$Step failed with exit code $LASTEXITCODE"
	}
}

function Invoke-CheckedCmd {
	param(
		[string]$Step,
		[string]$Command
	)
	& cmd.exe /d /s /c $Command
	if ($LASTEXITCODE -ne 0) {
		throw "$Step failed with exit code $LASTEXITCODE"
	}
}

function Get-VisualStudioDevCmd {
	$candidates = New-Object System.Collections.Generic.List[string]
	$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
	if (Test-Path -LiteralPath $vswhere) {
		$installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
		if ($installPath) {
			$candidates.Add((Join-Path $installPath "Common7\Tools\VsDevCmd.bat"))
		}
	}

	foreach ($version in @("18", "17", "16")) {
		foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
			$candidates.Add("C:\Program Files\Microsoft Visual Studio\$version\$edition\Common7\Tools\VsDevCmd.bat")
			$candidates.Add("C:\Program Files (x86)\Microsoft Visual Studio\$version\$edition\Common7\Tools\VsDevCmd.bat")
		}
	}

	foreach ($candidate in $candidates) {
		if (Test-Path -LiteralPath $candidate) {
			return $candidate
		}
	}
	return ""
}

function Assert-File {
	param(
		[string]$Path,
		[string]$Label,
		[string]$Fix
	)
	if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
		throw "$Label was not found: $Path`nfix: $Fix"
	}
}

function Add-ExistingFile {
	param(
		[System.Collections.Generic.List[string]]$List,
		[string]$Path
	)
	if (Test-Path -LiteralPath $Path -PathType Leaf) {
		$List.Add($Path) | Out-Null
	}
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Resolve-Path (Join-Path $scriptRoot "..")
$addonsRoot = Split-Path -Parent $addonRoot
$coreRoot = Join-Path $addonsRoot "ofxGgmlCore"
$testsDir = Join-Path $addonRoot "tests"
$stableInclude = Join-Path $addonRoot "libs\stable-diffusion\include"
$stableHeader = Join-Path $stableInclude "stable-diffusion.h"
$stableLibrary = if (Test-WindowsHost) {
	Join-Path $addonRoot "libs\stable-diffusion\lib\stable-diffusion.lib"
} else {
	Join-Path $addonRoot "libs/stable-diffusion/lib/libstable-diffusion.a"
}
$ggmlInclude = Join-Path $coreRoot "libs\ggml\include"
$ggmlHeader = Join-Path $ggmlInclude "ggml.h"
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
	$BuildDir = Join-Path ([System.IO.Path]::GetTempPath()) "ofxGgmlDiffusion-native-smoke"
}

$extraLibraries = [System.Collections.Generic.List[string]]::new()
if (Test-WindowsHost) {
	$coreLibDir = Join-Path $coreRoot "libs\ggml\lib"
	foreach ($library in @("ggml.lib", "ggml-base.lib", "ggml-cpu.lib", "ggml-cuda.lib", "ggml-vulkan.lib", "ggml-opencl.lib")) {
		Add-ExistingFile -List $extraLibraries -Path (Join-Path $coreLibDir $library)
	}
	if ($env:CUDA_PATH) {
		$cudaLibDir = Join-Path $env:CUDA_PATH "lib\x64"
		foreach ($library in @("cublas.lib", "cudart.lib", "cuda.lib")) {
			Add-ExistingFile -List $extraLibraries -Path (Join-Path $cudaLibDir $library)
		}
	}
} else {
	$coreLibDir = Join-Path $coreRoot "libs/ggml/lib"
	foreach ($library in @("libggml.a", "libggml-base.a", "libggml-cpu.a", "libggml-cuda.a", "libggml-vulkan.a", "libggml-opencl.a")) {
		Add-ExistingFile -List $extraLibraries -Path (Join-Path $coreLibDir $library)
	}
}
$extraLibraryText = $extraLibraries -join ";"

if ($DryRun) {
	Write-Step "stable-diffusion.cpp native smoke plan"
	Write-Host "  tests: $testsDir"
	Write-Host "  build: $BuildDir"
	Write-Host "  include: $stableInclude"
	Write-Host "  library: $stableLibrary"
	Write-Host "  ggml include: $ggmlInclude"
	Write-Host "  extra libraries: $extraLibraryText"
	Write-Host "  clean: $(if ($Clean) { 'ON' } else { 'OFF' })"
	Write-Step "Dry run complete; no files were changed"
	return
}

Assert-File $stableHeader "stable-diffusion.cpp header" "scripts\build-stable-diffusion.bat"
Assert-File $stableLibrary "stable-diffusion.cpp library" "scripts\build-stable-diffusion.bat"
Assert-File $ggmlHeader "ofxGgmlCore ggml header" "..\ofxGgmlCore\scripts\setup-ggml.bat -Auto"
if ($extraLibraries.Count -lt 3) {
	throw "ofxGgmlCore ggml libraries were not found. Run ..\ofxGgmlCore\scripts\setup-ggml.bat -Auto first."
}

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
	Write-Step "Cleaning $BuildDir"
	Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

if (Test-WindowsHost) {
	$vsDevCmd = Get-VisualStudioDevCmd
	if ([string]::IsNullOrWhiteSpace($vsDevCmd)) {
		throw "Visual Studio C++ build tools were not found."
	}

	$configure = "cmake -S $(Convert-ToCmdArgument $testsDir) -B $(Convert-ToCmdArgument $BuildDir) -G $(Convert-ToCmdArgument "NMake Makefiles") -DCMAKE_BUILD_TYPE=$Configuration -DOFXGGMLDIFFUSION_BUILD_NATIVE_SMOKE=ON -DOFXGGMLDIFFUSION_STABLE_DIFFUSION_INCLUDE_DIR=$(Convert-ToCmdArgument $stableInclude) -DOFXGGMLDIFFUSION_STABLE_DIFFUSION_LIBRARY=$(Convert-ToCmdArgument $stableLibrary) -DOFXGGMLDIFFUSION_GGML_INCLUDE_DIR=$(Convert-ToCmdArgument $ggmlInclude) -DOFXGGMLDIFFUSION_EXTRA_LIBRARIES=$(Convert-ToCmdArgument $extraLibraryText)"
	$build = "cmake --build $(Convert-ToCmdArgument $BuildDir) --target ofxGgmlDiffusion_native_smoke"
	$exe = Join-Path $BuildDir "ofxGgmlDiffusion_native_smoke.exe"
	$run = "$(Convert-ToCmdArgument $exe)"
	$command = "call $(Convert-ToCmdArgument $vsDevCmd) -arch=x64 -host_arch=x64 >nul && $configure && $build && $run"

	Write-Step "Building and running stable-diffusion.cpp native smoke with Visual Studio tools"
	Invoke-CheckedCmd "stable-diffusion.cpp native smoke" $command
} else {
	Write-Step "Configuring stable-diffusion.cpp native smoke"
	Invoke-CheckedNative "cmake configure stable-diffusion.cpp native smoke" {
		cmake -S $testsDir -B $BuildDir -DCMAKE_BUILD_TYPE=$Configuration `
			-DOFXGGMLDIFFUSION_BUILD_NATIVE_SMOKE=ON `
			-DOFXGGMLDIFFUSION_STABLE_DIFFUSION_INCLUDE_DIR=$stableInclude `
			-DOFXGGMLDIFFUSION_STABLE_DIFFUSION_LIBRARY=$stableLibrary `
			-DOFXGGMLDIFFUSION_GGML_INCLUDE_DIR=$ggmlInclude `
			-DOFXGGMLDIFFUSION_EXTRA_LIBRARIES=$extraLibraryText
	}
	Write-Step "Building stable-diffusion.cpp native smoke"
	Invoke-CheckedNative "cmake build stable-diffusion.cpp native smoke" {
		cmake --build $BuildDir --target ofxGgmlDiffusion_native_smoke --config $Configuration
	}
	Write-Step "Running stable-diffusion.cpp native smoke"
	$exe = Join-Path $BuildDir "ofxGgmlDiffusion_native_smoke"
	Invoke-CheckedNative "stable-diffusion.cpp native smoke" {
		& $exe
	}
}

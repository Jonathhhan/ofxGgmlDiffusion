param(
	[string]$Repo = "https://github.com/leejet/stable-diffusion.cpp.git",
	[string]$Revision = "master",
	[string]$Configuration = "Release",
	[string]$Generator = "",
	[int]$Jobs = 0,
	[string]$SourceDir = "",
	[string]$BuildDir = "",
	[string]$InstallDir = "",
	[string]$OfxGgmlCorePath = "",
	[switch]$Auto,
	[Alias("Cpu")][switch]$CpuOnly,
	[Alias("Gpu")][switch]$Cuda,
	[switch]$Vulkan,
	[switch]$Metal,
	[switch]$OpenCL,
	[switch]$BundledGgml,
	[switch]$BuildExamples,
	[switch]$Shared,
	[switch]$Clean,
	[switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Get-CommandPathOrNull {
	param([string]$Name)
	try {
		return (Get-Command $Name -ErrorAction Stop).Source
	} catch {
		return $null
	}
}

function Test-WindowsHost {
	return !($IsLinux -or $IsMacOS)
}

function Test-CudaAvailable {
	if ($env:CUDA_PATH -and (Test-Path -LiteralPath $env:CUDA_PATH)) {
		return $true
	}
	return [bool](Get-CommandPathOrNull "nvcc")
}

function Test-VulkanAvailable {
	if ($env:VULKAN_SDK -and (Test-Path -LiteralPath $env:VULKAN_SDK)) {
		return $true
	}
	return [bool](Get-CommandPathOrNull "glslc") -or [bool](Get-CommandPathOrNull "vulkaninfo")
}

function Test-MetalAvailable {
	return $IsMacOS -and [bool](Get-CommandPathOrNull "xcrun")
}

function Test-OpenCLAvailable {
	return [bool]$env:OPENCL_ROOT -or [bool](Get-CommandPathOrNull "clinfo")
}

function Test-CoreGgmlLibraryAvailable {
	param(
		[string]$CorePath,
		[string]$WindowsLibraryName,
		[string]$UnixLibraryName
	)
	if ([string]::IsNullOrWhiteSpace($CorePath)) {
		return $false
	}
	$libDir = Join-Path $CorePath "libs\ggml\lib"
	if (!(Test-Path -LiteralPath $libDir -PathType Container)) {
		return $false
	}
	$libraryName = if (Test-WindowsHost) { $WindowsLibraryName } else { $UnixLibraryName }
	return (Test-Path -LiteralPath (Join-Path $libDir $libraryName) -PathType Leaf)
}

function Test-CoreGgmlBaseAvailable {
	param([string]$CorePath)
	if ([string]::IsNullOrWhiteSpace($CorePath)) {
		return $false
	}
	$includeDir = Join-Path $CorePath "libs\ggml\include"
	if (!(Test-Path -LiteralPath (Join-Path $includeDir "ggml.h") -PathType Leaf)) {
		return $false
	}
	if (Test-WindowsHost) {
		foreach ($libraryName in @("ggml.lib", "ggml-base.lib", "ggml-cpu.lib")) {
			if (!(Test-CoreGgmlLibraryAvailable -CorePath $CorePath -WindowsLibraryName $libraryName -UnixLibraryName "")) {
				return $false
			}
		}
	} else {
		foreach ($libraryName in @("libggml.a", "libggml-base.a", "libggml-cpu.a")) {
			if (!(Test-CoreGgmlLibraryAvailable -CorePath $CorePath -WindowsLibraryName "" -UnixLibraryName $libraryName)) {
				return $false
			}
		}
	}
	return $true
}

function Get-DefaultGenerator {
	if (![string]::IsNullOrWhiteSpace($Generator)) {
		return $Generator
	}
	if (Test-WindowsHost) {
		return "Visual Studio 18 2026"
	}
	return ""
}

function Invoke-Checked {
	param(
		[string]$Step,
		[string]$FilePath,
		[string[]]$Arguments
	)
	& $FilePath @Arguments
	if ($LASTEXITCODE -ne 0) {
		throw "$Step failed with exit code $LASTEXITCODE"
	}
}

function Convert-ToCMakePath {
	param([string]$Path)
	return ([System.IO.Path]::GetFullPath($Path) -replace "\\", "/")
}

function Add-RequiredLibraryPath {
	param(
		[System.Collections.Generic.List[string]]$Libraries,
		[string]$Path,
		[string]$Description
	)
	if (!(Test-Path -LiteralPath $Path)) {
		throw "$Description was not found at: $Path"
	}
	$Libraries.Add($Path)
}

function New-OfxGgmlCoreCmakePackage {
	param(
		[string]$CorePath,
		[string]$PackageRoot,
		[bool]$EnableCuda,
		[bool]$EnableVulkan
	)

	$corePath = [System.IO.Path]::GetFullPath($CorePath)
	$includeDir = Join-Path $corePath "libs\ggml\include"
	$libDir = Join-Path $corePath "libs\ggml\lib"
	$packageDir = Join-Path $PackageRoot "ggml"
	$libraries = [System.Collections.Generic.List[string]]::new()

	if (!(Test-Path -LiteralPath $includeDir -PathType Container)) {
		throw "ofxGgmlCore ggml headers were not found at: $includeDir"
	}
	if (!(Test-Path -LiteralPath $libDir -PathType Container)) {
		throw "ofxGgmlCore ggml libraries were not found at: $libDir"
	}

	if (Test-WindowsHost) {
		Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "ggml.lib") -Description "ofxGgmlCore ggml.lib"
		Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "ggml-base.lib") -Description "ofxGgmlCore ggml-base.lib"
		Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "ggml-cpu.lib") -Description "ofxGgmlCore ggml-cpu.lib"
		if ($EnableCuda) {
			Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "ggml-cuda.lib") -Description "ofxGgmlCore ggml-cuda.lib"
			$cudaLibDir = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "lib\x64" } else { "" }
			Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $cudaLibDir "cublas.lib") -Description "CUDA cublas.lib"
			Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $cudaLibDir "cudart.lib") -Description "CUDA cudart.lib"
			Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $cudaLibDir "cuda.lib") -Description "CUDA cuda.lib"
		}
		if ($EnableVulkan) {
			Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "ggml-vulkan.lib") -Description "ofxGgmlCore ggml-vulkan.lib"
			$vulkanLibDir = if ($env:VULKAN_SDK) { Join-Path $env:VULKAN_SDK "Lib" } else { "" }
			Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $vulkanLibDir "vulkan-1.lib") -Description "Vulkan vulkan-1.lib"
		}
	} else {
		Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "libggml.a") -Description "ofxGgmlCore libggml.a"
		Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "libggml-base.a") -Description "ofxGgmlCore libggml-base.a"
		Add-RequiredLibraryPath -Libraries $libraries -Path (Join-Path $libDir "libggml-cpu.a") -Description "ofxGgmlCore libggml-cpu.a"
	}

	New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
	$cmakeIncludeDir = Convert-ToCMakePath $includeDir
	$cmakeLibraryEntries = ($libraries | ForEach-Object { "`t`"" + (Convert-ToCMakePath $_) + "`"" }) -join "`n"
	$configPath = Join-Path $packageDir "ggml-config.cmake"
	$versionPath = Join-Path $packageDir "ggml-version.cmake"

	$configContent = @"
include_guard(GLOBAL)

set(ggml_FOUND TRUE)
set(_ofxggmlcore_include_dir "$cmakeIncludeDir")
set(_ofxggmlcore_libraries
$cmakeLibraryEntries
)

if (NOT TARGET ggml::ggml)
	add_library(ggml::ggml INTERFACE IMPORTED)
	set_target_properties(ggml::ggml PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "`${_ofxggmlcore_include_dir}"
		INTERFACE_COMPILE_DEFINITIONS "GGML_MAX_NAME=128"
		INTERFACE_LINK_LIBRARIES "`${_ofxggmlcore_libraries}"
	)
endif()
"@

	$versionContent = @"
set(PACKAGE_VERSION "ofxGgmlCore")
set(PACKAGE_VERSION_COMPATIBLE TRUE)
"@

	Set-Content -LiteralPath $configPath -Value $configContent -Encoding ASCII
	Set-Content -LiteralPath $versionPath -Value $versionContent -Encoding ASCII
	return $packageDir
}

function Convert-ToOnOff {
	param([bool]$Value)
	if ($Value) { return "ON" }
	return "OFF"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot
$runtimeRoot = Join-Path $addonRoot "libs\stable-diffusion"
if ([string]::IsNullOrWhiteSpace($SourceDir)) {
	$SourceDir = Join-Path $runtimeRoot ".source"
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
	$BuildDir = Join-Path $runtimeRoot "build"
}
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
	$InstallDir = $runtimeRoot
}
if ([string]::IsNullOrWhiteSpace($OfxGgmlCorePath)) {
	$OfxGgmlCorePath = [System.IO.Path]::Combine($addonRoot, "..", "ofxGgmlCore")
}
$OfxGgmlCorePath = [System.IO.Path]::GetFullPath($OfxGgmlCorePath)
if ($Jobs -le 0) {
	$Jobs = [Math]::Max(1, [Environment]::ProcessorCount)
}

$explicitBackend = $Cuda -or $Vulkan -or $Metal -or $OpenCL -or $CpuOnly
if (!$explicitBackend) {
	$Auto = $true
}

$enableCuda = $false
$enableVulkan = $false
$enableMetal = $false
$enableOpenCL = $false
$mode = "Auto"
$coreGgmlAvailable = Test-CoreGgmlBaseAvailable -CorePath $OfxGgmlCorePath
$useBundledGgml = [bool]$BundledGgml -and !$coreGgmlAvailable
$ggmlMode = if ($useBundledGgml) { "Bundled" } else { "ofxGgmlCore" }
$ggmlModeDetail = if ($BundledGgml -and $coreGgmlAvailable) {
	"BundledGgml requested but ignored because ofxGgmlCore ggml is available"
} elseif ($useBundledGgml) {
	"BundledGgml requested and ofxGgmlCore ggml was not found"
} else {
	"using ofxGgmlCore ggml"
}
$coreGgmlPackageRoot = Join-Path $BuildDir "ofxggmlcore-cmake"
$coreGgmlPackageDir = Join-Path $coreGgmlPackageRoot "ggml"

if ($CpuOnly) {
	$mode = "CpuOnly"
} else {
	if ($Auto) {
		$enableCuda = Test-CudaAvailable
		$enableVulkan = Test-VulkanAvailable
		$enableMetal = Test-MetalAvailable
		$enableOpenCL = Test-OpenCLAvailable
		if (!$useBundledGgml) {
			$enableCuda = $enableCuda -and (Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-cuda.lib" -UnixLibraryName "libggml-cuda.a")
			$enableVulkan = $enableVulkan -and (Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-vulkan.lib" -UnixLibraryName "libggml-vulkan.a")
			$enableMetal = $enableMetal -and (Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-metal.lib" -UnixLibraryName "libggml-metal.a")
			$enableOpenCL = $enableOpenCL -and (Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-opencl.lib" -UnixLibraryName "libggml-opencl.a")
		}
	} else {
		$enableCuda = [bool]$Cuda
		$enableVulkan = [bool]$Vulkan
		$enableMetal = [bool]$Metal
		$enableOpenCL = [bool]$OpenCL
		$mode = "Explicit"
	}
}

if ($enableCuda -and !(Test-CudaAvailable)) {
	$enableCuda = $false
}
if ($enableVulkan -and !(Test-VulkanAvailable)) {
	$enableVulkan = $false
}
if ($enableMetal -and !(Test-MetalAvailable)) {
	$enableMetal = $false
}
if ($enableOpenCL -and !(Test-OpenCLAvailable)) {
	$enableOpenCL = $false
}
if (!$useBundledGgml) {
	if ($enableCuda -and !(Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-cuda.lib" -UnixLibraryName "libggml-cuda.a")) {
		$enableCuda = $false
	}
	if ($enableVulkan -and !(Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-vulkan.lib" -UnixLibraryName "libggml-vulkan.a")) {
		$enableVulkan = $false
	}
	if ($enableMetal -and !(Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-metal.lib" -UnixLibraryName "libggml-metal.a")) {
		$enableMetal = $false
	}
	if ($enableOpenCL -and !(Test-CoreGgmlLibraryAvailable -CorePath $OfxGgmlCorePath -WindowsLibraryName "ggml-opencl.lib" -UnixLibraryName "libggml-opencl.a")) {
		$enableOpenCL = $false
	}
}
if ($Cuda -and !$enableCuda) {
	throw "CUDA was requested but CUDA Toolkit or ofxGgmlCore ggml-cuda runtime was not found. Build Core with CUDA first or use default -Auto to skip unavailable backends."
}
if ($Vulkan -and !$enableVulkan) {
	throw "Vulkan was requested but Vulkan SDK/tools or ofxGgmlCore ggml-vulkan runtime was not found. Build Core with Vulkan first or use default -Auto to skip unavailable backends."
}
if ($Metal -and !$enableMetal) {
	throw "Metal was requested but this host does not look like a macOS/Xcode environment or ofxGgmlCore ggml-metal runtime was not found."
}
if ($OpenCL -and !$enableOpenCL) {
	throw "OpenCL was requested but OpenCL tools/root or ofxGgmlCore ggml-opencl runtime was not found. Use default -Auto to skip unavailable backends."
}

$resolvedGenerator = Get-DefaultGenerator
$cmakeConfigure = @()
if (![string]::IsNullOrWhiteSpace($resolvedGenerator)) {
	$cmakeConfigure += @("-G", $resolvedGenerator)
	if (Test-WindowsHost -and $resolvedGenerator -like "Visual Studio*") {
		$cmakeConfigure += @("-A", "x64")
	}
}
$cmakeConfigure += @(
	"-S", $SourceDir,
	"-B", $BuildDir,
	"-DCMAKE_INSTALL_PREFIX=$InstallDir",
	"-DSD_BUILD_EXAMPLES=$(Convert-ToOnOff $BuildExamples)",
	"-DSD_WEBP=OFF",
	"-DSD_WEBM=OFF",
	"-DSD_BUILD_SHARED_LIBS=$(Convert-ToOnOff $Shared)",
	"-DSD_CUDA=$(Convert-ToOnOff $enableCuda)",
	"-DSD_VULKAN=$(Convert-ToOnOff $enableVulkan)",
	"-DSD_METAL=$(Convert-ToOnOff $enableMetal)",
	"-DSD_OPENCL=$(Convert-ToOnOff $enableOpenCL)"
)
if (!$useBundledGgml) {
	$cmakeConfigure += @(
		"-DSD_USE_SYSTEM_GGML=ON",
		"-DCMAKE_PREFIX_PATH=$coreGgmlPackageRoot",
		"-Dggml_DIR=$coreGgmlPackageDir"
	)
}

if ($DryRun) {
	Write-Step "Dry run: stable-diffusion.cpp setup plan"
	Write-Host "  repo: $Repo"
	Write-Host "  revision: $Revision"
	Write-Host "  root: $runtimeRoot"
	Write-Host "  source: $SourceDir"
	Write-Host "  build: $BuildDir"
	Write-Host "  install: $InstallDir"
	Write-Host "  mode: $mode"
	Write-Host "  enabled backends: CPU=ON CUDA=$(Convert-ToOnOff $enableCuda) Vulkan=$(Convert-ToOnOff $enableVulkan) Metal=$(Convert-ToOnOff $enableMetal) OpenCL=$(Convert-ToOnOff $enableOpenCL)"
	Write-Host "  ggml: $ggmlMode"
	Write-Host "  ggml detail: $ggmlModeDetail"
	if (!$useBundledGgml) {
		Write-Host "  ofxGgmlCore: $OfxGgmlCorePath"
		Write-Host "  generated ggml package: $coreGgmlPackageDir"
	}
	Write-Host "  examples: $(Convert-ToOnOff $BuildExamples)"
	Write-Host "  shared: $(Convert-ToOnOff $Shared)"
	Write-Host "  jobs: $Jobs"
	Write-Host "  clean: $(Convert-ToOnOff $Clean)"
	Write-Host "cmake $($cmakeConfigure -join ' ')"
	Write-Host "cmake --build $BuildDir --config $Configuration --target stable-diffusion --parallel $Jobs"
	Write-Host "cmake --install $BuildDir --config $Configuration"
	Write-Step "Dry run complete; no files were changed"
	return
}

foreach ($tool in @("git", "cmake")) {
	if (!(Get-CommandPathOrNull $tool)) {
		throw "$tool was not found on PATH."
	}
}

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
	Write-Step "Cleaning $BuildDir"
	Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

if (!(Test-Path -LiteralPath $SourceDir -PathType Container)) {
	Write-Step "Cloning stable-diffusion.cpp"
	Invoke-Checked "git clone stable-diffusion.cpp" "git" @(
		"clone", "--recursive", "--depth", "1", "--branch", $Revision, $Repo, $SourceDir)
} else {
	Write-Step "stable-diffusion.cpp source already exists; skipping clone"
}

if (!$useBundledGgml) {
	if (!(Test-Path -LiteralPath $OfxGgmlCorePath -PathType Container)) {
		throw "ofxGgmlCore was not found at: $OfxGgmlCorePath. Use -OfxGgmlCorePath or pass -BundledGgml."
	}
	Write-Step "Generating ggml CMake package from ofxGgmlCore"
	New-OfxGgmlCoreCmakePackage `
		-CorePath $OfxGgmlCorePath `
		-PackageRoot $coreGgmlPackageRoot `
		-EnableCuda:$enableCuda `
		-EnableVulkan:$enableVulkan | Out-Null
}

Write-Step "Configuring stable-diffusion.cpp"
Invoke-Checked "cmake configure stable-diffusion.cpp" "cmake" $cmakeConfigure

Write-Step "Building stable-diffusion.cpp"
Invoke-Checked "cmake build stable-diffusion.cpp" "cmake" @(
	"--build", $BuildDir,
	"--config", $Configuration,
	"--target", "stable-diffusion",
	"--parallel", [string]$Jobs)

Write-Step "Installing stable-diffusion.cpp runtime"
Invoke-Checked "cmake install stable-diffusion.cpp" "cmake" @(
	"--install", $BuildDir,
	"--config", $Configuration)

Write-Step "Done. stable-diffusion.cpp runtime installed under $InstallDir"

param()

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Assert-Path {
	param(
		[string]$Path,
		[string]$Label,
		[switch]$Directory
	)

	if ($Directory) {
		if (!(Test-Path -LiteralPath $Path -PathType Container)) {
			throw "$Label was not found: $Path"
		}
	} elseif (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
		throw "$Label was not found: $Path"
	}
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot
$addonsRoot = Split-Path -Parent $addonRoot

Write-Step "Checking addon skeleton"
Assert-Path (Join-Path $addonRoot "addon_config.mk") "addon config"
Assert-Path (Join-Path $addonRoot "README.md") "README"
Assert-Path (Join-Path $addonRoot "LICENSE") "license"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion.h") "public header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionTypes.h") "types header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionUtils.h") "utility header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionUtils.cpp") "utility source"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionImageUtils.h") "image utility header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionImageUtils.cpp") "image utility source"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionNativeBackend.h") "native backend header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionNativeBackend.cpp") "native backend source"

Write-Step "Checking dependency layout"
Assert-Path (Join-Path $addonsRoot "ofxGgmlCore") "sibling ofxGgmlCore addon" -Directory

Write-Step "Checking example layout"
$exampleRoot = Join-Path $addonRoot "ofxGgmlDiffusionPromptExample"
Assert-Path $exampleRoot "root-level smoke example" -Directory
Assert-Path (Join-Path $exampleRoot "addons.make") "smoke example addons.make"
Assert-Path (Join-Path $exampleRoot "src\main.cpp") "smoke example main.cpp"
Assert-Path (Join-Path $exampleRoot "src\ofApp.h") "smoke example ofApp.h"
Assert-Path (Join-Path $exampleRoot "src\ofApp.cpp") "smoke example ofApp.cpp"
Assert-Path (Join-Path $addonRoot "tests\CMakeLists.txt") "test CMakeLists"
Assert-Path (Join-Path $addonRoot "tests\test_main.cpp") "test source"
Assert-Path (Join-Path $scriptRoot "build-stable-diffusion.ps1") "stable-diffusion build script"
Assert-Path (Join-Path $scriptRoot "build-stable-diffusion.bat") "stable-diffusion Windows build wrapper"
Assert-Path (Join-Path $scriptRoot "build-stable-diffusion.sh") "stable-diffusion shell build wrapper"
Assert-Path (Join-Path $scriptRoot "setup-stable-diffusion.ps1") "stable-diffusion setup script"
Assert-Path (Join-Path $scriptRoot "setup-stable-diffusion.bat") "stable-diffusion Windows setup wrapper"
Assert-Path (Join-Path $scriptRoot "setup-stable-diffusion.sh") "stable-diffusion shell setup wrapper"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-setup-dry-run.ps1") "stable-diffusion dry-run test"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-setup-dry-run.bat") "stable-diffusion Windows dry-run test wrapper"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-setup-dry-run.sh") "stable-diffusion shell dry-run test wrapper"
Assert-Path (Join-Path $addonRoot "libs\stable-diffusion\bin\.gitkeep") "stable-diffusion bin placeholder"
Assert-Path (Join-Path $addonRoot "libs\stable-diffusion\include\.gitkeep") "stable-diffusion include placeholder"
Assert-Path (Join-Path $addonRoot "libs\stable-diffusion\lib\.gitkeep") "stable-diffusion lib placeholder"

$nestedExamples = Join-Path $addonRoot "examples"
if (Test-Path -LiteralPath $nestedExamples -PathType Container) {
	throw "Examples should live at the addon root, not under: $nestedExamples"
}

Write-Step "Checking generated artifact hygiene"
$forbidden = @(
	"build",
	".vs",
	"ofxGgmlDiffusionPromptExample\bin",
	"ofxGgmlDiffusionPromptExample\obj",
	"ofxGgmlDiffusionPromptExample\.vs",
	"libs\stable-diffusion\.source",
	"libs\stable-diffusion\build",
	"models"
)

foreach ($relative in $forbidden) {
	$path = Join-Path $addonRoot $relative
	if (Test-Path -LiteralPath $path) {
		throw "Generated or local-only path should not be committed here: $relative"
	}
}

Write-Step "Checking stable-diffusion.cpp setup dry-runs"
& (Join-Path $scriptRoot "test-stable-diffusion-setup-dry-run.ps1")

Write-Step "Running headless tests"
& (Join-Path $scriptRoot "test-addon.ps1")

Write-Step "ofxGgmlDiffusion local validation passed"

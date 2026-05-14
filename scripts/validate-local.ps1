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

function Assert-FileContains {
	param(
		[string]$Path,
		[string]$Pattern,
		[string]$Label
	)

	$content = Get-Content -LiteralPath $Path -Raw
	if ($content -notmatch $Pattern) {
		throw "$Label did not contain expected pattern: $Pattern"
	}
}
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot
$addonsRoot = Split-Path -Parent $addonRoot

Write-Step "Checking addon skeleton"
Assert-Path (Join-Path $addonRoot "addon_config.mk") "addon config"
Assert-Path (Join-Path $addonRoot "README.md") "README"
Assert-Path (Join-Path $addonRoot "LICENSE") "license"
Assert-Path (Join-Path $addonRoot "docs\DIFFUSION_WORKFLOWS.md") "diffusion workflow docs"
Assert-FileContains (Join-Path $addonRoot "README.md") "docs/DIFFUSION_WORKFLOWS.md" "README"
Assert-FileContains (Join-Path $addonRoot "docs\DIFFUSION_WORKFLOWS.md") "Planning handoff" "diffusion workflow docs"
Assert-FileContains (Join-Path $addonRoot "docs\DIFFUSION_WORKFLOWS.md") "Validation ladder" "diffusion workflow docs"
Assert-FileContains (Join-Path $addonRoot "docs\DIFFUSION_WORKFLOWS.md") "generated artifacts" "diffusion workflow docs"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion.h") "public header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusionVersion.h") "version header"
Assert-FileContains (Join-Path $addonRoot "src\ofxGgmlDiffusion.h") "ofxGgmlDiffusionVersion.h" "public header"
Assert-FileContains (Join-Path $addonRoot "src\ofxGgmlDiffusionVersion.h") "OFXGGML_DIFFUSION_VERSION_STRING" "version header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionTypes.h") "types header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionUtils.h") "utility header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionUtils.cpp") "utility source"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionImageUtils.h") "image utility header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionImageUtils.cpp") "image utility source"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionImageGenerationBackend.h") "image generation backend header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionImageGenerationBackend.cpp") "image generation backend source"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionTinyGanBackend.h") "tiny GAN backend header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionTinyGanBackend.cpp") "tiny GAN backend source"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionTinyGanTraining.h") "tiny GAN training header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionTinyGanTraining.cpp") "tiny GAN training source"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionNativeBackend.h") "native backend header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionNativeBackend.cpp") "native backend source"
Assert-FileContains (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionImageUtils.h") "loadImage" "image utility load API"
Assert-FileContains (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionNativeBackend.h") "ofxGgmlDiffusionImageGenerationBackend" "native backend interface"
Assert-FileContains (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionNativeBackend.h") "ofxGgmlMakeNativeDiffusionImageGenerationBackend" "native backend factory"
Assert-FileContains (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionNativeBackend.h") "ofxGgmlDiffusionNativeCapabilities" "native capability descriptor"
Assert-FileContains (Join-Path $addonRoot "tests\test_native_smoke.cpp") "supportsPhotoMaker" "native PhotoMaker capability smoke"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionAsyncRunner.h") "async runner header"
Assert-Path (Join-Path $addonRoot "src\ofxGgmlDiffusion\ofxGgmlDiffusionAsyncRunner.cpp") "async runner source"

Write-Step "Checking dependency layout"
Assert-Path (Join-Path $addonsRoot "ofxGgmlCore") "sibling ofxGgmlCore addon" -Directory
Assert-Path (Join-Path $addonsRoot "ofxImGui") "sibling ofxImGui addon for examples" -Directory

Write-Step "Checking example layout"
$exampleRoot = Join-Path $addonRoot "ofxGgmlDiffusionPromptExample"
Assert-Path $exampleRoot "root-level smoke example" -Directory
Assert-Path (Join-Path $exampleRoot "addons.make") "smoke example addons.make"
Assert-FileContains (Join-Path $exampleRoot "addons.make") "(?m)^ofxImGui\s*$" "smoke example addons.make"
Assert-Path (Join-Path $exampleRoot "src\main.cpp") "smoke example main.cpp"
Assert-Path (Join-Path $exampleRoot "src\ofApp.h") "smoke example ofApp.h"
Assert-Path (Join-Path $exampleRoot "src\ofApp.cpp") "smoke example ofApp.cpp"
$ganExampleRoot = Join-Path $addonRoot "ofxGgmlDiffusionGanExample"
Assert-Path $ganExampleRoot "root-level GAN example" -Directory
Assert-Path (Join-Path $ganExampleRoot "addons.make") "GAN example addons.make"
Assert-FileContains (Join-Path $ganExampleRoot "addons.make") "(?m)^ofxImGui\s*$" "GAN example addons.make"
Assert-Path (Join-Path $ganExampleRoot "src\main.cpp") "GAN example main.cpp"
Assert-Path (Join-Path $ganExampleRoot "src\ofApp.h") "GAN example ofApp.h"
Assert-Path (Join-Path $ganExampleRoot "src\ofApp.cpp") "GAN example ofApp.cpp"
Assert-Path (Join-Path $addonRoot "tests\CMakeLists.txt") "test CMakeLists"
Assert-Path (Join-Path $addonRoot "tests\test_main.cpp") "test source"
Assert-Path (Join-Path $addonRoot "tests\test_native_smoke.cpp") "native bridge smoke test source"
Assert-Path (Join-Path $scriptRoot "build-stable-diffusion.ps1") "stable-diffusion build script"
Assert-Path (Join-Path $scriptRoot "build-stable-diffusion.bat") "stable-diffusion Windows build wrapper"
Assert-Path (Join-Path $scriptRoot "build-stable-diffusion.sh") "stable-diffusion shell build wrapper"
Assert-Path (Join-Path $scriptRoot "doctor-diffusion.ps1") "diffusion doctor script"
Assert-Path (Join-Path $scriptRoot "doctor-diffusion.bat") "diffusion doctor Windows wrapper"
Assert-Path (Join-Path $scriptRoot "doctor-diffusion.sh") "diffusion doctor shell wrapper"
Assert-Path (Join-Path $scriptRoot "test-doctor-diffusion.ps1") "diffusion doctor smoke test"
Assert-Path (Join-Path $scriptRoot "setup-stable-diffusion.ps1") "stable-diffusion setup script"
Assert-Path (Join-Path $scriptRoot "setup-stable-diffusion.bat") "stable-diffusion Windows setup wrapper"
Assert-Path (Join-Path $scriptRoot "setup-stable-diffusion.sh") "stable-diffusion shell setup wrapper"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-setup-dry-run.ps1") "stable-diffusion dry-run test"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-setup-dry-run.bat") "stable-diffusion Windows dry-run test wrapper"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-setup-dry-run.sh") "stable-diffusion shell dry-run test wrapper"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-native.ps1") "stable-diffusion native smoke script"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-native.bat") "stable-diffusion native smoke Windows wrapper"
Assert-Path (Join-Path $scriptRoot "test-stable-diffusion-native.sh") "stable-diffusion native smoke shell wrapper"
Assert-Path (Join-Path $scriptRoot "build-diffusion-example.ps1") "diffusion example build script"
Assert-FileContains (Join-Path $scriptRoot "build-diffusion-example.ps1") "Repair-DiffusionGeneratedProject" "diffusion generated project repair"
Assert-FileContains (Join-Path $scriptRoot "build-diffusion-example.ps1") "Initialize-DiffusionGeneratedProject" "diffusion generated project initializer"
Assert-Path (Join-Path $scriptRoot "build-diffusion-example.bat") "diffusion example Windows build wrapper"
Assert-Path (Join-Path $scriptRoot "build-diffusion-example.sh") "diffusion example shell build wrapper"
Assert-Path (Join-Path $scriptRoot "run-diffusion-example.ps1") "diffusion example run script"
Assert-Path (Join-Path $scriptRoot "run-diffusion-example.bat") "diffusion example Windows run wrapper"
Assert-Path (Join-Path $scriptRoot "run-diffusion-example.sh") "diffusion example shell run wrapper"
Assert-Path (Join-Path $scriptRoot "run-gan-example.ps1") "GAN example run script"
Assert-Path (Join-Path $scriptRoot "run-gan-example.bat") "GAN example Windows run wrapper"
Assert-Path (Join-Path $scriptRoot "run-gan-example.sh") "GAN example shell run wrapper"
Assert-Path (Join-Path $scriptRoot "create-tiny-gan-preset.ps1") "tiny GAN preset script"
Assert-Path (Join-Path $scriptRoot "create-tiny-gan-preset.bat") "tiny GAN preset Windows wrapper"
Assert-Path (Join-Path $scriptRoot "create-tiny-gan-preset.sh") "tiny GAN preset shell wrapper"
Assert-Path (Join-Path $scriptRoot "create-tiny-gan-fixtures.ps1") "tiny GAN fixture script"
Assert-Path (Join-Path $scriptRoot "create-tiny-gan-fixtures.bat") "tiny GAN fixture Windows wrapper"
Assert-Path (Join-Path $scriptRoot "create-tiny-gan-fixtures.sh") "tiny GAN fixture shell wrapper"
Assert-Path (Join-Path $scriptRoot "train-tiny-gan.ps1") "tiny GAN training script"
Assert-Path (Join-Path $scriptRoot "train-tiny-gan.bat") "tiny GAN training Windows wrapper"
Assert-Path (Join-Path $scriptRoot "train-tiny-gan.sh") "tiny GAN training shell wrapper"
Assert-Path (Join-Path $scriptRoot "test-launch-dry-run.ps1") "diffusion launch dry-run test"
Assert-Path (Join-Path $scriptRoot "test-launch-dry-run.bat") "diffusion launch dry-run Windows wrapper"
Assert-Path (Join-Path $scriptRoot "test-launch-dry-run.sh") "diffusion launch dry-run shell wrapper"
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

Write-Step "Checking stable-diffusion.cpp native smoke dry-run"
$nativeSmokeDryRun = & (Join-Path $scriptRoot "test-stable-diffusion-native.ps1") -DryRun 2>&1 6>&1 | Out-String
if (!$nativeSmokeDryRun.Contains("stable-diffusion.cpp native smoke plan") -or
	!$nativeSmokeDryRun.Contains("Dry run complete; no files were changed")) {
	throw "stable-diffusion.cpp native smoke dry-run output was unexpected:`n$nativeSmokeDryRun"
}

Write-Step "Checking diffusion doctor smoke"
& (Join-Path $scriptRoot "test-doctor-diffusion.ps1")

Write-Step "Checking launch dry-runs"
& (Join-Path $scriptRoot "test-launch-dry-run.ps1")

Write-Step "Running headless tests"
& (Join-Path $scriptRoot "test-addon.ps1")

Write-Step "ofxGgmlDiffusion local validation passed"

param(
	[string]$Model = $env:OFXGGML_DIFFUSION_MODEL,
	[string]$PhotoMakerModel = $env:OFXGGML_PHOTOMAKER_MODEL,
	[string]$PhotoMakerRefs = $env:OFXGGML_PHOTOMAKER_REFS
)

$ErrorActionPreference = "Stop"

function Write-Status {
	param([string]$State, [string]$Name, [string]$Detail = "")
	$line = "{0,-5} {1}" -f $State, $Name
	if (![string]::IsNullOrWhiteSpace($Detail)) {
		$line += " - $Detail"
	}
	Write-Host $line
}

function Resolve-OptionalPathText {
	param([string]$Path)
	if ([string]::IsNullOrWhiteSpace($Path)) {
		return ""
	}
	try {
		return [System.IO.Path]::GetFullPath($Path)
	} catch {
		return $Path
	}
}

function Test-FileStatus {
	param([string]$Name, [string]$Path, [string]$MissingDetail = "")
	if (![string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path -PathType Leaf)) {
		Write-Status "OK" $Name $Path
		return $true
	}
	Write-Status "WARN" $Name $MissingDetail
	return $false
}

function Test-DirectoryStatus {
	param([string]$Name, [string]$Path, [string]$MissingDetail = "")
	if (![string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path -PathType Container)) {
		Write-Status "OK" $Name $Path
		return $true
	}
	Write-Status "WARN" $Name $MissingDetail
	return $false
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Resolve-Path (Join-Path $scriptRoot "..")
$addonsRoot = Resolve-Path (Join-Path $addonRoot "..")
$ofRoot = Resolve-Path (Join-Path $addonsRoot "..")
$exampleRoot = Join-Path $addonRoot "ofxGgmlDiffusionPromptExample"

Write-Host "ofxGgmlDiffusion doctor"
Write-Host "Root  $addonRoot"
Write-Host ""

[void](Test-DirectoryStatus "ofxGgmlCore dependency" (Join-Path $addonsRoot "ofxGgmlCore") "clone sibling ofxGgmlCore")
[void](Test-DirectoryStatus "ofxImGui dependency" (Join-Path $addonsRoot "ofxImGui") "clone sibling ofxImGui for examples")
[void](Test-FileStatus "stable-diffusion.cpp header" (Join-Path $addonRoot "libs\stable-diffusion\include\stable-diffusion.h") "run scripts\build-stable-diffusion.bat")
[void](Test-FileStatus "stable-diffusion.cpp library" (Join-Path $addonRoot "libs\stable-diffusion\lib\stable-diffusion.lib") "run scripts\build-stable-diffusion.bat")
[void](Test-DirectoryStatus "stable-diffusion.cpp runtime bin" (Join-Path $addonRoot "libs\stable-diffusion\bin") "run scripts\build-stable-diffusion.bat")

$projectGenerator = Join-Path $ofRoot "projectGenerator\resources\app\app\projectGenerator.exe"
if (!(Test-Path -LiteralPath $projectGenerator -PathType Leaf)) {
	$projectGenerator = Join-Path $ofRoot "projectGenerator\projectGenerator.exe"
}
[void](Test-FileStatus "projectGenerator" $projectGenerator "install/open the openFrameworks projectGenerator")
[void](Test-FileStatus "prompt example vcxproj" (Join-Path $exampleRoot "ofxGgmlDiffusionPromptExample.vcxproj") "generate the prompt example project before building")

Write-Host ""
Write-Host "Models"
$Model = Resolve-OptionalPathText $Model
if (![string]::IsNullOrWhiteSpace($Model)) {
	[void](Test-FileStatus "OFXGGML_DIFFUSION_MODEL" $Model "set to an existing SD/SDXL/Flux model")
} else {
	Write-Status "WARN" "OFXGGML_DIFFUSION_MODEL" "not set"
}

$defaultModel = Join-Path $exampleRoot "bin\data\models\model.safetensors"
[void](Test-FileStatus "default example model" $defaultModel "optional fallback path")

$modelRoots = @(
	(Join-Path $addonRoot "models"),
	(Join-Path $exampleRoot "bin\data\models"),
	(Join-Path $addonsRoot "models")
)
$candidateModels = @()
foreach ($root in $modelRoots) {
	if (Test-Path -LiteralPath $root -PathType Container) {
		$candidateModels += Get-ChildItem -LiteralPath $root -File -Include *.safetensors,*.ckpt,*.gguf -Recurse -ErrorAction SilentlyContinue |
			Where-Object { $_.Name -match "sd|stable|xl|flux|diffusion|wan|qwen-image" } |
			Select-Object -First 5
	}
}
if ($candidateModels.Count -gt 0) {
	foreach ($candidate in $candidateModels | Select-Object -First 8) {
		Write-Status "NOTE" "candidate diffusion model" $candidate.FullName
	}
} else {
	Write-Status "WARN" "candidate diffusion model" "none found in addon/example/shared model folders"
}

Write-Host ""
Write-Host "PhotoMaker"
$PhotoMakerModel = Resolve-OptionalPathText $PhotoMakerModel
if (![string]::IsNullOrWhiteSpace($PhotoMakerModel)) {
	[void](Test-FileStatus "OFXGGML_PHOTOMAKER_MODEL" $PhotoMakerModel "set to an existing PhotoMaker model")
} else {
	Write-Status "WARN" "OFXGGML_PHOTOMAKER_MODEL" "not set"
}

if (![string]::IsNullOrWhiteSpace($PhotoMakerRefs)) {
	$refs = $PhotoMakerRefs -split ";" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
	$existingRefs = 0
	foreach ($ref in $refs) {
		$refPath = Resolve-OptionalPathText $ref
		if (Test-Path -LiteralPath $refPath -PathType Leaf) {
			++$existingRefs
			Write-Status "OK" "PhotoMaker reference" $refPath
		} else {
			Write-Status "WARN" "PhotoMaker reference" "$refPath does not exist"
		}
	}
	if ($existingRefs -eq 0) {
		Write-Status "WARN" "PhotoMaker references" "no existing reference image paths"
	}
} else {
	Write-Status "WARN" "OFXGGML_PHOTOMAKER_REFS" "not set; use semicolon-separated image paths"
}

Write-Host ""
Write-Host "Suggested next checks"
Write-Host "  scripts\test-stable-diffusion-native.bat"
Write-Host "  scripts\run-diffusion-example.bat -Build"

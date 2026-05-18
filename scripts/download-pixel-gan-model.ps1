param(
	[string]$OutputPath = "",
	[switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
	$OutputPath = Join-Path $addonRoot "ofxGgmlDiffusionGanExample\bin\data\models\pixel-model-f16.gguf"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDir = Split-Path -Parent $OutputPath
$url = "https://huggingface.co/gguf-org/pixel/resolve/main/model-f16.gguf?download=true"

if ((Test-Path -LiteralPath $OutputPath -PathType Leaf) -and !$Force) {
	Write-Step "Pixel/DCGAN GGUF already exists: $OutputPath"
	Write-Step "Use -Force to download it again."
	return
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Write-Step "Downloading gguf-org/pixel Pixel/DCGAN GGUF"
Write-Step "Destination: $OutputPath"
Invoke-WebRequest -Uri $url -OutFile $OutputPath
Write-Step "Done"
Write-Step "Use with: scripts\run-gan-example.bat -Build -Generator `"$OutputPath`""

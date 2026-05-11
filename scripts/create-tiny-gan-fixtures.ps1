param(
	[string]$OutputPath = "",
	[int]$Count = 8,
	[switch]$Force,
	[switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Get-FixturePixel {
	param(
		[int]$X,
		[int]$Y,
		[int]$Index,
		[int]$Channel
	)
	$stripe = (([Math]::Floor($X / 8) + $Index + $Channel) % 2)
	$checker = (([Math]::Floor($X / 8) + [Math]::Floor($Y / 8) + $Index) % 2)
	$gradient = (($X * 3 + $Y * 5 + $Index * 29 + $Channel * 47) % 256)
	if (($Index % 3) -eq 0) {
		if ($stripe) { return 224 }
		return 32
	}
	if (($Index % 3) -eq 1) {
		if ($checker) { return 196 }
		return 56
	}
	return $gradient
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Split-Path -Parent $scriptRoot

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
	$OutputPath = Join-Path $addonRoot "ofxGgmlDiffusionGanExample\bin\data\datasets\tiny-fixtures"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)

if ($Count -lt 1 -or $Count -gt 1024) {
	throw "Count must be between 1 and 1024."
}

if ($DryRun) {
	Write-Step "Tiny GAN fixture dataset output: $OutputPath"
	Write-Step "Image count: $Count"
	Write-Step "Image format: ASCII PPM, 64x64 RGB"
	Write-Step "Dry run complete; no files were written"
	return
}

if ((Test-Path -LiteralPath $OutputPath) -and !$Force) {
	throw "Fixture dataset already exists: $OutputPath. Pass -Force to overwrite."
}
if (Test-Path -LiteralPath $OutputPath) {
	Remove-Item -LiteralPath $OutputPath -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

for ($i = 0; $i -lt $Count; $i += 1) {
	$name = "fixture-{0:D3}.ppm" -f $i
	$path = Join-Path $OutputPath $name
	$lines = New-Object System.Collections.Generic.List[string]
	$lines.Add("P3")
	$lines.Add("64 64")
	$lines.Add("255")
	for ($y = 0; $y -lt 64; $y += 1) {
		$row = New-Object System.Text.StringBuilder
		for ($x = 0; $x -lt 64; $x += 1) {
			[void]$row.Append((Get-FixturePixel -X $x -Y $y -Index $i -Channel 0))
			[void]$row.Append(" ")
			[void]$row.Append((Get-FixturePixel -X $x -Y $y -Index $i -Channel 1))
			[void]$row.Append(" ")
			[void]$row.Append((Get-FixturePixel -X $x -Y $y -Index $i -Channel 2))
			if ($x -lt 63) {
				[void]$row.Append(" ")
			}
		}
		$lines.Add($row.ToString())
	}
	Set-Content -LiteralPath $path -Value $lines -Encoding ASCII
}

Write-Step "Wrote tiny GAN fixture dataset: $OutputPath"
Write-Step "Image count: $Count"
Write-Step "Image format: ASCII PPM, 64x64 RGB"

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$doctorScript = Join-Path $scriptRoot "doctor-diffusion.ps1"

$output = & $doctorScript *>&1 | ForEach-Object { $_.ToString() }
if (!$?) {
	throw "doctor-diffusion.ps1 failed."
}

$text = $output -join "`n"
foreach ($expected in @(
	"ofxGgmlDiffusion doctor",
	"ofxGgmlCore dependency",
	"Models",
	"PhotoMaker",
	"Suggested next checks"
)) {
	if ($text -notmatch [regex]::Escape($expected)) {
		throw "doctor output did not contain expected text: $expected"
	}
}

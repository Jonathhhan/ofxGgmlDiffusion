param(
	[Parameter(ValueFromRemainingArguments = $true)]
	[string[]]$RemainingArguments
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptRoot "build-stable-diffusion.ps1") @RemainingArguments
exit $LASTEXITCODE

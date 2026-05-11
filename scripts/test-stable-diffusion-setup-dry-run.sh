#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run_ps1="$script_dir/test-stable-diffusion-setup-dry-run.ps1"

if command -v pwsh >/dev/null 2>&1; then
	exec pwsh -NoProfile -ExecutionPolicy Bypass -File "$run_ps1" "$@"
fi

if command -v powershell >/dev/null 2>&1; then
	exec powershell -NoProfile -ExecutionPolicy Bypass -File "$run_ps1" "$@"
fi

echo "PowerShell 7+ is required to run test-stable-diffusion-setup-dry-run.sh" >&2
exit 1

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_PATH="${1:-$ADDON_ROOT/ofxGgmlDiffusionGanExample/bin/data/models/pixel-model-f16.gguf}"
URL="https://huggingface.co/gguf-org/pixel/resolve/main/model-f16.gguf?download=true"

mkdir -p "$(dirname "$OUTPUT_PATH")"
if [[ -f "$OUTPUT_PATH" && "${FORCE:-0}" != "1" ]]; then
	echo "==> Pixel/DCGAN GGUF already exists: $OUTPUT_PATH"
	echo "==> Set FORCE=1 to download it again."
	exit 0
fi

echo "==> Downloading gguf-org/pixel Pixel/DCGAN GGUF"
echo "==> Destination: $OUTPUT_PATH"
if command -v curl >/dev/null 2>&1; then
	curl -L "$URL" -o "$OUTPUT_PATH"
else
	wget -O "$OUTPUT_PATH" "$URL"
fi
echo "==> Done"

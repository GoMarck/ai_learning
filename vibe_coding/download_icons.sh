#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
IMG_DIR="$BASE_DIR/images"
mkdir -p "$IMG_DIR"

curl -L --fail https://github.com/favicon.ico -o "$IMG_DIR/github-copilot.ico"
curl -L --fail https://openai.com/favicon.ico -o "$IMG_DIR/openai-codex.ico"
curl -L --fail https://cursor.com/favicon.ico -o "$IMG_DIR/cursor.ico"
curl -L --fail https://www.anthropic.com/favicon.ico -o "$IMG_DIR/claude-code.ico"
curl -L --fail https://www.codegeex.cn/favicon.ico -o "$IMG_DIR/codegeex.ico"
curl -L --fail https://www.marscode.com/favicon.ico -o "$IMG_DIR/marscode.ico"

echo "Downloaded icons into $IMG_DIR"

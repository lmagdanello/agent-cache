#!/usr/bin/env bash
set -euo pipefail

src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest_root="${CODEX_HOME:-$HOME/.codex}/skills"
dest_dir="$dest_root/agent-cache"

mkdir -p "$dest_root"
rm -rf "$dest_dir"
cp -R "$src_dir" "$dest_dir"
printf 'Installed agent-cache skill to %s\n' "$dest_dir"

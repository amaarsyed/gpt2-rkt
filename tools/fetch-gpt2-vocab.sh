#!/bin/sh
# Download the pinned GPT-2 Small files: tokenizer, config, and weights.
# Pinned to openai-community/gpt2 at 607a30d783dfa663caf39e06633721c8d4cfcd7e.
# Digests live in tools/gpt2-assets.sha256. The weights stay out of git.
# GPT2_FETCH_WEIGHTS=0 skips model.safetensors. The default downloads it.

set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
dest="$root/models/gpt2"
assets="$root/tools/gpt2-assets.sha256"
mkdir -p "$dest"

base=https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "sha256sum or shasum is required" >&2
    exit 1
  fi
}

expected_sha256() {
  name=$1
  hash=$(awk -v name="$name" 'NF >= 2 && $1 !~ /^#/ && $2 == name { print $1; exit }' "$assets")
  if [ -z "$hash" ]; then
    echo "no digest for $name in $assets" >&2
    exit 1
  fi
  printf '%s\n' "$hash"
}

fetch() {
  name=$1
  hash=$(expected_sha256 "$name")
  path="$dest/$name"
  if [ -f "$path" ]; then
    got=$(file_sha256 "$path")
    if [ "$got" = "$hash" ]; then
      return 0
    fi
    echo "sha256 mismatch for $name; downloading again" >&2
  fi
  tmp=$(mktemp)
  trap 'rm -f "$tmp"' EXIT
  curl -fsSL -A "gpt2-rkt" -o "$tmp" "$base/$name"
  got=$(file_sha256 "$tmp")
  if [ "$got" != "$hash" ]; then
    echo "sha256 mismatch for $name" >&2
    echo "expected $hash" >&2
    echo "got      $got" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$path"
}

for name in $(awk 'NF >= 2 && $1 !~ /^#/ { print $2 }' "$assets"); do
  case "$name" in
    *.safetensors)
      if [ "${GPT2_FETCH_WEIGHTS:-1}" = "0" ]; then
        echo "skipping $name (GPT2_FETCH_WEIGHTS=0)" >&2
        continue
      fi
      ;;
  esac
  fetch "$name"
done

echo "cached pinned GPT-2 files in $dest"

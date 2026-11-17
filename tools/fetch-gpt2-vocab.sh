#!/bin/sh
# Download the GPT-2 BPE files used by the tests.
# Pinned to openai-community/gpt2 at 607a30d783dfa663caf39e06633721c8d4cfcd7e.
# Weights are not downloaded.

set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
dest="$root/models/gpt2"
mkdir -p "$dest"

vocab_hash=196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
merges_hash=1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
base=https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e

fetch() {
  name=$1
  hash=$2
  path="$dest/$name"
  if [ -f "$path" ]; then
    got=$(sha256sum "$path" | awk '{print $1}')
    if [ "$got" = "$hash" ]; then
      return 0
    fi
    echo "sha256 mismatch for $name; downloading again" >&2
  fi
  tmp=$(mktemp)
  curl -fsSL -A "gpt2-rkt" -o "$tmp" "$base/$name"
  got=$(sha256sum "$tmp" | awk '{print $1}')
  if [ "$got" != "$hash" ]; then
    echo "sha256 mismatch for $name" >&2
    echo "expected $hash" >&2
    echo "got      $got" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$path"
}

fetch vocab.json "$vocab_hash"
fetch merges.txt "$merges_hash"
echo "cached vocab.json and merges.txt in $dest"

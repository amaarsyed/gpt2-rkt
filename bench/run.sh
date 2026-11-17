#!/bin/sh
# Verify the pinned GPT-2 files, then measure load, prefill, decode, and
# matmul. Extra arguments are passed to bench/run.rkt, so
# `sh bench/run.sh --runs 5` changes the run count.
# This does not create a release tag.

set -eu
cd "$(dirname "$0")/.."
sh tools/fetch-gpt2-vocab.sh
exec racket bench/run.rkt --model-dir models/gpt2 "$@"

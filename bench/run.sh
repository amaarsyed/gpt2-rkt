#!/bin/sh
# Verify the pinned GPT-2 files, then measure load, prefill, decode, and
# matmul. Extra arguments are passed to bench/run.rkt, so
# `sh bench/run.sh --runs 5` changes the run count.
#
# On the M5 Pro, install the full Racket CS 9.3 and put `racket` and
# `raco` on PATH. A minimal Racket does not provide `raco test`. Expect
# about 5 minutes or more; prefill at 1000 tokens is the longest step.
# Keep the machine plugged in and idle. Commit the files written under
# bench/results/. Do not pass --not-m5-pro on that machine.

set -eu
cd "$(dirname "$0")/.."
sh tools/fetch-gpt2-vocab.sh
exec racket bench/run.rkt --model-dir models/gpt2 "$@"

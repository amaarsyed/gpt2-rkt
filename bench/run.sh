#!/bin/sh
# Check the pinned GPT-2 files, then measure load, prefill, decode, and
# matmul on whatever machine you run this on. Extra arguments are passed
# to bench/run.rkt, so `sh bench/run.sh --runs 5` changes the run count.
#
# Install the full Racket CS 9.3 distribution and put `racket` on PATH.
# A minimal Racket does not provide `raco test`. The script is POSIX sh:
# it runs on Linux and macOS, and on Windows from Git Bash, WSL, or
# another Unix shell. Expect about five minutes or more. Prefill at
# 1000 tokens is the longest step. Leave the machine idle, and on power
# if it has a battery. Results are written under bench/results/.

set -eu
cd "$(dirname "$0")/.."
sh tools/fetch-gpt2-vocab.sh
exec racket bench/run.rkt --model-dir models/gpt2 "$@"

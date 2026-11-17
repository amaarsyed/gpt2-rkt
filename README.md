# gpt2-rkt

Pure-Racket GPT-2 Small (124M) inference. Uses flvectors and Racket futures;
requires Racket CS 9.3. No Python or native math library at runtime.

## Run

Install the full Racket distribution (`racket` and `raco` on PATH), then:

```sh
sh tools/fetch-gpt2-vocab.sh
raco test .
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16
```

The fetch script downloads and checks the pinned assets. Weights stay out of
git. Set `GPT2_FETCH_WEIGHTS=0` to fetch only the tokenizer and config.

Generation streams text to stdout and load/memory/throughput stats to stderr.
Omit sampling flags for greedy decoding. For sampling, add:

```sh
--temperature 0.8 --top-k 40 --seed 1
```

Temperature must be positive; top-k and seed require sampling. The seed range
is 0–2147483647. Prompt plus output must fit `min(1024, n_positions)`.
Generation stops at EOS or the requested count. Use `--help` for CLI options.

From Racket:

```racket
#lang racket/base
(require "main.rkt")

(define result
  (generate (generate-config "models/gpt2" "Once upon a time" 16 #f #f #f)))
(displayln (generation-text result))
```

## Benchmarks

```sh
sh bench/run.sh
```

This checks assets and records load, prefill, generation, matrix timings, and
memory under `bench/results/`. Each workload gets a fresh process, one warmup,
and three timed calls. Allow five minutes or more; keep the machine idle and
plugged in. See [all recorded benchmarks](docs/BENCHMARKS.md) for results and
the phase reports for comparison commands.

## Reference

- [Inference and formats](docs/INFERENCE.md): kernels, caching, tokenizer, assets.
- [Phase 6/7](docs/PHASE6-vs-PHASE7.md), [phase 8](docs/PHASE8.md), [phase 9](docs/PHASE9.md): changes and measurements.
- [Phase 8 experiments](docs/PHASE8-EXPERIMENTS.md), [phase 9 experiments](docs/PHASE9-EXPERIMENTS.md), [bug log](docs/BUGLOG.md): trials, rejected ideas, and fixes.
- [Changelog](CHANGELOG.md), [0.1.0 release](docs/RELEASE-v0.1.0.md).

`main.rkt` exports the API and CLI. Implementation lives in `src/`, RackUnit
tests in `tests/`, benchmarks in `bench/`. Fixture-generation Python lives in
`tools/fixtures/`; tests and CI use the committed fixtures. CI runs unit tests
and a separate real-weight eager comparison.

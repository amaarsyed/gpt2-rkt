# gpt2-rkt

Pure-Racket inference for GPT-2 Small (124M). Numeric work uses `flvector` and `racket/flonum`, with futures for parallelism later. Model weights are not in git.

This is the 0.1.0 line. `generate` parses a request and checks the model directory. Weight loading is not implemented yet, so a missing directory is a missing-assets error, and a directory that already has the expected files stops with "model loading is not implemented".

Tested with Racket CS 9.3. CI installs that version and runs `raco test`. Python is not required.

## Commands

Run the tests:

```sh
raco test .
```

Show CLI help:

```sh
racket main.rkt --help
racket main.rkt generate --help
```

Generate. This exits non-zero until model assets are installed:

```sh
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16
```

Decoding is greedy unless `--temperature` is greater than 0. `--top-k` and `--seed` require sampling:

```sh
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 1
```

Benchmark harness. This runs a placeholder `flvector` dot product, prints the Racket version and VM, and reports load time, first-token latency, steady-state tokens per second, and peak memory for a short prompt length (16) and a longer one (128):

```sh
racket bench/harness.rkt
```

## Layout

- `main.rkt` — CLI entry and public API
- `src/` — implementation modules
- `tests/` — RackUnit tests
- `bench/` — benchmark harness
- `tools/fixtures/` — Python fixture generator only, not used by tests or CI
- `docs/BUGLOG.md` — bug log

## Model directory

`generate` expects these files in `--model-dir`:

- `config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors`

A download script with a sha256 check will land in a later milestone. Do not commit weights.

# gpt2-rkt

Pure-Racket inference for GPT-2 Small (124M). Numeric work uses `flvector` and `racket/flonum`. On more than one processor, a matrix multiply of 32 or more rows runs one future per processor, and each future writes its own rows. Model weights are not in git.

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

Decoding is greedy unless `--temperature` is greater than 0. `--top-k` and `--seed` require sampling. A flag's value is the next argument, even when that text starts with `-`. `-h` and `--help` are help only in flag position. Range checks live on `generate-config`; the parser only decodes arguments.

```sh
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 1
```

Benchmark harness. This prints the Racket version and VM, then two reports. The prompt report is still a placeholder `flvector` dot product: load time, first-token latency, steady-state tokens per second, and `(current-memory-use 'peak)` for a short prompt length (16) and a longer one (128). The matmul report times one 768×768 Conv1D multiply and prints milliseconds, GFLOP/s, bytes allocated, and `vector-set-performance-stats!` counters:

```sh
racket bench/harness.rkt
```

## Numeric kernels

`src/tensor.rkt` is a row-major tensor on one `flvector`, with explicit dimensions. `src/nn.rkt` is the GPT-2 arithmetic:

- `matmul` is Hugging Face Conv1D: `y[n, out] = x[n, in] · W[in, out] + b[out]`. The inner loop is i-k-j so each step reads a contiguous weight row.
- `matmul-transposed-b` is `h[n, in] · Wᵀ` with `W` stored as `[out, in]`, for tied-embedding logits. It does not allocate a transpose.
- `tensor-add` is the residual add.
- `softmax` is row-wise on the last axis. It subtracts the row max, and an all-masked row (every value `-inf.0`) is zeros.
- `layer-norm` uses biased variance (divide by `n`), epsilon `1e-5`, and a gain and bias.
- `gelu` is the GPT-2 tanh approximation.
- `slice-rows`, `slice-cols`, `split-last`, and `tensor-reshape` cut and regroup axes for later `q`, `k`, `v`, and heads. Slices copy. Reshape shares the `flvector`.

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
